// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Opt-in validation with real Kyutai STT models and caller-provided audio.

package kyutaisrv

import (
	"net/http"
	"os"
	"slices"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genaipy/providers/kyutai"
)

// TestServerSmoke transcribes the clip with each model, or with
// GENAIPY_KYUTAI_SMOKE_MODEL, and logs the final transcript.
func TestServerSmoke(t *testing.T) {
	path := os.Getenv("GENAIPY_KYUTAI_SMOKE_PCM")
	if path == "" {
		t.Skip("set GENAIPY_KYUTAI_SMOKE_PCM to a 24 kHz mono S16LE speech clip")
	}
	pcm, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	models := kyutai.Models
	if m := os.Getenv("GENAIPY_KYUTAI_SMOKE_MODEL"); m != "" {
		models = []kyutai.Model{kyutai.Model(m)}
	}
	for _, model := range models {
		t.Run(string(model), func(t *testing.T) {
			s, err := New(t.Context(), Config{Model: model})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := s.Close(); err != nil {
					t.Error(err)
				}
			})
			c, err := kyutai.New(http.DefaultClient, s.URL(), model, "")
			if err != nil {
				t.Fatal(err)
			}
			final, drafts := transcribe(t, c, pcm)
			if final == "" {
				t.Fatal("empty transcript for speech clip")
			}
			t.Logf("%d drafts: %s", drafts, final)
		})
	}
}

// transcribe sends pcm in 100 ms chunks and returns the final transcript and
// the number of drafts before it.
func transcribe(t *testing.T, c *kyutai.Client, pcm []byte) (string, int) {
	rt, err := c.ConnectRealtime(t.Context(), genai.RealtimeOptionTranscription{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := rt.Close(); err != nil {
			t.Error(err)
		}
	})
	for chunk := range slices.Chunk(pcm, kyutai.SampleRate/10*2) {
		if err := rt.Send(t.Context(), genai.RealtimeInputContent{Audio: chunk}); err != nil {
			t.Fatal(err)
		}
	}
	if err := rt.Send(t.Context(), genai.RealtimeInputContent{Commit: true}); err != nil {
		t.Fatal(err)
	}
	drafts, final := 0, ""
	for e, err := range rt.Receive(t.Context()) {
		if err != nil {
			t.Fatal(err)
		}
		switch {
		case e.Kind == genai.RealtimeEventTurnDone:
			return final, drafts
		case e.Kind == genai.RealtimeEventTranscriptInput && e.Content.Final:
			final = e.Content.Text
		default:
			drafts++
		}
	}
	return final, drafts
}
