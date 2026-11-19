// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Opt-in validation with the real vosk package, a downloaded model and caller-provided audio.

package vosksrv

import (
	"errors"
	"io"
	"net/http"
	"os"
	"testing"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genaipy/providers/vosk"
)

func TestVoskSmoke(t *testing.T) {
	path := os.Getenv("GENAIPY_VOSK_SMOKE_PCM")
	if path == "" {
		t.Skip("set GENAIPY_VOSK_SMOKE_PCM to a 16 kHz mono S16LE English speech clip")
	}
	pcm, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	s, err := New(t.Context(), vosk.English)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := s.Close(); err != nil {
			t.Error(err)
		}
	})
	c, err := vosk.New(http.DefaultClient, s.URL(), vosk.English)
	if err != nil {
		t.Fatal(err)
	}
	for range 2 {
		r, err := c.ConnectRealtime(t.Context(), genai.RealtimeOptionTranscription{})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = r.Close() })
		// 20 ms messages, as a microphone sends them.
		start := time.Now()
		go func() {
			for p := pcm; len(p) > 0; p = p[min(len(p), 640):] {
				if err := r.Send(t.Context(), genai.RealtimeInputContent{Audio: p[:min(len(p), 640)]}); err != nil {
					t.Error(err)
					return
				}
			}
			if err := r.Send(t.Context(), genai.RealtimeInputContent{Commit: true}); err != nil {
				t.Error(err)
			}
		}()
		drafts, text := 0, ""
		for e, err := range r.Receive(t.Context()) {
			if err != nil {
				if !errors.Is(err, io.EOF) {
					t.Fatal(err)
				}
				break
			}
			switch {
			case e.Kind == genai.RealtimeEventTranscriptInput && e.Content.Final:
				text = e.Content.Text
			case e.Kind == genai.RealtimeEventTranscriptInputInterim:
				drafts++
			}
		}
		if text == "" {
			t.Fatal("empty transcript for speech clip")
		}
		t.Logf("%s, %d drafts: %s", time.Since(start), drafts, text)
	}
}
