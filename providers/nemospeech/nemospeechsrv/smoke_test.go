// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Opt-in validation with the real NeMo-Speech.cpp release and caller-provided audio.

package nemospeechsrv

import (
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genaipy/providers/nemospeech"
)

func TestServerSmoke(t *testing.T) {
	path := os.Getenv("GENAIPY_NEMOSPEECH_SMOKE_PCM")
	if path == "" {
		t.Skip("set GENAIPY_NEMOSPEECH_SMOKE_PCM to a 16 kHz mono S16LE speech clip")
	}
	pcm, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	model := nemospeech.Model(os.Getenv("GENAIPY_NEMOSPEECH_SMOKE_MODEL"))
	if model == "" {
		model = nemospeech.ParakeetTDT
	}
	cache, err := os.UserCacheDir()
	if err != nil {
		t.Fatal(err)
	}
	exe, err := Download(t.Context(), filepath.Join(cache, "genaipy", "nemospeech"), "")
	if err != nil {
		t.Fatal(err)
	}
	s, err := New(t.Context(), exe, model)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := s.Close(); err != nil {
			t.Error(err)
		}
	})
	c, err := nemospeech.New(http.DefaultClient, s.URL(), model)
	if err != nil {
		t.Fatal(err)
	}
	for _, lang := range []string{"", "en"} {
		start := time.Now()
		final, drafts := stream(t, c, pcm, lang)
		if final == "" {
			t.Fatal("empty transcript for speech clip")
		}
		t.Logf("language %q, %s, %d drafts: %s", lang, time.Since(start), drafts, final)
	}
}

// stream sends pcm in 100 ms frames, as a microphone delivers them, and
// returns the final transcript and the number of drafts before it.
func stream(t *testing.T, c *nemospeech.Client, pcm []byte, lang string) (string, int) {
	opts := []genai.RealtimeOption{genai.RealtimeOptionTranscription{}}
	if lang != "" {
		opts = append(opts, &genai.RealtimeOptionAudio{InputLanguage: lang})
	}
	rt, err := c.ConnectRealtime(t.Context(), opts...)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := rt.Close(); err != nil {
			t.Error(err)
		}
	})
	for chunk := range slices.Chunk(pcm, 3200) {
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
		if e.Kind == genai.RealtimeEventTranscriptInput && e.Content.Final {
			final = e.Content.Text
		} else if e.Kind != genai.RealtimeEventTurnDone {
			drafts++
		}
		if e.Kind == genai.RealtimeEventTurnDone {
			break
		}
	}
	return final, drafts
}
