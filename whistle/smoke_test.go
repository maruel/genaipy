// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Opt-in validation with real Whistle dependencies and caller-provided audio.
package whistle

import (
	"os"
	"testing"
	"time"
)

func TestWhistleSmoke(t *testing.T) {
	path := os.Getenv("GENAIPY_WHISTLE_SMOKE_PCM")
	if path == "" {
		t.Skip("set GENAIPY_WHISTLE_SMOKE_PCM to a 16 kHz mono S16LE speech clip")
	}
	pcm, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	w, err := New(t.Context(), "")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := w.Close(); err != nil {
			t.Error(err)
		}
	})
	for range 2 {
		start := time.Now()
		text, err := w.Transcribe(t.Context(), pcm, 16000)
		if err != nil {
			t.Fatal(err)
		}
		if text == "" {
			t.Fatal("empty transcript for speech clip")
		}
		t.Logf("%s: %s", time.Since(start), text)
	}
}
