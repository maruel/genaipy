// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Offline tests of the embedded worker behind the whistle client, with a fake
// Needle engine.

package whistlesrv

import (
	"context"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genaipy/providers/whistle"
)

func TestServer(t *testing.T) {
	dir := t.TempDir()
	err := os.WriteFile(filepath.Join(dir, "needle.py"), []byte(`class Whistle:
    def __init__(self, weights=None):
        pass
    def transcribe(self, audio):
        return {"text": str(len(audio))}
`), 0o600)
	if err != nil {
		t.Fatal(err)
	}
	s, err := start(t.Context(), func(ctx context.Context) *exec.Cmd {
		cmd := exec.CommandContext(ctx, "python3", "-u", "-c", script)
		cmd.Env = append(os.Environ(), "PYTHONPATH="+dir)
		return cmd
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := s.Close(); err != nil {
			t.Error(err)
		}
	})
	if text := transcribe(t, s, make([]byte, 4)); text != "2" {
		t.Errorf("transcript = %q", text)
	}
}

// transcribe recognizes pcm in one session through the whistle client.
func transcribe(t *testing.T, s *Server, pcm []byte) string {
	c, err := whistle.New(&http.Client{}, s.URL(), "")
	if err != nil {
		t.Fatal(err)
	}
	r, err := c.ConnectRealtime(t.Context(), genai.RealtimeOptionTranscription{})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = r.Close() }()
	if err := r.Send(t.Context(), genai.RealtimeInputContent{Audio: pcm, Commit: true}); err != nil {
		t.Fatal(err)
	}
	text := ""
	for e, err := range r.Receive(t.Context()) {
		if err != nil {
			t.Fatal(err)
		}
		if e.Content.Final {
			text = e.Content.Text
		}
		if e.Kind == genai.RealtimeEventTurnDone {
			break
		}
	}
	return text
}
