// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Offline tests of the Whistle worker lifecycle and ASR windowing.
package whistle

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestWhistle(t *testing.T) {
	dir := t.TempDir()
	err := os.WriteFile(filepath.Join(dir, "needle.py"), []byte(`import time
class Whistle:
    def __init__(self, weights=None):
        pass
    def transcribe(self, audio):
        if audio[0] == -1:
            time.sleep(0.2)
        return {"text": str(len(audio))}
`), 0o600)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	cmd := exec.CommandContext(ctx, "python3", "-u", "-c", whistleWorker)
	cmd.Env = append(os.Environ(), "PYTHONPATH="+dir)
	w, err := launchWhistle(ctx, cancel, cmd)
	if err != nil {
		cancel()
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := w.Close(); err != nil {
			t.Error(err)
		}
	})
	t.Run("long utterance", func(t *testing.T) {
		text, err := w.Transcribe(t.Context(), make([]byte, 16000*2*31), 16000)
		if err != nil || text != "480000 16000" {
			t.Fatalf("got %q, %v", text, err)
		}
	})
	t.Run("invalid PCM", func(t *testing.T) {
		if _, err := w.Transcribe(t.Context(), []byte{1}, 16000); err == nil {
			t.Fatal("accepted odd PCM")
		}
	})
	t.Run("cancellation and reuse", func(t *testing.T) {
		c, done := context.WithTimeout(t.Context(), 50*time.Millisecond)
		defer done()
		if _, err := w.Transcribe(c, []byte{0, 128}, 16000); !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("got %v", err)
		}
		text, err := w.Transcribe(t.Context(), []byte{0, 0}, 16000)
		if err != nil || text != "1" {
			t.Fatalf("got %q,%v", text, err)
		}
	})
	cancel()
	if _, err := w.Transcribe(t.Context(), []byte{0, 0}, 16000); !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v", err)
	}
}

func TestWhistleReadinessFailure(t *testing.T) {
	for _, script := range []string{`print('not json')`, `print('{"url":"http://example.com:80"}')`, `raise SystemExit(3)`} {
		t.Run(strings.ReplaceAll(script, "/", "_"), func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			cmd := exec.CommandContext(ctx, "python3", "-u", "-c", script)
			if _, err := launchWhistle(ctx, cancel, cmd); err == nil {
				t.Fatal("accepted failed worker")
			}
		})
	}
}

func TestWhistleStartupCancellation(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "needle.py"), []byte("import time\nclass Whistle:\n    def __init__(self, weights=None):\n        time.sleep(2)\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	// Mimic uv: its Python child inherits the stdout and stdin lifeline. Killing
	// only the launcher must not leave model startup holding these pipes open.
	ctx, cancel := context.WithTimeout(t.Context(), 100*time.Millisecond)
	defer cancel()
	cmd := exec.CommandContext(ctx, "python3", "-u", "-c", "import subprocess, sys; subprocess.run([sys.executable, '-u', '-c', sys.argv[1]], check=True)", whistleWorker)
	cmd.Env = append(os.Environ(), "PYTHONPATH="+dir)
	start := time.Now()
	if _, err := launchWhistle(ctx, cancel, cmd); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("got %v", err)
	}
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Fatalf("startup cancellation left child alive for %s", elapsed)
	}
}
