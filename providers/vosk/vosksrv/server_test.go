// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Offline tests of the embedded worker behind the vosk client, with a fake
// native recognizer.

package vosksrv

import (
	"context"
	"errors"
	"io"
	"maps"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genaipy/providers/vosk"
)

// fakeVosk recognizes the sample count: an audio message starting with sample
// 1 ends a phrase and one starting with sample -1 fails natively.
const fakeVosk = `import json

class Model:
    def __init__(self, path):
        self.path = path

class KaldiRecognizer:
    def __init__(self, model, rate):
        assert model.path == "model-dir" and rate == 16000
        self.samples = 0
    def AcceptWaveform(self, data):
        if data[:2] == b"\xff\xff":
            raise RuntimeError("native decoding failed")
        self.samples += len(data) // 2
        return data[:2] == b"\x01\x00"
    def Result(self):
        return json.dumps({"text": "phrase %d" % self.samples})
    def PartialResult(self):
        return json.dumps({"partial": "partial %d" % self.samples})
    def FinalResult(self):
        return json.dumps({"text": "final %d" % self.samples})
`

func TestServer(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "vosk.py"), []byte(fakeVosk), 0o600); err != nil {
		t.Fatal(err)
	}
	s, err := start(t.Context(), func(ctx context.Context) *exec.Cmd {
		cmd := exec.CommandContext(ctx, "python3", "-u", "-c", script, "--model", "model-dir")
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
	c, err := vosk.New(http.DefaultClient, s.URL(), vosk.English)
	if err != nil {
		t.Fatal(err)
	}
	t.Run("valid", func(t *testing.T) {
		r, err := c.ConnectRealtime(t.Context(), genai.RealtimeOptionTranscription{})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = r.Close() })
		// The large message uses a 64-bit frame length.
		endpoint := []byte{1, 0, 0, 0}
		for _, pcm := range [][]byte{make([]byte, 4), endpoint, make([]byte, 200000)} {
			if err := r.Send(t.Context(), genai.RealtimeInputContent{Audio: pcm}); err != nil {
				t.Fatal(err)
			}
		}
		if err := r.Send(t.Context(), genai.RealtimeInputContent{Commit: true}); err != nil {
			t.Fatal(err)
		}
		var got []string
		for e, err := range r.Receive(t.Context()) {
			if err != nil {
				if !errors.Is(err, io.EOF) {
					t.Fatal(err)
				}
				break
			}
			got = append(got, string(e.Kind)+":"+e.Content.Text)
		}
		want := []string{
			string(genai.RealtimeEventTranscriptInputInterim) + ":partial 2",
			string(genai.RealtimeEventTranscriptInput) + ":phrase 4",
			string(genai.RealtimeEventTranscriptInputInterim) + ": partial 100004",
			string(genai.RealtimeEventTranscriptInput) + ": final 100004",
			string(genai.RealtimeEventTranscriptInput) + ":phrase 4 final 100004",
			string(genai.RealtimeEventTurnDone) + ":",
		}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("events = %q\nwant %q", got, want)
		}
	})
	t.Run("error", func(t *testing.T) {
		r, err := c.ConnectRealtime(t.Context(), genai.RealtimeOptionTranscription{})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = r.Close() })
		if err := r.Send(t.Context(), genai.RealtimeInputContent{Audio: []byte{0xff, 0xff}}); err != nil {
			t.Fatal(err)
		}
		for _, err := range r.Receive(t.Context()) {
			if err == nil || !strings.Contains(err.Error(), "native decoding failed") {
				t.Fatalf("error = %v", err)
			}
			break
		}
	})
}

func TestDigests(t *testing.T) {
	if got := slices.Sorted(maps.Keys(digests)); !slices.Equal(got, vosk.Models) {
		t.Errorf("digests cover %q, want %q", got, vosk.Models)
	}
}
