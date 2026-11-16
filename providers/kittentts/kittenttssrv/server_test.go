// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the KittenTTS worker command, configuration and readiness.

package kittenttssrv

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/maruel/genaipy/providers/kittentts"
)

func TestServer(t *testing.T) {
	t.Parallel()
	cfg, _, err := normalizeConfig(Config{CacheDir: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	t.Run("ready", func(t *testing.T) {
		t.Parallel()
		s, err := start(t.Context(), cfg, helper("ready"))
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := s.Close(); err != nil {
				t.Error(err)
			}
		})
		c, err := kittentts.New(&http.Client{}, s.URL())
		if err != nil {
			t.Fatal(err)
		}
		var pcm []byte
		for b, err := range c.Synthesize(t.Context(), kittentts.Voice{}, "hello") {
			if err != nil {
				t.Fatal(err)
			}
			pcm = append(pcm, b...)
		}
		if !bytes.Equal(pcm, []byte{1, 2}) {
			t.Errorf("pcm = %v", pcm)
		}
	})
	t.Run("readiness", func(t *testing.T) {
		t.Parallel()
		for mode, want := range map[string]string{"startup-error": "missing model", "bad-rate": "sample rate", "bad-kind": "kind", "missing-voice": "KittenML/kitten-tts-2 has no voice AirportAnnouncer"} {
			if _, err := start(t.Context(), cfg, helper(mode)); err == nil || !strings.Contains(err.Error(), want) {
				t.Errorf("%s: %v", mode, err)
			}
		}
	})
}

func TestCommand(t *testing.T) { //nolint:paralleltest // t.Setenv redirects os.UserCacheDir.
	want := isolatedCacheDir(t)
	if _, _, err := normalizeConfig(Config{CacheDir: "relative"}); err == nil {
		t.Error("relative cache directory accepted")
	}
	if _, _, err := normalizeConfig(Config{Model: "KittenML/kitten-tts-mini-0.8"}); err == nil {
		t.Error("model without a voice catalog accepted")
	}
	for _, model := range []string{"", "KittenML/kitten-tts-2-nano"} {
		cfg, revision, err := normalizeConfig(Config{Model: model})
		if err != nil {
			t.Fatal(err)
		}
		cmd := command(t.Context(), "/opt/uv", cfg, revision)
		if cmd.Path != "/opt/uv" {
			t.Errorf("cmd.Path = %q, want /opt/uv", cmd.Path)
		}
		if cmd.Dir != want {
			t.Errorf("cmd.Dir = %q, want %q", cmd.Dir, want)
		}
		wantModel := cmp.Or(model, DefaultModel)
		wantRevision, err := kittentts.Revision(wantModel)
		if err != nil {
			t.Fatal(err)
		}
		if n := len(cmd.Args); n < 4 || !slices.Equal(cmd.Args[n-4:], []string{"--model", wantModel, "--revision", wantRevision}) {
			t.Errorf("model and revision arguments missing: %v", cmd.Args)
		}
		// kittenml reads its voices JSON with the default encoding.
		if !slices.Contains(cmd.Env, "PYTHONUTF8=1") {
			t.Error("UTF-8 mode not enabled")
		}
	}
}

func isolatedCacheDir(t *testing.T) string {
	base := t.TempDir()
	switch runtime.GOOS {
	case "darwin":
		t.Setenv("HOME", base)
		return filepath.Join(base, "Library", "Caches", "genaipy", "kittentts")
	case "windows":
		t.Setenv("LOCALAPPDATA", base)
		return filepath.Join(base, "genaipy", "kittentts")
	default:
		t.Setenv("XDG_CACHE_HOME", base)
		return filepath.Join(base, "genaipy", "kittentts")
	}
}

func TestHelperProcess(t *testing.T) { //nolint:paralleltest // The helper subprocess exits instead of running as a test.
	if os.Getenv("GENAIPY_KITTEN_TTS_HELPER") != "1" {
		return
	}
	switch mode := os.Args[len(os.Args)-1]; mode {
	case "ready":
		serve()
	case "bad-rate":
		fmt.Println(`{"kind":"ready","url":"http://127.0.0.1:1","sample_rate":16000}`)
	case "bad-kind":
		fmt.Println(`{"kind":"other","url":"http://127.0.0.1:1","sample_rate":24000}`)
	case "missing-voice":
		fmt.Println(`{"kind":"ready","url":"http://127.0.0.1:1","voices":["Jasper"],"sample_rate":24000}`)
	case "startup-error":
		fmt.Println(`{"kind":"error","error":"missing model"}`)
	}
	// Mirror worker.py: exit once the server closes stdin.
	_, _ = io.Copy(io.Discard, os.Stdin)
	os.Exit(0)
}

// serve mirrors worker.py's readiness and streamed response.
func serve() {
	go func() {
		_, _ = io.Copy(io.Discard, os.Stdin)
		os.Exit(0)
	}()
	ln, err := (&net.ListenConfig{}).Listen(context.Background(), "tcp", "127.0.0.1:0")
	if err != nil {
		os.Exit(1)
	}
	voices, err := kittentts.Voices(DefaultModel)
	if err != nil {
		os.Exit(1)
	}
	names := make([]string, len(voices))
	for i, v := range voices {
		names[i] = v.Name
	}
	ready, err := json.Marshal(map[string]any{"kind": "ready", "url": "http://" + ln.Addr().String(), "voices": names, "sample_rate": kittentts.SampleRate})
	if err != nil {
		os.Exit(1)
	}
	fmt.Println(string(ready))
	srv := &http.Server{ReadHeaderTimeout: time.Second, Handler: http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "audio/pcm")
		_, _ = w.Write([]byte{1, 2})
	})}
	_ = srv.Serve(ln)
	os.Exit(1)
}

func helper(mode string) func(context.Context) *exec.Cmd {
	return func(ctx context.Context) *exec.Cmd {
		cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestHelperProcess$", "--", mode)
		cmd.Env = append(os.Environ(), "GENAIPY_KITTEN_TTS_HELPER=1")
		return cmd
	}
}
