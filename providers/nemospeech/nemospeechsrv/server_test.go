// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Offline server lifecycle tests against a fake nemo-speech executable.

package nemospeechsrv

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genaipy/providers/nemospeech"
)

// TestMain turns the test binary into a fake "nemo-speech serve" when
// NEMOSPEECH_FAKE selects a behavior.
func TestMain(m *testing.M) {
	if mode := os.Getenv("NEMOSPEECH_FAKE"); mode != "" {
		os.Exit(fakeServe(mode, os.Args[1:]))
	}
	os.Exit(m.Run())
}

func fakeServe(mode string, args []string) int {
	if len(args) == 0 || args[0] != "serve" {
		fmt.Fprintf(os.Stderr, "unexpected arguments %q\n", args)
		return 2
	}
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	model := fs.String("asr-model", "", "")
	host := fs.String("host", "", "")
	port := fs.String("port", "", "")
	fs.Bool("no-ui", false, "")
	if err := fs.Parse(args[1:]); err != nil {
		return 2
	}
	if mode == "fail" {
		fmt.Fprintln(os.Stderr, "error: cannot load model")
		return 3
	}
	if os.Getenv("NEMO_SPEECH_HTTP_API_KEY") != "" {
		fmt.Fprintln(os.Stderr, "inherited listener setting")
		return 4
	}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /ready", func(w http.ResponseWriter, _ *http.Request) {
		if mode == "never-ready" {
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = w.Write([]byte(`{"ready":false}`))
			return
		}
		_, _ = w.Write([]byte(`{"capabilities":["asr"],"ready":true}`))
	})
	mux.HandleFunc("POST /v1/audio/transcriptions", func(w http.ResponseWriter, r *http.Request) {
		form, err := r.MultipartReader()
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		lang := ""
		for part, err := form.NextPart(); err == nil; part, err = form.NextPart() {
			if part.FormName() == "language" {
				b, _ := io.ReadAll(io.LimitReader(part, 64))
				lang = string(b)
			}
		}
		_ = json.NewEncoder(w).Encode(map[string]string{"text": *model + " " + lang})
	})
	srv := &http.Server{Addr: *host + ":" + *port, Handler: mux, ReadHeaderTimeout: time.Second}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	go func() {
		<-ctx.Done()
		_ = srv.Shutdown(context.WithoutCancel(ctx))
	}()
	if err := srv.ListenAndServe(); !errors.Is(err, http.ErrServerClosed) {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	return 0
}

func TestDownload(t *testing.T) {
	t.Run("error", func(t *testing.T) {
		if _, err := Download(t.Context(), "relative", ""); err == nil {
			t.Error("relative cache accepted")
		}
		if _, err := Download(t.Context(), t.TempDir(), "tpu"); err == nil {
			t.Error("unknown backend accepted")
		}
	})
}

func TestServer(t *testing.T) {
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	t.Run("valid", func(t *testing.T) {
		t.Setenv("NEMOSPEECH_FAKE", "serve")
		t.Setenv("NEMO_SPEECH_HTTP_API_KEY", "for another server")
		s, err := New(t.Context(), exe, nemospeech.ParakeetTDT)
		if err != nil {
			t.Fatal(err)
		}
		c, err := nemospeech.New(http.DefaultClient, s.URL(), nemospeech.ParakeetTDT)
		if err != nil {
			t.Fatal(err)
		}
		// The fake echoes the served model and the request language, which
		// Parakeet TDT never sends.
		if text, err := transcribe(t, c); err != nil || text != "parakeet-tdt" {
			t.Errorf("transcribe = %q, %v", text, err)
		}
		if err := s.Close(); err != nil {
			t.Fatalf("Close = %v", err)
		}
		if _, err := transcribe(t, c); err == nil {
			t.Error("transcribed after Close")
		}
	})
	t.Run("error", func(t *testing.T) {
		t.Run("exit", func(t *testing.T) {
			t.Setenv("NEMOSPEECH_FAKE", "fail")
			if _, err := New(t.Context(), exe, nemospeech.ParakeetTDT); err == nil {
				t.Fatal("startup failure accepted")
			}
		})
		t.Run("cancelled before ready", func(t *testing.T) {
			t.Setenv("NEMOSPEECH_FAKE", "never-ready")
			ctx, cancel := context.WithTimeout(t.Context(), 500*time.Millisecond)
			t.Cleanup(cancel)
			if _, err := New(ctx, exe, nemospeech.ParakeetTDT); !errors.Is(err, context.DeadlineExceeded) {
				t.Fatalf("error = %v", err)
			}
		})
		t.Run("missing executable", func(t *testing.T) {
			if _, err := New(t.Context(), filepath.Join(t.TempDir(), "nemo-speech"), nemospeech.ParakeetTDT); err == nil {
				t.Fatal("missing executable accepted")
			}
		})
	})
}

// transcribe sends one sample in a session and returns the final transcript.
func transcribe(t *testing.T, c *nemospeech.Client) (string, error) {
	s, err := c.ConnectRealtime(t.Context(), genai.RealtimeOptionTranscription{})
	if err != nil {
		return "", err
	}
	defer func() { _ = s.Close() }()
	if err := s.Send(t.Context(), genai.RealtimeInputContent{Audio: []byte{0, 0}, Commit: true}); err != nil {
		return "", err
	}
	text := ""
	for e, err := range s.Receive(t.Context()) {
		if err != nil {
			return "", err
		}
		if e.Content.Final {
			text = e.Content.Text
		}
		if e.Kind == genai.RealtimeEventTurnDone {
			break
		}
	}
	return text, nil
}

func TestReleaseURL(t *testing.T) {
	t.Parallel()
	u, err := releaseURL("")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(u, "https://github.com/NVIDIA/NeMo-Speech.cpp/releases/download/v"+Version+"/") {
		t.Errorf("releaseURL = %q", u)
	}
}
