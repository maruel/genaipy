// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Offline launcher, process ownership, current genai integration, and opt-in model smoke tests.

package genaipy

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers/openaicompatible"
)

// TestMain also acts as a fake Python executable in a prepared virtualenv. This
// exercises the public launcher without Python packages or model downloads.
func TestMain(m *testing.M) {
	mode := os.Getenv("GENAIPY_PROCESS_FIXTURE")
	if mode == "" {
		os.Exit(m.Run())
	}
	if mode == "exit" {
		os.Exit(7)
	}
	var port string
	for i, arg := range os.Args {
		if arg == "--port" {
			port = os.Args[i+1]
		}
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		switch mode {
		case "unhealthy":
			w.WriteHeader(http.StatusServiceUnavailable)
		case "hang":
			<-r.Context().Done()
		case "wrong-pid":
			_, _ = io.WriteString(w, `{"status":"ok","pid":1}`)
		default:
			_, _ = fmt.Fprintf(w, `{"status":"ok","pid":%d}`, os.Getpid())
		}
	})
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"Hello"}}]}`)
	})
	mux.HandleFunc("/exit", func(http.ResponseWriter, *http.Request) { os.Exit(7) })
	server := &http.Server{Addr: "127.0.0.1:" + port, Handler: mux, ReadHeaderTimeout: time.Second}
	if err := server.ListenAndServe(); err != nil {
		os.Exit(8)
	}
	os.Exit(0)
}

func preparedCache(t *testing.T) string {
	cache := t.TempDir()
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	bin, name := "bin", "python3"
	if runtime.GOOS == "windows" {
		bin, name = "Scripts", "python.exe"
	}
	dir := filepath.Join(cache, "venv", bin)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	src, err := os.Open(exe)
	if err != nil {
		t.Fatal(err)
	}
	dst, err := os.OpenFile(filepath.Join(dir, name), os.O_CREATE|os.O_WRONLY, 0o755)
	if err != nil {
		t.Fatal(err)
	}
	_, copyErr := io.Copy(dst, src)
	if err := errors.Join(copyErr, dst.Close(), src.Close()); err != nil {
		t.Fatal(err)
	}
	files := map[string][]byte{
		"llm.py": llmPy, "image_gen.py": imageGenPy, "setup.sh": setupSh, "setup.bat": setupBat,
		"requirements.txt": requirementsTxt,
		"venv/pyvenv.cfg":  []byte("fixture"),
		".setup-complete":  nil,
	}
	for name, data := range files {
		if err := os.WriteFile(filepath.Join(cache, name), data, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return cache
}

func TestServer(t *testing.T) {
	t.Run("Close", func(t *testing.T) {
		t.Setenv("GENAIPY_PROCESS_FIXTURE", "healthy")
		cache := preparedCache(t)
		ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
		t.Cleanup(cancel)
		srv, err := NewServer(ctx, "llm.py", cache, filepath.Join(cache, "server.log"), nil)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := srv.Close(); err != nil {
				t.Error(err)
			}
		})
		client, err := openaicompatible.New(ctx, genai.ProviderOptionRemote(srv.URL+"/v1/chat/completions"))
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := client.Close(); err != nil {
				t.Error(err)
			}
		})
		res, err := client.GenSync(ctx, genai.Messages{genai.NewTextMessage("Say hello")})
		if err != nil {
			t.Fatal(err)
		}
		if res.String() != "Hello" {
			t.Fatalf("got %q", res.String())
		}
	})
	t.Run("Done", func(t *testing.T) {
		t.Setenv("GENAIPY_PROCESS_FIXTURE", "healthy")
		cache := preparedCache(t)
		srv, err := NewServer(t.Context(), "image_gen.py", cache, filepath.Join(cache, "server.log"), nil)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = srv.Close() })
		req, err := http.NewRequestWithContext(t.Context(), http.MethodGet, srv.URL+"/exit", http.NoBody)
		if err != nil {
			t.Fatal(err)
		}
		resp, err := http.DefaultClient.Do(req)
		if err == nil {
			if err := resp.Body.Close(); err != nil {
				t.Error(err)
			}
		}
		select {
		case err := <-srv.Done():
			if err == nil {
				t.Fatal("unexpected exit succeeded")
			}
		case <-time.After(5 * time.Second):
			t.Fatal("process was not reaped")
		}
		if err := srv.Close(); err == nil {
			t.Fatal("Close lost the error after Done was read")
		}
	})
}

func TestNewServer(t *testing.T) {
	t.Run("error", func(t *testing.T) {
		for _, mode := range []string{"exit", "hang", "unhealthy", "wrong-pid"} {
			t.Run(mode, func(t *testing.T) {
				t.Setenv("GENAIPY_PROCESS_FIXTURE", mode)
				cache := preparedCache(t)
				ctx, cancel := context.WithTimeout(t.Context(), 350*time.Millisecond)
				t.Cleanup(cancel)
				srv, err := NewServer(ctx, "llm.py", cache, filepath.Join(cache, "server.log"), nil)
				if srv != nil || err == nil {
					t.Fatalf("got %v, %v", srv, err)
				}
				if mode != "exit" && !errors.Is(err, context.DeadlineExceeded) {
					t.Fatalf("got %v", err)
				}
				// The child's inherited log descriptor must be closed by failed startup.
				if err := os.Remove(filepath.Join(cache, "server.log")); err != nil {
					t.Fatal(err)
				}
			})
		}
		t.Run("arguments", func(t *testing.T) {
			for _, arg := range []string{"--host=0.0.0.0", "--port", "--prompt=test"} {
				_, err := NewServer(t.Context(), "llm.py", t.TempDir(), filepath.Join(t.TempDir(), "server.log"), []string{arg})
				if err == nil {
					t.Fatalf("accepted %q", arg)
				}
			}
			for _, script := range []string{"", "../llm.py", "other.py"} {
				_, err := NewServer(t.Context(), script, t.TempDir(), filepath.Join(t.TempDir(), "server.log"), nil)
				if err == nil {
					t.Fatalf("accepted %q", script)
				}
			}
			if _, err := NewServer(t.Context(), "llm.py", "relative", "/server.log", nil); err == nil {
				t.Fatal("accepted relative cache")
			}
			if _, err := NewServer(t.Context(), "llm.py", t.TempDir(), "relative", nil); err == nil {
				t.Fatal("accepted relative log")
			}
		})
	})
	t.Run("cancellation", func(t *testing.T) {
		t.Setenv("GENAIPY_PROCESS_FIXTURE", "healthy")
		cache := preparedCache(t)
		ctx, cancel := context.WithCancel(t.Context())
		t.Cleanup(cancel)
		srv, err := NewServer(ctx, "image_gen.py", cache, filepath.Join(cache, "server.log"), nil)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := srv.Close(); err != nil {
				t.Error(err)
			}
		})
		cancel()
		select {
		case err := <-srv.Done():
			if err != nil {
				t.Fatal(err)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("cancellation did not reap process")
		}
	})
}

func TestNeedRecreate(t *testing.T) {
	cache := preparedCache(t)
	if needRecreate(cache) {
		t.Fatal("prepared cache considered stale")
	}
	if err := os.WriteFile(filepath.Join(cache, "llm.py"), []byte("outdated"), 0o644); err != nil {
		t.Fatal(err)
	}
	if !needRecreate(cache) {
		t.Fatal("changed runtime considered current")
	}
}

func TestRecreate(t *testing.T) {
	cache := preparedCache(t)
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if err := recreate(ctx, cache); !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v", err)
	}
	if _, err := os.Stat(filepath.Join(cache, "venv", "pyvenv.cfg")); err != nil {
		t.Fatal(err)
	}
	if !needRecreate(cache) {
		t.Fatal("failed setup left a reusable environment")
	}
}

func TestModelSmoke(t *testing.T) {
	if os.Getenv("GENAIPY_MODEL_SMOKE") != "1" {
		t.Skip("set GENAIPY_MODEL_SMOKE=1 to install Python dependencies and download a real model")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 15*time.Minute)
	t.Cleanup(cancel)
	cache := os.Getenv("GENAIPY_MODEL_CACHE")
	if cache == "" {
		cache = t.TempDir()
	}
	cache, err := filepath.Abs(cache)
	if err != nil {
		t.Fatal(err)
	}
	srv, err := NewServer(ctx, "llm.py", cache, filepath.Join(cache, "server.log"), []string{"--model", "HuggingFaceTB/SmolLM2-135M-Instruct"})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := srv.Close(); err != nil {
			t.Error(err)
		}
	})
	client, err := openaicompatible.New(ctx, genai.ProviderOptionRemote(srv.URL+"/v1/chat/completions"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := client.Close(); err != nil {
			t.Error(err)
		}
	})
	res, err := client.GenSync(ctx, genai.Messages{genai.NewTextMessage("Say hello. Reply with only one word.")})
	if err != nil {
		t.Fatal(err)
	}
	if strings.TrimSpace(res.String()) == "" {
		t.Fatal("empty model response")
	}
}
