// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the KittenTTS subprocess adapter.

package kittentts

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestRuntime(t *testing.T) {
	t.Parallel()
	t.Run("Synthesize", func(t *testing.T) {
		t.Run("configured voice", func(t *testing.T) {
			t.Parallel()
			cfg, err := normalizeConfig(Config{Model: "owner/model", Voice: "Luna"})
			if err != nil {
				t.Fatal(err)
			}
			a, err := newWithCommand(t.Context(), cfg, kittenTTSTestCommand("configured"))
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := a.Close(); err != nil {
					t.Error(err)
				}
			})
			if chunks, err := collectKittenTTS(t.Context(), a, "hello"); err != nil || len(chunks) == 0 {
				t.Fatalf("configured synthesis = %v, %v", chunks, err)
			}
		})

		t.Run("valid", func(t *testing.T) {
			t.Parallel()
			ctx, cancel := kittenTTSTestContext(t)
			t.Cleanup(cancel)
			a, err := newTestRuntime(ctx, kittenTTSTestCommand("valid"))
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := a.Close(); err != nil {
					t.Fatal(err)
				}
			})
			chunks, err := collectKittenTTS(ctx, a, "hello")
			if err != nil {
				t.Fatal(err)
			}
			pcm := bytes.Join(chunks, nil)
			if !bytes.Equal(pcm, []byte{1, 2, 3, 4}) {
				t.Fatalf("pcm = %v, want [1 2 3 4]", pcm)
			}
		})

		t.Run("empty audio", func(t *testing.T) {
			t.Parallel()
			ctx, cancel := kittenTTSTestContext(t)
			t.Cleanup(cancel)
			a, err := newTestRuntime(ctx, kittenTTSTestCommand("empty"))
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := a.Close(); err != nil {
					t.Fatal(err)
				}
			})
			chunks, err := collectKittenTTS(ctx, a, "emoji")
			if err != nil || len(chunks) != 0 {
				t.Fatalf("chunks = %v, error = %v", chunks, err)
			}
		})

		t.Run("concurrent", func(t *testing.T) {
			t.Parallel()
			ctx, cancel := kittenTTSTestContext(t)
			t.Cleanup(cancel)
			a, err := newTestRuntime(ctx, kittenTTSTestCommand("concurrent"))
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := a.Close(); err != nil {
					t.Fatal(err)
				}
			})
			var wg sync.WaitGroup
			errs := make(chan error, 2)
			for i := range 2 {
				wg.Go(func() {
					chunks, err := collectKittenTTS(ctx, a, fmt.Sprintf("hello %d", i))
					if err != nil {
						errs <- err
						return
					}
					pcm := bytes.Join(chunks, nil)
					if !bytes.Equal(pcm, []byte{1, 2, 3, 4}) {
						errs <- fmt.Errorf("pcm = %v, want [1 2 3 4]", pcm)
					}
				})
			}
			wg.Wait()
			close(errs)
			for err := range errs {
				t.Fatal(err)
			}
		})

		t.Run("error", func(t *testing.T) {
			t.Parallel()
			ctx, cancel := kittenTTSTestContext(t)
			t.Cleanup(cancel)
			a, err := newTestRuntime(ctx, kittenTTSTestCommand("error"))
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := a.Close(); err != nil {
					t.Fatal(err)
				}
			})
			_, err = collectKittenTTS(ctx, a, "hello")
			if err == nil || !strings.Contains(err.Error(), "synthesis failed") {
				t.Fatalf("err = %v, want synthesis failed", err)
			}
		})

		t.Run("regressions", testRuntimeSynthesizeRegressions)
	})
	t.Run("Close", func(t *testing.T) {
		t.Parallel()
		ctx, cancel := kittenTTSTestContext(t)
		t.Cleanup(cancel)
		// The launcher runs the worker as a grandchild, as uv run does.
		a, err := newTestRuntime(ctx, kittenTTSTestCommand("launcher"))
		if err != nil {
			t.Fatal(err)
		}
		endpoint := a.baseURL + kittenTTSSynthesize
		if err := a.Close(); err != nil {
			t.Fatal(err)
		}
		for {
			req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, strings.NewReader(`{}`))
			if err != nil {
				t.Fatal(err)
			}
			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				if ctx.Err() != nil {
					t.Fatal("worker still serves after Close")
				}
				return
			}
			if err := resp.Body.Close(); err != nil {
				t.Fatal(err)
			}
			if !sleepCtx(ctx, 10*time.Millisecond) {
				t.Fatal("worker still serves after Close")
			}
		}
	})
}

func testRuntimeStartup(t *testing.T) {
	t.Parallel()
	t.Run("startup error", func(t *testing.T) {
		t.Parallel()
		ctx, cancel := kittenTTSTestContext(t)
		t.Cleanup(cancel)
		_, err := newTestRuntime(ctx, kittenTTSTestCommand("startup-error"))
		if err == nil || !strings.Contains(err.Error(), "missing model") {
			t.Fatalf("err = %v, want missing model", err)
		}
	})
}

func TestKittenTTSCommand(t *testing.T) { //nolint:paralleltest // Uses t.Setenv to validate platform-specific os.UserCacheDir behavior.
	want := isolatedKittenTTSCacheDir(t)

	cfg, err := normalizeConfig(Config{Model: "owner/model"})
	if err != nil {
		t.Fatal(err)
	}
	cmd, err := kittenTTSCommand(t.Context(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	if cmd.Dir != want {
		t.Fatalf("cmd.Dir = %q, want %q", cmd.Dir, want)
	}
	if n := len(cmd.Args); n < 2 || cmd.Args[n-2] != "--model" || cmd.Args[n-1] != "owner/model" {
		t.Fatalf("model argument missing: %v", cmd.Args)
	}
}

func isolatedKittenTTSCacheDir(t *testing.T) string {
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

func TestRuntimeHelperProcess(t *testing.T) { //nolint:paralleltest // helper subprocess exits instead of running as a normal test.
	if os.Getenv("GENAIPY_KITTEN_TTS_HELPER") != "1" {
		t.Parallel()
		return
	}
	args := os.Args
	mode := args[len(args)-1]
	switch mode {
	case "valid", "configured":
		kittenTTSHelperServe(t, false, false, false)
	case "concurrent":
		kittenTTSHelperServe(t, false, true, false)
	case "error":
		kittenTTSHelperServe(t, true, false, false)
	case "empty":
		kittenTTSHelperServe(t, false, false, true)
	case "launcher":
		cmd := exec.CommandContext(t.Context(), os.Args[0], "-test.run=TestRuntimeHelperProcess", "--", "valid")
		cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
		if err := cmd.Run(); err != nil {
			fmt.Fprintf(os.Stderr, "worker: %v\n", err)
			os.Exit(1)
		}
	case "startup-hang":
		<-time.After(time.Hour)
	case "bad-rate":
		fmt.Println(`{"kind":"ready","url":"http://127.0.0.1:1","sample_rate":16000}`)
		<-time.After(time.Hour)
	case "startup-error":
		fmt.Println(`{"kind":"error","error":"missing model"}`)
	default:
		fmt.Printf(`{"kind":"error","error":"unknown mode %s"}`+"\n", mode)
	}
	os.Exit(0)
}

func kittenTTSTestContext(t *testing.T) (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.WithoutCancel(t.Context()), 5*time.Second)
}

func kittenTTSTestCommand(mode string) kittenTTSCommandFactory {
	return func(ctx context.Context) (*exec.Cmd, error) {
		args := []string{"-test.run=TestRuntimeHelperProcess", "--", mode}
		cmd := exec.CommandContext(ctx, os.Args[0], args...)
		cmd.Env = append(os.Environ(), "GENAIPY_KITTEN_TTS_HELPER=1")
		return cmd, nil
	}
}

func kittenTTSHelperServe(t *testing.T, alwaysFail, requireConcurrent, empty bool) {
	// Mirror kittentts.py: exit once the adapter closes stdin.
	go func() {
		_, _ = io.Copy(io.Discard, os.Stdin)
		os.Exit(0)
	}()
	var active atomic.Int32
	var maxActive atomic.Int32
	mux := http.NewServeMux()
	mux.HandleFunc(kittenTTSSynthesize, func(w http.ResponseWriter, r *http.Request) {
		now := active.Add(1)
		for {
			old := maxActive.Load()
			if now <= old || maxActive.CompareAndSwap(old, now) {
				break
			}
		}
		defer active.Add(-1)
		if requireConcurrent {
			time.Sleep(100 * time.Millisecond)
		}
		if alwaysFail {
			writeKittenTTSTestJSON(w, http.StatusInternalServerError, kittenTTSResponse{Error: "synthesis failed"})
			return
		}
		var req kittenTTSRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeKittenTTSTestJSON(w, http.StatusBadRequest, kittenTTSResponse{Error: err.Error()})
			return
		}
		if requireConcurrent && maxActive.Load() < 2 {
			time.Sleep(150 * time.Millisecond)
			if maxActive.Load() < 2 {
				writeKittenTTSTestJSON(w, http.StatusInternalServerError, kittenTTSResponse{Error: "requests were serialized"})
				return
			}
		}
		w.Header().Set("Content-Type", "application/octet-stream")
		w.WriteHeader(http.StatusOK)
		if os.Args[len(os.Args)-1] == "configured" && req.Voice != "Luna" {
			writeKittenTTSTestJSON(w, http.StatusBadRequest, kittenTTSResponse{Error: "configured voice not forwarded"})
			return
		}
		if req.Text == "transport-error" {
			conn, _, err := w.(http.Hijacker).Hijack()
			if err != nil {
				fmt.Fprintf(os.Stderr, "hijack: %v\n", err)
				return
			}
			if err := conn.Close(); err != nil {
				fmt.Fprintf(os.Stderr, "close: %v\n", err)
			}
			return
		}
		if req.Text == "cancel" {
			writeKittenTTSStream(w, []byte{1, 2})
			<-r.Context().Done()
			return
		}
		if empty {
			return
		}
		writeKittenTTSStream(w, []byte{1, 2})
		writeKittenTTSStream(w, []byte{3, 4})
	})
	var lc net.ListenConfig
	ln, err := lc.Listen(t.Context(), "tcp", "127.0.0.1:0")
	if err != nil {
		fmt.Printf(`{"kind":"error","error":%q}`+"\n", err.Error())
		os.Exit(1)
	}
	fmt.Printf(`{"kind":"ready","url":"http://%s","voices":["Jasper"],"sample_rate":24000}`+"\n", ln.Addr().String())
	srv := &http.Server{Handler: mux, ReadHeaderTimeout: time.Second}
	if err := srv.Serve(ln); err != nil && !strings.Contains(err.Error(), "use of closed network connection") {
		fmt.Fprintf(os.Stderr, "serve: %v\n", err)
		os.Exit(1)
	}
}

func writeKittenTTSTestJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		fmt.Fprintf(os.Stderr, "encode response: %v\n", err)
	}
}

func writeKittenTTSStream(w http.ResponseWriter, pcm []byte) {
	if _, err := w.Write(pcm); err != nil {
		fmt.Fprintf(os.Stderr, "write stream response: %v\n", err)
	}
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
}

func collectKittenTTS(ctx context.Context, a *Runtime, text string) ([][]byte, error) {
	var chunks [][]byte
	for pcm, err := range a.Synthesize(ctx, text) {
		if err != nil {
			return nil, err
		}
		chunks = append(chunks, pcm)
	}
	return chunks, nil
}

func newTestRuntime(ctx context.Context, start kittenTTSCommandFactory) (*Runtime, error) {
	cfg, err := normalizeConfig(Config{})
	if err != nil {
		return nil, err
	}
	return newWithCommand(ctx, cfg, start)
}

func testRuntimeSynthesizeRegressions(t *testing.T) {
	t.Parallel()
	t.Run("constructor cancellation and cleanup", func(t *testing.T) {
		t.Parallel()
		ctx, cancel := context.WithCancel(t.Context())
		t.Cleanup(cancel)
		a, err := newTestRuntime(ctx, kittenTTSTestCommand("valid"))
		if err != nil {
			t.Fatal(err)
		}
		cancel()
		<-a.wait
		var got error
		for _, err := range a.Synthesize(t.Context(), "hello") {
			got = err
		}
		if !errors.Is(got, context.Canceled) {
			t.Fatalf("constructor cancellation = %v", got)
		}
		if err := a.Close(); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("closed", func(t *testing.T) {
		t.Parallel()
		a, err := newTestRuntime(t.Context(), kittenTTSTestCommand("valid"))
		if err != nil {
			t.Fatal(err)
		}
		if err := a.Close(); err != nil {
			t.Fatal(err)
		}
		for _, text := range []string{"hello", ""} {
			var got error
			for _, err := range a.Synthesize(t.Context(), text) {
				got = err
			}
			if got == nil || !strings.Contains(got.Error(), "closed") {
				t.Fatalf("error = %v", got)
			}
		}
	})
	t.Run("cancel then reuse", func(t *testing.T) {
		t.Parallel()
		a, err := newTestRuntime(t.Context(), kittenTTSTestCommand("valid"))
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := a.Close(); err != nil {
				t.Error(err)
			}
		})
		ctx, cancel := context.WithCancel(t.Context())
		t.Cleanup(cancel)
		var got error
		for pcm, err := range a.Synthesize(ctx, "cancel") {
			if len(pcm) > 0 {
				cancel()
			}
			if err != nil {
				got = err
			}
		}
		if !errors.Is(got, context.Canceled) {
			t.Fatalf("error = %v", got)
		}
		if chunks, err := collectKittenTTS(t.Context(), a, "hello"); err != nil || len(chunks) == 0 {
			t.Fatalf("reuse = %v, %v", chunks, err)
		}
	})
	t.Run("restart retains constructor lifetime", func(t *testing.T) {
		t.Parallel()
		a, err := newTestRuntime(t.Context(), kittenTTSTestCommand("valid"))
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := a.Close(); err != nil {
				t.Error(err)
			}
		})
		if _, err := collectKittenTTS(t.Context(), a, "transport-error"); err == nil {
			t.Fatal("expected HTTP failure")
		}
		ctx, cancel := context.WithCancel(t.Context())
		t.Cleanup(cancel)
		if chunks, err := collectKittenTTS(ctx, a, "hello"); err != nil || len(chunks) == 0 {
			t.Fatalf("restart = %v, %v", chunks, err)
		}
		cancel()
		if chunks, err := collectKittenTTS(t.Context(), a, "hello"); err != nil || len(chunks) == 0 {
			t.Fatalf("after request cancellation = %v, %v", chunks, err)
		}
	})
	t.Run("cancel startup", func(t *testing.T) {
		t.Parallel()
		ctx, cancel := context.WithTimeout(t.Context(), 100*time.Millisecond)
		t.Cleanup(cancel)
		if _, err := newTestRuntime(ctx, kittenTTSTestCommand("startup-hang")); !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("error = %v", err)
		}
	})
	t.Run("wrong sample rate", func(t *testing.T) {
		t.Parallel()
		if _, err := newTestRuntime(t.Context(), kittenTTSTestCommand("bad-rate")); err == nil || !strings.Contains(err.Error(), "sample rate") {
			t.Fatalf("error = %v", err)
		}
	})
}

func TestNew(t *testing.T) {
	t.Parallel()
	t.Run("error", func(t *testing.T) {
		t.Run("worker startup", testRuntimeStartup)
		if _, err := New(t.Context(), Config{CacheDir: "relative"}); err == nil {
			t.Fatal("relative cache directory accepted")
		}
	})
}

func sleepCtx(ctx context.Context, d time.Duration) bool {
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-timer.C:
		return true
	}
}

type kittenTTSResponse struct {
	Error string `json:"error"`
}
