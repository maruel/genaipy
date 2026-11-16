// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for worker readiness, startup failures and process ownership.

package worker

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"slices"
	"strings"
	"testing"
	"time"
)

func TestStart(t *testing.T) {
	t.Parallel()
	accept := func([]byte) error { return nil }
	// Sequential: it replaces the default logger before the parallel subtests
	// run.
	t.Run("stderr lines", func(t *testing.T) { //nolint:paralleltest // slog.SetDefault is global.
		records := make(chan slog.Record, 16)
		prev := slog.Default()
		slog.SetDefault(slog.New(&chanHandler{records: records}))
		t.Cleanup(func() { slog.SetDefault(prev) })
		p, _, err := Start(t.Context(), "test", helper("stderr"), accept)
		if err != nil {
			t.Fatal(err)
		}
		if err := p.Close(); err != nil {
			t.Fatal(err)
		}
		var got []string
		for len(got) < 2 {
			select {
			case r := <-records:
				r.Attrs(func(a slog.Attr) bool {
					if a.Key == "stderr" {
						got = append(got, a.Value.String())
					}
					return true
				})
			case <-time.After(10 * time.Second):
				t.Fatalf("stderr records = %q", got)
			}
		}
		if want := []string{"first line", "second"}; !slices.Equal(got, want) {
			t.Errorf("stderr records = %q, want %q", got, want)
		}
	})
	t.Run("ready", func(t *testing.T) {
		t.Parallel()
		var got string
		// The launcher runs the worker as a grandchild, as uv run does: Close
		// returns only once the grandchild released stdout.
		p, u, err := Start(t.Context(), "test", helper("launcher"), func(line []byte) error {
			got = string(line)
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		if u != "http://127.0.0.1:1" || !strings.Contains(got, `"extra":1`) {
			t.Errorf("url = %q, line = %q", u, got)
		}
		if err := p.Close(); err != nil {
			t.Fatal(err)
		}
		<-p.Done()
		if err := p.Err(); err != nil {
			t.Errorf("requested stop reported %v", err)
		}
		if err := p.Close(); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("exit after ready", func(t *testing.T) {
		t.Parallel()
		p, _, err := Start(t.Context(), "test", helper("crash"), accept)
		if err != nil {
			t.Fatal(err)
		}
		<-p.Done()
		if err := p.Err(); err == nil {
			t.Error("Err did not report the unexpected exit")
		}
		if err := p.Close(); err == nil {
			t.Error("unexpected exit not reported")
		}
	})
	t.Run("startup failure", func(t *testing.T) {
		t.Parallel()
		for _, mode := range []string{"exit", "not-json", "bad-url"} {
			if _, _, err := Start(t.Context(), "test", helper(mode), accept); err == nil {
				t.Errorf("%s accepted", mode)
			}
		}
		if _, _, err := Start(t.Context(), "test", helper("exit"), accept); err == nil || !strings.Contains(err.Error(), "exit status 3") {
			t.Errorf("exit status lost: %v", err)
		}
		want := errors.New("rejected")
		if _, _, err := Start(t.Context(), "test", helper("serve"), func([]byte) error { return want }); !errors.Is(err, want) {
			t.Errorf("ready error = %v", err)
		}
		bad := func(ctx context.Context) *exec.Cmd { return exec.CommandContext(ctx, "/nonexistent") }
		// exec.Cmd.Start closes the stdin pipe itself on failure.
		if _, _, err := Start(t.Context(), "test", bad, accept); err == nil || errors.Is(err, os.ErrClosed) {
			t.Errorf("missing executable error = %v", err)
		}
	})
	t.Run("startup cancellation", func(t *testing.T) {
		t.Parallel()
		ctx, cancel := context.WithTimeout(t.Context(), 100*time.Millisecond)
		defer cancel()
		start := time.Now()
		if _, _, err := Start(ctx, "test", helper("hang-launcher"), accept); !errors.Is(err, context.DeadlineExceeded) {
			t.Fatal(err)
		}
		if d := time.Since(start); d > 5*time.Second {
			t.Errorf("startup cancellation took %s", d)
		}
	})
}

func TestHelperProcess(t *testing.T) { //nolint:paralleltest // The helper subprocess exits instead of running as a test.
	if os.Getenv("GENAIPY_WORKER_HELPER") != "1" {
		return
	}
	lifeline := func() {
		_, _ = io.Copy(io.Discard, os.Stdin)
		os.Exit(0)
	}
	switch mode := os.Args[len(os.Args)-1]; mode {
	case "serve":
		fmt.Println(`{"url":"http://127.0.0.1:1","extra":1}`)
		lifeline()
	case "stderr":
		for _, s := range []string{"first ", "line\n", "sec", "ond\n"} {
			_, _ = os.Stderr.WriteString(s)
		}
		fmt.Println(`{"url":"http://127.0.0.1:1"}`)
		lifeline()
	case "hang":
		lifeline()
	case "launcher", "hang-launcher":
		child := map[string]string{"launcher": "serve", "hang-launcher": "hang"}[mode]
		cmd := helper(child)(context.Background())
		cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
		if err := cmd.Run(); err != nil {
			os.Exit(1)
		}
	case "crash":
		fmt.Println(`{"url":"http://127.0.0.1:1"}`)
		os.Exit(3)
	case "exit":
		os.Exit(3)
	case "not-json":
		fmt.Println("not json")
		lifeline()
	case "bad-url":
		fmt.Println(`{"url":"http://example.com:80"}`)
		lifeline()
	}
	os.Exit(0)
}

// chanHandler sends every record to records.
type chanHandler struct {
	records chan<- slog.Record
}

func (h *chanHandler) Enabled(context.Context, slog.Level) bool { return true }

func (h *chanHandler) Handle(_ context.Context, r slog.Record) error { //nolint:gocritic // slog.Handler fixes the signature.
	h.records <- r.Clone()
	return nil
}

func (h *chanHandler) WithAttrs([]slog.Attr) slog.Handler { return h }

func (h *chanHandler) WithGroup(string) slog.Handler { return h }

func helper(mode string) func(context.Context) *exec.Cmd {
	return func(ctx context.Context) *exec.Cmd {
		cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestHelperProcess$", "--", mode)
		cmd.Env = append(os.Environ(), "GENAIPY_WORKER_HELPER=1")
		return cmd
	}
}
