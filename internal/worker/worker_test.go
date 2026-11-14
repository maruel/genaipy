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
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"
)

func TestStart(t *testing.T) {
	t.Parallel()
	accept := func([]byte) error { return nil }
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
		<-p.done
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
		if _, _, err := Start(t.Context(), "test", bad, accept); err == nil {
			t.Error("missing executable accepted")
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

func helper(mode string) func(context.Context) *exec.Cmd {
	return func(ctx context.Context) *exec.Cmd {
		cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestHelperProcess$", "--", mode)
		cmd.Env = append(os.Environ(), "GENAIPY_WORKER_HELPER=1")
		return cmd
	}
}
