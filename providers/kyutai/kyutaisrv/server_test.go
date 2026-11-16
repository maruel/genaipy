// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the Kyutai worker command, configuration and readiness.

package kyutaisrv

import (
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"runtime"
	"slices"
	"strings"
	"testing"

	"github.com/maruel/genaipy/providers/kyutai"
)

func TestServer(t *testing.T) {
	t.Parallel()
	cfg, err := normalizeConfig(Config{CacheDir: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	t.Run("ready", func(t *testing.T) {
		t.Parallel()
		s, err := start(t.Context(), cfg, helper("ready"))
		if err != nil {
			t.Fatal(err)
		}
		if s.URL() != "http://127.0.0.1:1" {
			t.Errorf("URL = %q", s.URL())
		}
		if err := s.Close(); err != nil {
			t.Error(err)
		}
	})
	t.Run("readiness", func(t *testing.T) {
		t.Parallel()
		for mode, want := range map[string]string{"bad-rate": "sample rate", "bad-kind": "kind"} {
			if _, err := start(t.Context(), cfg, helper(mode)); err == nil || !strings.Contains(err.Error(), want) {
				t.Errorf("%s: %v", mode, err)
			}
		}
	})
}

func TestCommand(t *testing.T) {
	t.Parallel()
	if _, err := normalizeConfig(Config{CacheDir: "relative"}); err == nil {
		t.Error("relative cache directory accepted")
	}
	if _, err := normalizeConfig(Config{Model: "kyutai/stt-1b-en_fr-candle"}); err == nil {
		t.Error("model without a checkpoint accepted")
	}
	backend := "torch"
	if runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" {
		backend = "mlx"
	}
	dir := t.TempDir()
	for _, model := range kyutai.Models {
		cfg, err := normalizeConfig(Config{Model: model, CacheDir: dir})
		if err != nil {
			t.Fatal(err)
		}
		cmd := command(t.Context(), "/opt/uv", cfg)
		if cmd.Path != "/opt/uv" || cmd.Dir != dir {
			t.Errorf("cmd = %s in %s", cmd.Path, cmd.Dir)
		}
		// The repository is the model's, or its -mlx variant on Apple silicon.
		i := slices.Index(cmd.Args, "--model")
		if i < 0 || i+5 >= len(cmd.Args) || !strings.HasPrefix(cmd.Args[i+1], string(model)) || cmd.Args[i+2] != "--revision" || len(cmd.Args[i+3]) != 40 || !slices.Equal(cmd.Args[i+4:], []string{"--backend", backend}) {
			t.Errorf("%s: worker arguments %q", model, cmd.Args)
		}
	}
}

func TestHelperProcess(t *testing.T) { //nolint:paralleltest // The helper subprocess exits instead of running as a test.
	if os.Getenv("GENAIPY_KYUTAI_HELPER") != "1" {
		return
	}
	switch mode := os.Args[len(os.Args)-1]; mode {
	case "ready":
		fmt.Println(`{"kind":"ready","url":"http://127.0.0.1:1","sample_rate":24000}`)
	case "bad-rate":
		fmt.Println(`{"kind":"ready","url":"http://127.0.0.1:1","sample_rate":16000}`)
	case "bad-kind":
		fmt.Println(`{"kind":"other","url":"http://127.0.0.1:1","sample_rate":24000}`)
	}
	// Mirror worker.py: exit once the server closes stdin.
	_, _ = io.Copy(io.Discard, os.Stdin)
	os.Exit(0)
}

func helper(mode string) func(context.Context) *exec.Cmd {
	return func(ctx context.Context) *exec.Cmd {
		cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestHelperProcess$", "--", mode)
		cmd.Env = append(os.Environ(), "GENAIPY_KYUTAI_HELPER=1")
		return cmd
	}
}
