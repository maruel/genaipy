// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package kyutaisrv runs the Python Kyutai STT worker for the kyutai client.
//
// The worker runs moshi (PyTorch, on CUDA when available) except on Apple
// silicon, where it runs moshi-mlx. It decodes one session at a time; later
// sessions wait for it to end. Neither model transcribes in real time on CPU.
package kyutaisrv

import (
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"

	"github.com/maruel/genai"
	"github.com/maruel/genaipy/internal/uv"
	"github.com/maruel/genaipy/internal/worker"
	"github.com/maruel/genaipy/providers/kyutai"
)

//go:embed worker.py
var script string

const (
	// DefaultModel is the model New loads when Config.Model is empty.
	DefaultModel = kyutai.STT1B
	// The model repositories were published for these releases; the
	// delayed-streams-modeling examples pin moshi 0.2.11 and moshi_mlx 0.2.12.
	moshiPackage    = "moshi==0.2.13"
	moshiMLXPackage = "moshi-mlx==0.3.0"
	// Neither moshi package depends on msgpack; moshi-server's protocol does.
	msgpackPackage = "msgpack==1.2.4"
	pythonVersion  = "3.12"
)

// Config selects the worker's model and model cache.
//
// Zero values use DefaultModel and the user's cache directory under
// genaipy/kyutai. Model must be one of kyutai.Models. CacheDir must be
// absolute when set.
type Config struct {
	Model    kyutai.Model
	CacheDir string
}

var _ genai.Server = (*Server)(nil)

// Server owns a Kyutai STT worker process.
type Server struct {
	p   *worker.Process
	url string
}

// New starts the worker, installing the pinned uv and its pinned Python
// packages if needed. Startup downloads the model at its pinned revision when
// it is absent from the cache. ctx owns the worker lifetime; call Close when
// finished.
func New(ctx context.Context, cfg Config) (*Server, error) {
	cfg, err := normalizeConfig(cfg)
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(cfg.CacheDir, 0o750); err != nil {
		return nil, fmt.Errorf("create Kyutai cache dir: %w", err)
	}
	uvExe, err := uv.Install(ctx)
	if err != nil {
		return nil, err
	}
	return start(ctx, cfg, func(ctx context.Context) *exec.Cmd { return command(ctx, uvExe, cfg) })
}

// start waits for the worker to load the model.
func start(ctx context.Context, cfg Config, newCmd func(context.Context) *exec.Cmd) (*Server, error) {
	p, u, err := worker.Start(ctx, "kyutai", newCmd, func(line []byte) error {
		var r readyLine
		if err := json.Unmarshal(line, &r); err != nil {
			return err
		}
		if r.Kind != "ready" {
			return fmt.Errorf("kind %q, want ready", r.Kind)
		}
		if r.SampleRate != kyutai.SampleRate {
			return fmt.Errorf("sample rate %d, want %d", r.SampleRate, kyutai.SampleRate)
		}
		slog.InfoContext(ctx, "kyutai: ready", "model", cfg.Model)
		return nil
	})
	if err != nil {
		return nil, err
	}
	return &Server{p: p, url: u}, nil
}

// URL is the worker's base URL for kyutai.New.
func (s *Server) URL() string { return s.url }

// Close stops the worker and reports an unexpected exit. It is idempotent.
func (s *Server) Close() error { return s.p.Close() }

// Done implements genai.Server.
func (s *Server) Done() <-chan struct{} { return s.p.Done() }

// Err implements genai.Server.
func (s *Server) Err() error { return s.p.Err() }

// readyLine is the worker's readiness line; worker.Start reads its "url".
type readyLine struct {
	Kind       string `json:"kind"`
	SampleRate int    `json:"sample_rate"`
}

// command runs the worker with moshi-mlx on Apple silicon and moshi elsewhere.
func command(ctx context.Context, uvExe string, cfg Config) *exec.Cmd {
	backend, pkg, c := "torch", moshiPackage, checkpoints[cfg.Model].torch
	if runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" {
		backend, pkg, c = "mlx", moshiMLXPackage, checkpoints[cfg.Model].mlx
	}
	cmd := exec.CommandContext(ctx, uvExe, "run", "--isolated", "--python", pythonVersion, "--with", pkg, "--with", msgpackPackage, "python", "-u", "-c", script, "--cache-dir", cfg.CacheDir, "--model", c.repo, "--revision", c.revision, "--backend", backend)
	cmd.Dir = cfg.CacheDir
	// Keep model download progress out of logs without hiding warnings or
	// errors. UTF-8 mode makes open() and stdio default to UTF-8 instead of
	// the Windows ANSI code page.
	cmd.Env = append(cmd.Environ(), "HF_HUB_DISABLE_PROGRESS_BARS=1", "PYTHONUTF8=1")
	return cmd
}

func normalizeConfig(cfg Config) (Config, error) {
	if cfg.Model == "" {
		cfg.Model = DefaultModel
	}
	if _, ok := checkpoints[cfg.Model]; !ok {
		return Config{}, fmt.Errorf("kyutai STT model %q is not supported; use one of %q", cfg.Model, kyutai.Models)
	}
	if cfg.CacheDir == "" {
		base, err := os.UserCacheDir()
		if err != nil {
			return Config{}, fmt.Errorf("get user cache dir: %w", err)
		}
		cfg.CacheDir = filepath.Join(base, "genaipy", "kyutai")
	}
	if !filepath.IsAbs(cfg.CacheDir) {
		return Config{}, errors.New("kyutai cache directory must be absolute")
	}
	return cfg, nil
}

// checkpoint is a Hugging Face repository at a commit.
type checkpoint struct {
	repo, revision string
}

// modelCheckpoints are a model's PyTorch and MLX repositories.
type modelCheckpoints struct {
	torch, mlx checkpoint
}

// checkpoints are each model's repositories. The revisions pin the stt_config
// that kyutai declares.
var checkpoints = map[kyutai.Model]modelCheckpoints{
	kyutai.STT1B: {
		torch: checkpoint{"kyutai/stt-1b-en_fr", "1c34c6b4f7e9299bb61985f145052ff131005dde"},
		mlx:   checkpoint{"kyutai/stt-1b-en_fr-mlx", "2b995724eef1e964b7ccb6a762b35a665c4abe0d"},
	},
	kyutai.STT2B6: {
		torch: checkpoint{"kyutai/stt-2.6b-en", "a07aec56d22be5589cd0bc8709c75b6cf3e3039d"},
		mlx:   checkpoint{"kyutai/stt-2.6b-en-mlx", "5ccfc8f844ab8abaf591c7954f70c07e72325554"},
	},
}
