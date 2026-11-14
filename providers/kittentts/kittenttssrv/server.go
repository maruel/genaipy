// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package kittenttssrv runs the Python KittenTTS worker for the kittentts
// client.
package kittenttssrv

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

	"github.com/maruel/genaipy/internal/worker"
	"github.com/maruel/genaipy/providers/kittentts"
)

//go:embed worker.py
var script string

const (
	// DefaultModel is the model New loads when Config.Model is empty.
	DefaultModel = "KittenML/kitten-tts-2"
	// Upstream has no 0.9.3 tag; 0.8.1 predates KittenTTS 2. Pin the tested
	// source revision. Its API is defined in kittenml/get_model.py:
	// https://github.com/KittenML/KittenTTS/blob/ab5592c28b6f376ea4c3963e84ce6d6688241eca/kittenml/get_model.py
	kittenTTSPackage = "kittenml @ git+https://github.com/KittenML/KittenTTS@ab5592c28b6f376ea4c3963e84ce6d6688241eca"
	pythonVersion    = "3.12"
)

// Config selects the worker's model and model cache.
//
// Zero values use DefaultModel and the user's cache directory under
// genaipy/kittentts. CacheDir must be absolute when set.
type Config struct {
	Model    string
	CacheDir string
}

// Server owns a KittenTTS worker process producing kittentts.SampleRate PCM.
type Server struct {
	p   *worker.Process
	url string
}

// New starts the worker, installing its pinned Python package with uv if
// needed. Startup downloads the model when it is absent from the cache. ctx
// owns the worker lifetime; call Close when finished.
func New(ctx context.Context, cfg Config) (*Server, error) {
	cfg, err := normalizeConfig(cfg)
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(cfg.CacheDir, 0o750); err != nil {
		return nil, fmt.Errorf("create KittenTTS cache dir: %w", err)
	}
	return start(ctx, cfg, func(ctx context.Context) *exec.Cmd { return command(ctx, cfg) })
}

func start(ctx context.Context, cfg Config, newCmd func(context.Context) *exec.Cmd) (*Server, error) {
	p, u, err := worker.Start(ctx, "kittentts", newCmd, func(line []byte) error {
		var r struct {
			Kind       string   `json:"kind"`
			Error      string   `json:"error"`
			SampleRate int      `json:"sample_rate"`
			Voices     []string `json:"voices"`
		}
		if err := json.Unmarshal(line, &r); err != nil {
			return err
		}
		if r.Error != "" {
			return errors.New(r.Error)
		}
		if r.Kind != "ready" {
			return fmt.Errorf("kind %q, want ready", r.Kind)
		}
		if r.SampleRate != kittentts.SampleRate {
			return fmt.Errorf("sample rate %d, want %d", r.SampleRate, kittentts.SampleRate)
		}
		slog.InfoContext(ctx, "kittentts: ready", "model", cfg.Model, "voices", r.Voices)
		return nil
	})
	if err != nil {
		return nil, err
	}
	return &Server{p: p, url: u}, nil
}

// URL is the worker's base URL for kittentts.New.
func (s *Server) URL() string { return s.url }

// Close stops the worker and reports an unexpected exit. It is idempotent.
func (s *Server) Close() error { return s.p.Close() }

func command(ctx context.Context, cfg Config) *exec.Cmd {
	cmd := exec.CommandContext(ctx, "uv", "run", "--isolated", "--python", pythonVersion, "--with", kittenTTSPackage, "python", "-u", "-c", script, "--cache-dir", cfg.CacheDir, "--model", cfg.Model)
	cmd.Dir = cfg.CacheDir
	// Keep model download progress out of logs without hiding warnings or errors.
	cmd.Env = append(cmd.Environ(), "HF_HUB_DISABLE_PROGRESS_BARS=1")
	return cmd
}

func normalizeConfig(cfg Config) (Config, error) {
	if cfg.Model == "" {
		cfg.Model = DefaultModel
	}
	if cfg.CacheDir == "" {
		base, err := os.UserCacheDir()
		if err != nil {
			return Config{}, fmt.Errorf("get user cache dir: %w", err)
		}
		cfg.CacheDir = filepath.Join(base, "genaipy", "kittentts")
	}
	if !filepath.IsAbs(cfg.CacheDir) {
		return Config{}, errors.New("KittenTTS cache directory must be absolute")
	}
	return cfg, nil
}
