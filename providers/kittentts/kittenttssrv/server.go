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
	"slices"

	"github.com/maruel/genai"
	"github.com/maruel/genaipy/internal/uv"
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
// genaipy/kittentts. Model must be one kittentts.Voices supports. CacheDir
// must be absolute when set.
type Config struct {
	Model    string
	CacheDir string
}

var _ genai.Server = (*Server)(nil)

// Server owns a KittenTTS worker process producing kittentts.SampleRate PCM.
type Server struct {
	p   *worker.Process
	url string
}

// New starts the worker, installing the pinned uv and its pinned Python
// package if needed. Startup downloads the model at kittentts.Revision when it
// is absent from the cache. ctx owns the worker lifetime; call Close when
// finished.
func New(ctx context.Context, cfg Config) (*Server, error) {
	cfg, err := normalizeConfig(cfg)
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(cfg.CacheDir, 0o750); err != nil {
		return nil, fmt.Errorf("create KittenTTS cache dir: %w", err)
	}
	uvExe, err := uv.Install(ctx)
	if err != nil {
		return nil, err
	}
	return start(ctx, cfg, func(ctx context.Context) *exec.Cmd { return command(ctx, uvExe, cfg) })
}

// start waits for the worker to load every voice kittentts.Voices lists.
func start(ctx context.Context, cfg Config, newCmd func(context.Context) *exec.Cmd) (*Server, error) {
	want, err := kittentts.Voices(cfg.Model)
	if err != nil {
		return nil, err
	}
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
		for _, v := range want {
			if !slices.Contains(r.Voices, v.Name) {
				return fmt.Errorf("%s has no voice %s", cfg.Model, v.Name)
			}
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

// Done implements genai.Server.
func (s *Server) Done() <-chan struct{} { return s.p.Done() }

// Err implements genai.Server.
func (s *Server) Err() error { return s.p.Err() }

func command(ctx context.Context, uvExe string, cfg Config) *exec.Cmd {
	cmd := exec.CommandContext(ctx, uvExe, "run", "--isolated", "--python", pythonVersion, "--with", kittenTTSPackage, "python", "-u", "-c", script, "--cache-dir", cfg.CacheDir, "--model", cfg.Model, "--revision", kittentts.Revision)
	cmd.Dir = cfg.CacheDir
	// Keep model download progress out of logs without hiding warnings or
	// errors. UTF-8 mode makes open() default to UTF-8 instead of the Windows
	// ANSI code page: kittenml reads its voices JSON without an encoding.
	cmd.Env = append(cmd.Environ(), "HF_HUB_DISABLE_PROGRESS_BARS=1", "PYTHONUTF8=1")
	return cmd
}

func normalizeConfig(cfg Config) (Config, error) {
	if cfg.Model == "" {
		cfg.Model = DefaultModel
	}
	if _, err := kittentts.Voices(cfg.Model); err != nil {
		return Config{}, err
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
