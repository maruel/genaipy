// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package whistlesrv runs the Cactus Whistle ASR worker for the whistle
// client.
//
// Whistle requires a prebuilt Needle engine, fetched by the pinned
// cactus-needle Python package that the pinned uv installs. No llama.cpp is
// involved.
package whistlesrv

import (
	"context"
	_ "embed"
	"os"
	"os/exec"

	"github.com/maruel/genai"
	"github.com/maruel/genaipy/internal/uv"
	"github.com/maruel/genaipy/internal/worker"
)

//go:embed worker.py
var script string

var _ genai.Server = (*Server)(nil)

// Server owns a Whistle worker process.
type Server struct {
	p   *worker.Process
	url string
}

// New starts a worker, installing the pinned uv and cactus-needle 3.1.0 and
// downloading weights when needed. model optionally selects a local .cact weights file;
// otherwise the worker downloads whistle.DefaultModel. ctx owns the worker
// lifetime; call Close when finished.
func New(ctx context.Context, model string) (*Server, error) {
	uvExe, err := uv.Install(ctx)
	if err != nil {
		return nil, err
	}
	return start(ctx, func(ctx context.Context) *exec.Cmd {
		cmd := exec.CommandContext(ctx, uvExe, "run", "--isolated", "--no-project", "--python", "3.12", "--with", "cactus-needle==3.1.0", "python", "-u", "-c", script, "--model", model)
		// UTF-8 mode makes open() and stdio default to UTF-8 instead of the
		// Windows ANSI code page.
		cmd.Env = append(os.Environ(), "NEEDLE_TELEMETRY=0", "DO_NOT_TRACK=1", "PYTHONUTF8=1")
		return cmd
	})
}

func start(ctx context.Context, newCmd func(context.Context) *exec.Cmd) (*Server, error) {
	p, u, err := worker.Start(ctx, "whistle", newCmd, func([]byte) error { return nil })
	if err != nil {
		return nil, err
	}
	return &Server{p: p, url: u}, nil
}

// URL is the worker's base URL for whistle.New.
func (s *Server) URL() string { return s.url }

// Close stops the worker and reports an unexpected exit. It is idempotent.
func (s *Server) Close() error { return s.p.Close() }

// Done implements genai.Server.
func (s *Server) Done() <-chan struct{} { return s.p.Done() }

// Err implements genai.Server.
func (s *Server) Err() error { return s.p.Err() }
