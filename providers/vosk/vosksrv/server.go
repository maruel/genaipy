// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package vosksrv runs the Vosk ASR worker for the vosk client.
//
// The pinned uv installs the pinned vosk Python package, whose wheels carry
// the native recognizer for Linux, macOS and Windows. New downloads the model
// archive from alphacephei.com, verified against a pinned SHA-256, into the
// user cache directory.
package vosksrv

import (
	"context"
	_ "embed"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/maruel/genai"
	"github.com/maruel/genaipy/internal/release"
	"github.com/maruel/genaipy/internal/uv"
	"github.com/maruel/genaipy/internal/worker"
	"github.com/maruel/genaipy/providers/vosk"
)

// Version is the pinned vosk Python package. 0.3.45 publishes no macOS wheel.
const Version = "0.3.44"

//go:embed worker.py
var script string

// digests are the SHA-256 of each vosk.Models archive.
var digests = map[vosk.Model]string{
	vosk.English: "47f9a81ebb039dbb0bd319175c36ac393c0893b796c2b6303e64cf58c27b69f6",
}

var _ genai.Server = (*Server)(nil)

// Server owns a Vosk worker process.
type Server struct {
	p   *worker.Process
	url string
}

// New starts a worker serving model, installing the pinned uv and vosk and
// downloading the model when needed. ctx owns the worker lifetime; call Close
// when finished.
func New(ctx context.Context, model vosk.Model) (*Server, error) {
	digest, ok := digests[model]
	if !ok {
		return nil, fmt.Errorf("unsupported Vosk model %q", model)
	}
	cache, err := os.UserCacheDir()
	if err != nil {
		return nil, fmt.Errorf("get user cache dir: %w", err)
	}
	name := string(model)
	conf, err := release.Install(ctx, http.DefaultClient, "https://alphacephei.com/vosk/models/"+name+".zip", digest, name, filepath.Join(cache, "genaipy", "vosk", name), "conf/model.conf")
	if err != nil {
		return nil, fmt.Errorf("vosk model %s: %w", name, err)
	}
	dir := filepath.Dir(filepath.Dir(conf))
	uvExe, err := uv.Install(ctx)
	if err != nil {
		return nil, err
	}
	return start(ctx, func(ctx context.Context) *exec.Cmd {
		cmd := exec.CommandContext(ctx, uvExe, "run", "--isolated", "--no-project", "--python", "3.12", "--with", "vosk=="+Version, "python", "-u", "-c", script, "--model", dir)
		// UTF-8 mode makes open() and stdio default to UTF-8 instead of the
		// Windows ANSI code page.
		cmd.Env = append(os.Environ(), "PYTHONUTF8=1")
		return cmd
	})
}

func start(ctx context.Context, newCmd func(context.Context) *exec.Cmd) (*Server, error) {
	p, u, err := worker.Start(ctx, "vosk", newCmd, func([]byte) error { return nil })
	if err != nil {
		return nil, err
	}
	return &Server{p: p, url: u}, nil
}

// URL is the worker's base URL for vosk.New.
func (s *Server) URL() string { return s.url }

// Close stops the worker and reports an unexpected exit. It is idempotent.
func (s *Server) Close() error { return s.p.Close() }

// Done implements genai.Server.
func (s *Server) Done() <-chan struct{} { return s.p.Done() }

// Err implements genai.Server.
func (s *Server) Err() error { return s.p.Err() }
