// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package nemospeechsrv runs NVIDIA NeMo-Speech.cpp as a managed local ASR
// server for the nemospeech client.
//
// NeMo-Speech.cpp is NVIDIA's native runtime for NeMo speech models, including
// Parakeet and Nemotron ASR. llama.cpp cannot serve these models: its Parakeet
// support is only the audio encoder of Nemotron Omni, without the TDT or CTC
// decoder. Download fetches a pinned release build, verified against the
// SHA-256 published beside it, and New starts "nemo-speech serve", which
// downloads the
// model into NeMo-Speech.cpp's own model cache ($NEMO_SPEECH_MODEL_DIR, else
// $XDG_CACHE_HOME/nemo-speech/models, else ~/.cache/nemo-speech/models) on
// first use. No Python is involved.
//
// See https://github.com/NVIDIA/NeMo-Speech.cpp/blob/main/docs/server.md.
package nemospeechsrv

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genaipy/internal/release"
	"github.com/maruel/genaipy/providers/nemospeech"
)

// Version is the pinned NVIDIA/NeMo-Speech.cpp release.
const Version = "0.3.0"

// Backend selects a NeMo-Speech.cpp release build.
//
// Each platform publishes a subset: Linux x86-64 and Windows x86-64 have CPU,
// CUDA and Vulkan; Linux arm64 has CPU, CUDA12, CUDA13 and Vulkan; macOS arm64
// has CPU and Metal; macOS x86-64 has CPU; Windows arm64 has CUDA. New fails
// for a build the release does not publish.
type Backend string

// Release builds.
const (
	CPU    Backend = "cpu"
	CUDA   Backend = "cuda"
	CUDA12 Backend = "cuda12"
	CUDA13 Backend = "cuda13"
	Metal  Backend = "metal"
	Vulkan Backend = "vulkan"
)

var _ genai.Server = (*Server)(nil)

// Server owns a NeMo-Speech.cpp server process.
type Server struct {
	url    string
	cancel context.CancelFunc
	done   chan struct{}
	err    error // Written before done closes.
}

// New starts exe, from Download, serving model on a free loopback port, and
// waits until the model is loaded.
//
// Startup may download the model; ctx bounds it and owns the process
// lifetime. The caller must call Close on success.
func New(ctx context.Context, exe string, model nemospeech.Model) (*Server, error) {
	l, err := (&net.ListenConfig{}).Listen(ctx, "tcp", "127.0.0.1:0")
	if err != nil {
		return nil, err
	}
	port := strconv.Itoa(l.Addr().(*net.TCPAddr).Port) //nolint:forcetypeassert // TCP listeners use TCP addresses.
	if err := l.Close(); err != nil {
		return nil, err
	}
	// The process binds the port after loading the model; another process can
	// take it meanwhile, which fails startup with a bind error.
	u := "http://127.0.0.1:" + port
	client := &http.Client{Transport: &http.Transport{Proxy: nil}}
	procCtx, cancel := context.WithCancel(ctx)
	cmd := exec.CommandContext(procCtx, exe, "serve", "--asr-model", string(model), "--host", "127.0.0.1", "--port", port, "--no-ui") // #nosec G204 -- The executable comes from the verified release; arguments use no shell.
	hideConsole(cmd)
	cmd.Env = serverEnv()
	logs := slog.NewLogLogger(slog.Default().Handler(), slog.LevelInfo).Writer()
	cmd.Stdout, cmd.Stderr = logs, logs
	// SIGINT stops HTTP admission and releases the models. Windows has no
	// console signal for a child without a console.
	cmd.Cancel = func() error {
		if runtime.GOOS == "windows" {
			return cmd.Process.Kill()
		}
		return cmd.Process.Signal(os.Interrupt)
	}
	cmd.WaitDelay = 10 * time.Second
	if err := cmd.Start(); err != nil {
		cancel()
		return nil, fmt.Errorf("nemospeechsrv: start: %w", err)
	}
	s := &Server{url: u, cancel: cancel, done: make(chan struct{})}
	go func() {
		// "nemo-speech serve" stops on SIGINT or SIGTERM and returns 0; one that
		// outlives WaitDelay is killed, which is reported.
		s.err = base.ServerExit(cmd.Wait(), cmd.ProcessState, procCtx.Err() != nil, os.Interrupt)
		client.CloseIdleConnections()
		close(s.done)
	}()
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-s.done:
			cancel()
			return nil, errors.Join(errors.New("nemospeechsrv: server exited during startup"), s.err)
		case <-ctx.Done():
			return nil, errors.Join(ctx.Err(), s.Close())
		case <-ticker.C:
			if ready(procCtx, client, u) {
				return s, nil
			}
		}
	}
}

// URL is the server's base URL for nemospeech.New.
func (s *Server) URL() string {
	return s.url
}

// Close stops the server and reports an unexpected exit.
func (s *Server) Close() error {
	s.cancel()
	<-s.done
	return s.err
}

// Done implements genai.Server.
func (s *Server) Done() <-chan struct{} { return s.done }

// Err implements genai.Server.
func (s *Server) Err() error {
	select {
	case <-s.done:
		return s.err
	default:
		return nil
	}
}

// Download installs the release build for backend into cacheDir unless it is
// already there, and returns its executable. Concurrent callers for one
// cacheDir, in this process or others, wait for a single install. Empty backend
// selects Metal on macOS arm64, CUDA on Windows arm64, and CPU elsewhere.
func Download(ctx context.Context, cacheDir string, backend Backend) (string, error) {
	if !filepath.IsAbs(cacheDir) {
		return "", errors.New("nemospeechsrv: cache directory must be an absolute path")
	}
	u, err := releaseURL(backend)
	if err != nil {
		return "", err
	}
	stem := strings.TrimSuffix(strings.TrimSuffix(path.Base(u), ".tar.gz"), ".zip")
	exe := "bin/nemo-speech"
	if runtime.GOOS == "windows" {
		exe += ".exe"
	}
	exe, err = release.Install(ctx, http.DefaultClient, u, stem, filepath.Join(cacheDir, stem), exe)
	if err != nil {
		return "", fmt.Errorf("NeMo-Speech.cpp %s: %w", Version, err)
	}
	return exe, nil
}

// releaseURL returns the archive URL of backend for this platform. An empty
// backend selects Metal on macOS arm64, CUDA on Windows arm64 (its only
// build), and CPU elsewhere.
func releaseURL(backend Backend) (string, error) {
	osName := map[string]string{"darwin": "macos", "linux": "linux", "windows": "windows"}[runtime.GOOS]
	arch := map[string]string{"amd64": "x86_64", "arm64": "aarch64"}[runtime.GOARCH]
	if osName == "" || arch == "" {
		return "", fmt.Errorf("NeMo-Speech.cpp has no build for %s/%s", runtime.GOOS, runtime.GOARCH)
	}
	switch backend {
	case CPU, CUDA, CUDA12, CUDA13, Metal, Vulkan:
	case "":
		switch {
		case runtime.GOOS == "darwin" && runtime.GOARCH == "arm64":
			backend = Metal
		case runtime.GOOS == "windows" && runtime.GOARCH == "arm64":
			backend = CUDA
		default:
			backend = CPU
		}
	default:
		return "", fmt.Errorf("unknown NeMo-Speech.cpp backend %q", backend)
	}
	ext := ".tar.gz"
	if runtime.GOOS == "windows" {
		ext = ".zip"
	}
	return "https://github.com/NVIDIA/NeMo-Speech.cpp/releases/download/v" + Version + "/nemo-speech-" + Version + "-" + osName + "-" + arch + "-" + string(backend) + ext, nil
}

// ready reports whether GET /ready succeeds. The server answers 503 until an
// engine is ready and refuses connections until it binds.
func ready(ctx context.Context, client *http.Client, u string) bool {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u+"/ready", http.NoBody)
	if err != nil {
		return false
	}
	resp, err := client.Do(req)
	if err != nil {
		return false
	}
	defer func() { _ = resp.Body.Close() }()
	var r struct {
		Ready bool `json:"ready"`
	}
	return resp.StatusCode == http.StatusOK && json.NewDecoder(resp.Body).Decode(&r) == nil && r.Ready
}

// serverEnv drops NEMO_SPEECH_HTTP_* settings, which would override the
// listener this package owns; an inherited API key would reject its requests.
func serverEnv() []string {
	var env []string
	for _, kv := range os.Environ() {
		if !strings.HasPrefix(kv, "NEMO_SPEECH_HTTP_") {
			env = append(env, kv)
		}
	}
	return env
}
