// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package genaipy provides Python backends for genai.
package genaipy

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
)

var _ genai.Server = (*Server)(nil)

// Server owns a Python backend process. Its lifetime is bounded by the context
// passed to NewServer or by Close.
type Server struct {
	url    string
	cancel context.CancelFunc
	done   chan struct{}
	err    error // Written before done closes.
}

// NewServer prepares a Python virtualenv and starts llm.py or image_gen.py.
//
// It may install dependencies and download models. Readiness uses the model's
// health endpoint and process identity, without generating a response. Startup
// is bounded by ctx and a ten minute timeout. The caller owns Close on success.
// extraArgs cannot override the launcher's address or server mode.
func NewServer(ctx context.Context, script, cacheDir, logName string, extraArgs []string) (*Server, error) {
	if script != "llm.py" && script != "image_gen.py" {
		return nil, errors.New("script must be llm.py or image_gen.py")
	}
	if !filepath.IsAbs(cacheDir) {
		return nil, errors.New("cacheDir must be an absolute path")
	}
	if !filepath.IsAbs(logName) {
		return nil, errors.New("logName must be an absolute path")
	}
	for _, arg := range extraArgs {
		if arg == "--host" || strings.HasPrefix(arg, "--host=") || arg == "--port" || strings.HasPrefix(arg, "--port=") || arg == "--prompt" || strings.HasPrefix(arg, "--prompt=") {
			return nil, fmt.Errorf("launcher owns server address and mode: %s", arg)
		}
	}
	readyCtx, readyCancel := context.WithTimeout(ctx, 10*time.Minute)
	defer readyCancel()
	if needRecreate(cacheDir) {
		if err := recreate(readyCtx, cacheDir); err != nil {
			return nil, err
		}
	}
	l, err := (&net.ListenConfig{}).Listen(readyCtx, "tcp", "127.0.0.1:0")
	if err != nil {
		return nil, err
	}
	port := strconv.Itoa(l.Addr().(*net.TCPAddr).Port) //nolint:forcetypeassert // TCP listeners use TCP addresses.
	if err := l.Close(); err != nil {
		return nil, err
	}
	bin, exe := "bin", "python3"
	if runtime.GOOS == "windows" {
		bin, exe = "Scripts", "python.exe"
	}
	args := append([]string{script, "--host", "127.0.0.1", "--port", port}, extraArgs...)
	log, err := os.OpenFile(logName, os.O_WRONLY|os.O_APPEND|os.O_CREATE, 0o644)
	if err != nil {
		return nil, fmt.Errorf("failed to create log file: %w", err)
	}
	procCtx, cancel := context.WithCancel(ctx)
	cmd := exec.CommandContext(procCtx, filepath.Join(cacheDir, "venv", bin, exe), args...)
	cmd.Dir = cacheDir
	cmd.Stdout, cmd.Stderr = log, log
	if err = cmd.Start(); err != nil {
		cancel()
		return nil, errors.Join(fmt.Errorf("failed to start Python: %w", err), log.Close())
	}
	s := &Server{url: "http://127.0.0.1:" + port, cancel: cancel, done: make(chan struct{})}
	go func() {
		// CommandContext kills the child on cancellation.
		err := base.ServerExit(cmd.Wait(), cmd.ProcessState, procCtx.Err() != nil, os.Kill)
		s.err = errors.Join(err, log.Close())
		close(s.done)
	}()
	hc := &http.Client{Timeout: time.Second, Transport: &http.Transport{Proxy: nil}}
	defer hc.CloseIdleConnections()
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-s.done:
			err = s.err
			cancel()
			if err == nil {
				err = errors.New("process exited early; look at logs to diagnose")
			}
			return nil, err
		case <-readyCtx.Done():
			return nil, errors.Join(readyCtx.Err(), s.Close())
		case <-ticker.C:
			req, err := http.NewRequestWithContext(readyCtx, http.MethodGet, s.url+"/health", http.NoBody)
			if err != nil {
				return nil, errors.Join(err, s.Close())
			}
			resp, err := hc.Do(req)
			if err != nil {
				continue
			}
			var health struct {
				Status string `json:"status"`
				PID    int    `json:"pid"`
			}
			err = json.NewDecoder(resp.Body).Decode(&health)
			err = errors.Join(err, resp.Body.Close())
			if err == nil && resp.StatusCode == http.StatusOK && health.Status == "ok" && health.PID == cmd.Process.Pid {
				return s, nil
			}
		}
	}
}

// Close implements genai.Server.
func (s *Server) Close() error {
	s.cancel()
	<-s.done
	return s.err
}

// URL implements genai.Server.
func (s *Server) URL() string { return s.url }

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

var (
	//go:embed image_gen.py
	imageGenPy []byte
	//go:embed llm.py
	llmPy []byte
	//go:embed requirements.txt
	requirementsTxt []byte
	//go:embed setup.bat
	setupBat []byte
	//go:embed setup.sh
	setupSh []byte
)

func needRecreate(cache string) bool {
	// pyvenv.cfg can exist even when dependency installation failed.
	if _, err := os.Stat(filepath.Join(cache, ".setup-complete")); err != nil {
		return true
	}
	if _, err := os.Stat(filepath.Join(cache, "venv", "pyvenv.cfg")); err != nil {
		return true
	}
	if b, err := os.ReadFile(filepath.Join(cache, "image_gen.py")); err != nil || !bytes.Equal(b, imageGenPy) {
		return true
	}
	if b, err := os.ReadFile(filepath.Join(cache, "llm.py")); err != nil || !bytes.Equal(b, llmPy) {
		return true
	}
	if b, err := os.ReadFile(filepath.Join(cache, "requirements.txt")); err != nil || !bytes.Equal(b, requirementsTxt) {
		return true
	}
	name := "setup.sh"
	content := setupSh
	if runtime.GOOS == "windows" {
		name = "setup.bat"
		content = setupBat
	}
	if b, err := os.ReadFile(filepath.Join(cache, name)); err != nil || !bytes.Equal(b, content) {
		return true
	}
	return false
}

func recreate(ctx context.Context, cache string) error {
	if err := os.MkdirAll(cache, 0o755); err != nil {
		return err
	}
	if err := os.Remove(filepath.Join(cache, ".setup-complete")); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	if err := os.WriteFile(filepath.Join(cache, "image_gen.py"), imageGenPy, 0o755); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(cache, "llm.py"), llmPy, 0o755); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(cache, "requirements.txt"), requirementsTxt, 0o644); err != nil {
		return err
	}
	name := "setup.sh"
	content := setupSh
	if runtime.GOOS == "windows" {
		name = "setup.bat"
		content = setupBat
	}
	if err := os.WriteFile(filepath.Join(cache, name), content, 0o755); err != nil {
		return err
	}
	c := exec.CommandContext(ctx, filepath.Join(cache, name))
	if runtime.GOOS == "windows" {
		c = exec.CommandContext(ctx, "cmd.exe", "/c", filepath.Join(cache, name))
	}
	c.Dir = cache
	c.Stdout = os.Stdout
	c.Stderr = os.Stderr
	if err := c.Run(); err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(cache, ".setup-complete"), nil, 0o644)
}
