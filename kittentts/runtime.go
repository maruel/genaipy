// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package kittentts runs a managed Python KittenTTS speech worker.
package kittentts

import (
	"bufio"
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"log/slog"
	"mime"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"

	"github.com/maruel/genaipy/speech"
)

//go:embed worker.py
var kittenTTSWorkerScript string

const (
	kittenTTSModel = "KittenML/kitten-tts-2"
	// Upstream has no 0.9.3 tag; 0.8.1 predates KittenTTS 2. Pin the tested
	// source revision. Its API is defined in kittenml/get_model.py:
	// https://github.com/KittenML/KittenTTS/blob/ab5592c28b6f376ea4c3963e84ce6d6688241eca/kittenml/get_model.py
	kittenTTSPackage    = "kittenml @ git+https://github.com/KittenML/KittenTTS@ab5592c28b6f376ea4c3963e84ce6d6688241eca"
	kittenTTSPython     = "3.12"
	kittenTTSVoice      = "Jasper"
	kittenTTSReadyKind  = "ready"
	kittenTTSStdoutName = "stdout"
	kittenTTSSynthesize = "/synthesize"
)

// Config selects the worker's model, voice and model cache.
//
// Zero values use KittenTTS 2, Jasper, and the user's
// cache directory under genaipy/kittentts. CacheDir must be absolute when set.
type Config struct {
	Model    string
	Voice    string
	CacheDir string
}

// Runtime owns a long-lived KittenTTS worker producing 24 kHz mono S16LE PCM.
type Runtime struct {
	config Config
	ctx    context.Context

	mu     sync.Mutex
	closed bool

	start         kittenTTSCommandFactory
	client        *http.Client
	baseURL       string
	processCtx    context.Context
	processCancel context.CancelFunc
	cmd           *exec.Cmd
	// stdin is the worker's lifeline. The worker exits when it reads EOF,
	// which happens when the runtime closes it or the owning process dies.
	// Killing the uv launcher alone would orphan its Python child.
	stdin  io.Closer
	stdout *bufio.Reader
	wait   chan error
}

// New starts the worker, installing its pinned Python package with uv if needed.
//
// ctx owns the worker lifetime; request cancellation only affects that request.
// Startup downloads a model when it is absent from the cache. Close must be
// called when the runtime is no longer needed. Output is 24 kHz mono S16LE PCM.
func New(ctx context.Context, cfg Config) (*Runtime, error) {
	cfg, err := normalizeConfig(cfg)
	if err != nil {
		return nil, err
	}
	return newWithCommand(ctx, cfg, func(ctx context.Context) (*exec.Cmd, error) { return kittenTTSCommand(ctx, cfg) })
}

func newWithCommand(ctx context.Context, cfg Config, start kittenTTSCommandFactory) (*Runtime, error) {
	a := &Runtime{config: cfg, ctx: ctx, start: start, client: http.DefaultClient}
	if err := a.ensureStartedLocked(ctx); err != nil {
		return nil, err
	}
	return a, nil
}

// Close stops the worker and releases its resources.
//
// A closed runtime cannot be restarted or used to synthesize speech.
func (a *Runtime) Close() error {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.closed = true
	return a.stopLocked()
}

// Synthesize streams aligned PCM chunks with independent owned storage.
//
// Stopping iteration closes the response. Empty text yields no audio;
// cancellation terminates only the request, leaving the worker reusable.
func (a *Runtime) Synthesize(ctx context.Context, text string) iter.Seq2[[]byte, error] {
	return func(yield func([]byte, error) bool) {
		a.mu.Lock()
		if a.closed {
			a.mu.Unlock()
			yield(nil, errors.New("KittenTTS runtime is closed"))
			return
		}
		if text == "" {
			a.mu.Unlock()
			return
		}
		if err := a.ensureStartedLocked(ctx); err != nil {
			a.mu.Unlock()
			yield(nil, err)
			return
		}
		endpoint := a.baseURL + kittenTTSSynthesize
		client := a.client
		a.mu.Unlock()

		req := kittenTTSRequest{Text: text, Voice: a.config.Voice}
		data, err := json.Marshal(req)
		if err != nil {
			yield(nil, fmt.Errorf("encode KittenTTS request: %w", err))
			return
		}
		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(data))
		if err != nil {
			yield(nil, fmt.Errorf("create KittenTTS request: %w", err))
			return
		}
		httpReq.Header.Set("Content-Type", "application/json")
		httpResp, err := client.Do(httpReq)
		if err != nil {
			if ctx.Err() != nil {
				yield(nil, ctx.Err())
				return
			}
			yield(nil, errors.Join(fmt.Errorf("post KittenTTS request: %w", err), a.stop()))
			return
		}
		defer func() {
			if err := httpResp.Body.Close(); err != nil {
				slog.WarnContext(ctx, "kittentts: close KittenTTS response body", "err", err)
			}
		}()
		if httpResp.StatusCode != http.StatusOK {
			body, err := io.ReadAll(io.LimitReader(httpResp.Body, 4096))
			yield(nil, errors.Join(fmt.Errorf("KittenTTS HTTP status %s: %s", httpResp.Status, strings.TrimSpace(string(body))), err))
			return
		}
		contentType, _, err := mime.ParseMediaType(httpResp.Header.Get("Content-Type"))
		if err != nil || contentType != "audio/pcm" && contentType != "application/octet-stream" {
			yield(nil, fmt.Errorf("KittenTTS returned %q instead of PCM", httpResp.Header.Get("Content-Type")))
			return
		}
		for pcm, err := range speech.PCMChunks(httpResp.Body) {
			if !yield(pcm, err) {
				return
			}
		}
	}
}

func (a *Runtime) ensureStartedLocked(ctx context.Context) error {
	if err := a.ctx.Err(); err != nil {
		return err
	}
	if a.cmd != nil {
		return nil
	}
	processCtx, cancel := context.WithCancel(a.ctx)
	cmd, err := a.start(processCtx)
	if err != nil {
		cancel()
		return err
	}
	stdin, err := cmd.StdinPipe()
	if err != nil {
		cancel()
		return fmt.Errorf("open KittenTTS stdin: %w", err)
	}
	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		cancel()
		return fmt.Errorf("open KittenTTS stdout: %w", err)
	}
	stderrPipe, err := cmd.StderrPipe()
	if err != nil {
		cancel()
		return fmt.Errorf("open KittenTTS stderr: %w", err)
	}
	if err := cmd.Start(); err != nil {
		cancel()
		return fmt.Errorf("start KittenTTS worker: %w", err)
	}
	a.processCtx = processCtx
	a.processCancel = cancel
	a.cmd = cmd
	a.stdin = stdin
	a.stdout = bufio.NewReader(stdoutPipe)
	go logKittenTTSOutput(ctx, "stderr", stderrPipe)

	line, err := a.readLineLocked(ctx, kittenTTSStdoutName)
	if err != nil {
		return errors.Join(err, a.stopLocked())
	}
	var ready kittenTTSReady
	if err := json.Unmarshal(line, &ready); err != nil {
		return errors.Join(fmt.Errorf("decode KittenTTS ready message: %w", err), a.stopLocked())
	}
	if ready.Error != "" {
		return errors.Join(fmt.Errorf("KittenTTS startup: %s", ready.Error), a.stopLocked())
	}
	if ready.Kind != kittenTTSReadyKind {
		return errors.Join(fmt.Errorf("KittenTTS startup returned kind %q, want %q", ready.Kind, kittenTTSReadyKind), a.stopLocked())
	}
	if err := validateKittenTTSURL(ready.URL); err != nil {
		return errors.Join(err, a.stopLocked())
	}
	if ready.SampleRate != speech.SynthesisSampleRate {
		return errors.Join(fmt.Errorf("KittenTTS sample rate %d, want %d", ready.SampleRate, speech.SynthesisSampleRate), a.stopLocked())
	}
	a.baseURL = strings.TrimRight(ready.URL, "/")
	slog.InfoContext(ctx, "kittentts: KittenTTS ready", "model", a.config.Model, "url", a.baseURL, "voices", ready.Voices)
	go logKittenTTSOutput(ctx, "stdout", a.stdout)
	a.beginWaitLocked()
	return nil
}

func (a *Runtime) readLineLocked(ctx context.Context, name string) ([]byte, error) {
	type readResult struct {
		line []byte
		err  error
	}
	done := make(chan readResult, 1)
	stdout := a.stdout
	go func() {
		line, err := stdout.ReadBytes('\n')
		done <- readResult{line: line, err: err}
	}()
	select {
	case r := <-done:
		if r.err != nil {
			if errors.Is(r.err, io.EOF) {
				return nil, a.processExitErrorLocked("read KittenTTS " + name)
			}
			return nil, fmt.Errorf("read KittenTTS %s: %w", name, r.err)
		}
		return r.line, nil
	case <-ctx.Done():
		return nil, errors.Join(ctx.Err(), a.stopLocked())
	}
}

func (a *Runtime) processExitErrorLocked(op string) error {
	a.beginWaitLocked()
	select {
	case err, ok := <-a.wait:
		if !ok {
			return fmt.Errorf("%s: worker exited", op)
		}
		if err != nil {
			return fmt.Errorf("%s: worker exited: %w", op, err)
		}
		return fmt.Errorf("%s: worker exited", op)
	default:
		return fmt.Errorf("%s: worker closed pipe", op)
	}
}

func (a *Runtime) beginWaitLocked() {
	if a.cmd == nil || a.wait != nil {
		return
	}
	cmd := a.cmd
	ctx := a.processCtx
	wait := make(chan error, 1)
	a.wait = wait
	go func() {
		err := cmd.Wait()
		if ctx.Err() != nil {
			// CommandContext termination is expected on every platform. Kill's exit
			// status differs on Windows, so never classify shutdown by error strings.
			err = nil
		}
		wait <- err
		close(wait)
	}()
}

func (a *Runtime) stop() error {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.stopLocked()
}

func (a *Runtime) stopLocked() error {
	var errs []error
	if a.stdin != nil {
		// exec.Cmd.StdinPipe closes once, so the close in Wait is harmless.
		if err := a.stdin.Close(); err != nil && !errors.Is(err, os.ErrClosed) {
			errs = append(errs, fmt.Errorf("close KittenTTS stdin: %w", err))
		}
	}
	if a.processCancel != nil {
		a.processCancel()
	}
	if a.cmd != nil {
		a.beginWaitLocked()
	}
	if a.wait != nil {
		if err, ok := <-a.wait; ok && err != nil {
			errs = append(errs, err)
		}
	}
	a.processCtx = nil
	a.processCancel = nil
	a.cmd = nil
	a.baseURL = ""
	a.stdin = nil
	a.stdout = nil
	a.wait = nil
	return errors.Join(errs...)
}

type kittenTTSCommandFactory func(context.Context) (*exec.Cmd, error)

type kittenTTSReady struct {
	SampleRate int      `json:"sample_rate"`
	Kind       string   `json:"kind"`
	URL        string   `json:"url"`
	Voices     []string `json:"voices"`
	Error      string   `json:"error"`
}

type kittenTTSRequest struct {
	Text  string `json:"text"`
	Voice string `json:"voice"`
}

func validateKittenTTSURL(rawURL string) error {
	u, err := url.Parse(rawURL)
	if err != nil {
		return fmt.Errorf("parse KittenTTS URL: %w", err)
	}
	if u.Scheme != "http" || u.Host == "" {
		return fmt.Errorf("KittenTTS URL must be an http URL, got %q", rawURL)
	}
	host := u.Hostname()
	if host != "127.0.0.1" && host != "localhost" {
		return fmt.Errorf("KittenTTS URL must be loopback, got %q", rawURL)
	}
	return nil
}

func kittenTTSCommand(ctx context.Context, cfg Config) (*exec.Cmd, error) {
	cache := cfg.CacheDir
	if err := os.MkdirAll(cache, 0o750); err != nil {
		return nil, fmt.Errorf("create KittenTTS cache dir: %w", err)
	}
	args := []string{
		"run",
		"--isolated",
		"--python", kittenTTSPython,
		"--with", kittenTTSPackage,
		"python", "-u", "-c", kittenTTSWorkerScript,
		"--cache-dir", cache,
		"--model", cfg.Model,
	}
	cmd := exec.CommandContext(ctx, "uv", args...)
	cmd.Dir = cache
	// Keep model download progress out of logs without hiding warnings or errors.
	cmd.Env = append(cmd.Environ(), "HF_HUB_DISABLE_PROGRESS_BARS=1")
	return cmd, nil
}

func defaultCacheDir() (string, error) {
	base, err := os.UserCacheDir()
	if err != nil {
		return "", fmt.Errorf("get user cache dir: %w", err)
	}
	return filepath.Join(base, "genaipy", "kittentts"), nil
}

func normalizeConfig(cfg Config) (Config, error) {
	if cfg.Model == "" {
		cfg.Model = kittenTTSModel
	}
	if cfg.Voice == "" {
		cfg.Voice = kittenTTSVoice
	}
	if cfg.CacheDir == "" {
		var err error
		cfg.CacheDir, err = defaultCacheDir()
		if err != nil {
			return Config{}, err
		}
	}
	if !filepath.IsAbs(cfg.CacheDir) {
		return Config{}, errors.New("KittenTTS cache directory must be absolute")
	}
	return cfg, nil
}

func logKittenTTSOutput(ctx context.Context, stream string, r io.Reader) {
	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line != "" {
			slog.InfoContext(ctx, "kittentts: KittenTTS", "stream", stream, "line", line)
		}
	}
	if err := scanner.Err(); err != nil {
		if strings.Contains(err.Error(), "file already closed") {
			return
		}
		slog.WarnContext(ctx, "kittentts: read KittenTTS output", "stream", stream, "err", err)
	}
}
