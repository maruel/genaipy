// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Managed Cactus Whistle ASR process and completed-utterance adapter.
package whistle

import (
	"bufio"
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"
)

//go:embed worker.py
var whistleWorker string

// New starts a worker, installing cactus-needle 3.1.0 with uv and downloading
// weights when needed. ctx owns the worker lifetime; call Close when finished.
// model optionally selects a local .cact file.
//
// Whistle requires a prebuilt Needle engine, fetched by this pinned Python
// package. No genai/llama.cpp dependency is involved. model, when set, is a
// local .cact weights path; otherwise Needle downloads Cactus-Compute/whistle.
func New(ctx context.Context, model string) (*Runtime, error) {
	procCtx, cancel := context.WithCancel(ctx)
	cmd := exec.CommandContext(procCtx, "uv", "run", "--isolated", "--no-project", "--python", "3.12", "--with", "cactus-needle==3.1.0", "python", "-u", "-c", whistleWorker, "--model", model)
	cmd.Env = append(os.Environ(), "NEEDLE_TELEMETRY=0", "DO_NOT_TRACK=1")
	w, err := launchWhistle(procCtx, cancel, cmd)
	if err != nil {
		cancel()
	}
	return w, err
}

// Runtime owns a persistent Whistle worker. Close stops it and is idempotent.
type Runtime struct {
	ctx       context.Context
	cancel    context.CancelFunc
	stdin     io.Closer
	done      chan struct{}
	err       error // Written before done closes.
	closeOnce sync.Once
	client    *http.Client
	url       string
}

// launchWhistle owns the child and both its pipes. The command boundary also
// permits exercising the launcher offline without installing the native engine.
func launchWhistle(ctx context.Context, cancel context.CancelFunc, cmd *exec.Cmd) (*Runtime, error) {
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	rd, wr := io.Pipe()
	cmd.Stdout = wr
	cmd.Stderr = slog.NewLogLogger(slog.Default().Handler(), slog.LevelInfo).Writer()
	if err := cmd.Start(); err != nil {
		return nil, errors.Join(err, stdin.Close(), rd.Close(), wr.Close())
	}
	w := &Runtime{ctx: ctx, cancel: cancel, stdin: stdin, done: make(chan struct{}), client: &http.Client{Transport: &http.Transport{Proxy: nil}}}
	ready := make(chan string, 1)
	go func() {
		defer func() { _ = rd.Close() }()
		sc := bufio.NewScanner(rd)
		if sc.Scan() {
			ready <- sc.Text()
		} else {
			ready <- ""
		}
		for sc.Scan() {
			slog.InfoContext(ctx, "whistle worker", "line", sc.Text())
		}
		if err := sc.Err(); err != nil {
			slog.WarnContext(ctx, "whistle stdout", "err", err)
		}
	}()
	go func() {
		w.err = cmd.Wait()
		if ctx.Err() != nil {
			w.err = nil
		}
		_ = wr.Close()
		close(w.done)
	}()
	// uv may spawn a Python child: cancellation must close its stdin lifeline.
	stop := context.AfterFunc(ctx, func() { _ = stdin.Close() })
	go func() { <-w.done; stop() }()
	timer := time.NewTimer(10 * time.Minute)
	defer timer.Stop()
	select {
	case line := <-ready:
		var msg struct {
			URL string `json:"url"`
		}
		if err := json.Unmarshal([]byte(line), &msg); err != nil {
			return nil, errors.Join(fmt.Errorf("whistle readiness: %w", err), w.Close())
		}
		u, err := url.Parse(msg.URL)
		if err != nil || u.Scheme != "http" || u.Hostname() != "127.0.0.1" || u.Port() == "" || u.Path != "" || u.User != nil || u.RawQuery != "" || u.Fragment != "" {
			return nil, errors.Join(fmt.Errorf("invalid Whistle readiness URL %q", msg.URL), w.Close())
		}
		w.url = msg.URL
		return w, nil
	case <-ctx.Done():
		return nil, errors.Join(ctx.Err(), w.Close())
	case <-timer.C:
		return nil, errors.Join(errors.New("whistle startup timed out"), w.Close())
	}
}

// Close stops the worker and reports any unexpected process exit.
func (w *Runtime) Close() error {
	w.closeOnce.Do(func() { w.cancel(); _ = w.stdin.Close() })
	<-w.done
	w.client.CloseIdleConnections()
	return w.err
}

// Transcribe accepts mono S16LE PCM at 16 kHz. Long clips are split into
// 30-second windows. Cancelling a request leaves the worker available.
func (w *Runtime) Transcribe(ctx context.Context, pcm []byte, sampleRate int) (string, error) {
	if sampleRate != 16000 {
		return "", errors.New("whistle requires 16 kHz audio")
	}
	if len(pcm)%2 != 0 {
		return "", errors.New("whistle requires aligned S16LE PCM")
	}
	if err := w.ctx.Err(); err != nil {
		return "", err
	}
	select {
	case <-w.done:
		return "", errors.Join(errors.New("whistle worker exited"), w.err)
	default:
	}
	var parts []string
	// The encoder accepts at most 30 seconds. Long utterances use consecutive
	// windows; a word crossing the boundary can lose accuracy.
	const window = 16000 * 2 * 30
	for len(pcm) > 0 {
		n := min(len(pcm), window)
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, w.url+"/transcribe", bytes.NewReader(pcm[:n]))
		if err != nil {
			return "", err
		}
		req.Header.Set("Content-Type", "application/octet-stream")
		resp, err := w.client.Do(req)
		if err != nil {
			return "", fmt.Errorf("whistle transcription: %w", err)
		}
		var result struct {
			Text  string `json:"text"`
			Error string `json:"error"`
		}
		err = json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&result)
		err = errors.Join(err, resp.Body.Close())
		if err != nil {
			return "", fmt.Errorf("whistle response: %w", err)
		}
		if resp.StatusCode != http.StatusOK || result.Error != "" {
			return "", fmt.Errorf("whistle HTTP %s: %s", resp.Status, result.Error)
		}
		if text := strings.TrimSpace(result.Text); text != "" {
			parts = append(parts, text)
		}
		pcm = pcm[n:]
	}
	return strings.Join(parts, " "), nil
}
