// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package worker runs a loopback HTTP worker process whose stdin is its
// lifeline and whose first stdout line announces its URL.
package worker

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/url"
	"os"
	"os/exec"
	"sync"
	"time"

	"github.com/maruel/genai/base"
)

// startupTimeout bounds readiness, which may include a package install and a
// model download.
const startupTimeout = 10 * time.Minute

// Process owns a started worker.
type Process struct {
	cancel    context.CancelFunc
	stdin     io.Closer
	done      chan struct{}
	err       error // Written before done closes.
	closeOnce sync.Once
}

// Start runs the command newCmd builds with the context that owns the
// process, and waits for the worker's readiness line: a JSON object whose
// "url" member is http://127.0.0.1:<port> without a path. ready first
// validates the worker's other members, such as a startup error. Later stdout
// lines and stderr are logged under name.
//
// ctx owns the process lifetime. The worker must exit when its stdin reaches
// EOF: a launcher such as uv may run it as a grandchild that killing the
// launcher would orphan.
func Start(ctx context.Context, name string, newCmd func(context.Context) *exec.Cmd, ready func([]byte) error) (*Process, string, error) {
	procCtx, cancel := context.WithCancel(ctx)
	cmd := newCmd(procCtx)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		cancel()
		return nil, "", err
	}
	// An io.Pipe instead of StdoutPipe lets Wait finish copying before the
	// reader sees EOF.
	rd, wr := io.Pipe()
	cmd.Stdout = wr
	cmd.Stderr = slog.NewLogLogger(slog.Default().Handler(), slog.LevelInfo).Writer()
	if err := cmd.Start(); err != nil {
		// Start already closed the stdin pipe.
		cancel()
		return nil, "", errors.Join(fmt.Errorf("%s: start: %w", name, err), rd.Close(), wr.Close())
	}
	p := &Process{cancel: cancel, stdin: stdin, done: make(chan struct{})}
	// The scanner reuses its buffer, so the line is copied.
	lines := make(chan []byte, 1)
	go func() {
		defer func() { _ = rd.Close() }()
		sc := bufio.NewScanner(rd)
		if sc.Scan() {
			lines <- append([]byte{}, sc.Bytes()...)
		} else {
			close(lines)
		}
		for sc.Scan() {
			slog.InfoContext(ctx, name, "line", sc.Text())
		}
		if err := sc.Err(); err != nil {
			slog.WarnContext(ctx, name+" stdout", "err", err)
		}
	}()
	go func() {
		// Cancellation kills the launcher, as exec.CommandContext does by
		// default; the worker exits on its own when stdin closes.
		p.err = base.ServerExit(cmd.Wait(), cmd.ProcessState, procCtx.Err() != nil, os.Kill)
		_ = wr.Close()
		close(p.done)
	}()
	stop := context.AfterFunc(procCtx, func() { _ = stdin.Close() })
	go func() { <-p.done; stop() }()
	timer := time.NewTimer(startupTimeout)
	defer timer.Stop()
	select {
	case line, ok := <-lines:
		if !ok {
			// Waiting for the exit keeps its status, which a requested stop discards.
			select {
			case <-p.done:
			case <-ctx.Done():
			}
			return nil, "", errors.Join(fmt.Errorf("%s exited during startup", name), p.Close())
		}
		err := ready(line)
		u := ""
		if err == nil {
			u, err = readyURL(line)
		}
		if err != nil {
			return nil, "", errors.Join(fmt.Errorf("%s readiness: %w", name, err), p.Close())
		}
		return p, u, nil
	case <-ctx.Done():
		return nil, "", errors.Join(ctx.Err(), p.Close())
	case <-timer.C:
		return nil, "", errors.Join(fmt.Errorf("%s startup timed out", name), p.Close())
	}
}

// Close stops the worker and reports an exit it did not request. It is
// idempotent.
func (p *Process) Close() error {
	p.closeOnce.Do(func() { p.cancel(); _ = p.stdin.Close() })
	<-p.done
	return p.err
}

// Done is closed when the worker exits, as genai.Server specifies.
func (p *Process) Done() <-chan struct{} { return p.done }

// Err is the exit result once Done is closed, as genai.Server specifies.
func (p *Process) Err() error {
	select {
	case <-p.done:
		return p.err
	default:
		return nil
	}
}

func readyURL(line []byte) (string, error) {
	var msg struct {
		URL string `json:"url"`
	}
	if err := json.Unmarshal(line, &msg); err != nil {
		return "", err
	}
	u, err := url.Parse(msg.URL)
	if err != nil || u.Scheme != "http" || u.Hostname() != "127.0.0.1" || u.Port() == "" || u.Path != "" || u.User != nil || u.RawQuery != "" || u.Fragment != "" {
		return "", fmt.Errorf("invalid URL %q", msg.URL)
	}
	return msg.URL, nil
}
