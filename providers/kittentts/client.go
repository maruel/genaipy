// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package kittentts synthesizes speech through a KittenTTS worker, such as
// one kittenttssrv runs.
package kittentts

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"log/slog"
	"mime"
	"net/http"
	"net/url"
	"strings"
)

// DefaultVoice is the voice New selects when none is given.
const DefaultVoice = "Jasper"

// Client sends synthesis requests to a KittenTTS worker. It is safe for
// concurrent use.
type Client struct {
	client *http.Client
	remote string
}

// New configures a KittenTTS worker at remote, its base URL. client is
// required.
func New(client *http.Client, remote string) (*Client, error) {
	if client == nil {
		return nil, errors.New("KittenTTS HTTP client is required")
	}
	u, err := url.Parse(remote)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" || u.User != nil || u.RawQuery != "" || u.Fragment != "" {
		return nil, errors.New("KittenTTS worker must be an HTTP base URL without credentials, query or fragment")
	}
	return &Client{client: client, remote: strings.TrimRight(remote, "/")}, nil
}

// Synthesize streams text spoken by voice as aligned SampleRate PCM chunks
// with independent owned storage.
//
// voice.Name must be one of the loaded model's voices; empty means
// DefaultVoice. The worker applies its English text normalization only when
// voice.Language is empty or "en", because the upstream normalizer mangles
// numbers, dates and abbreviations in other languages. Stopping iteration
// closes the response. Empty text yields no audio.
func (c *Client) Synthesize(ctx context.Context, voice Voice, text string) iter.Seq2[[]byte, error] {
	return func(yield func([]byte, error) bool) {
		if text == "" {
			return
		}
		data, err := json.Marshal(request{Text: text, Voice: cmp.Or(voice.Name, DefaultVoice), Language: voice.Language})
		if err != nil {
			yield(nil, fmt.Errorf("encode KittenTTS request: %w", err))
			return
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.remote+"/synthesize", bytes.NewReader(data))
		if err != nil {
			yield(nil, fmt.Errorf("create KittenTTS request: %w", err))
			return
		}
		req.Header.Set("Content-Type", "application/json")
		resp, err := c.client.Do(req)
		if err != nil {
			yield(nil, fmt.Errorf("post KittenTTS request: %w", err))
			return
		}
		defer func() {
			if err := resp.Body.Close(); err != nil {
				slog.WarnContext(ctx, "kittentts: close response body", "err", err)
			}
		}()
		if resp.StatusCode != http.StatusOK {
			body, err := io.ReadAll(io.LimitReader(resp.Body, 4096))
			yield(nil, errors.Join(fmt.Errorf("KittenTTS HTTP status %s: %s", resp.Status, strings.TrimSpace(string(body))), err))
			return
		}
		contentType, _, err := mime.ParseMediaType(resp.Header.Get("Content-Type"))
		if err != nil || contentType != "audio/pcm" && contentType != "application/octet-stream" {
			yield(nil, fmt.Errorf("KittenTTS returned %q instead of PCM", resp.Header.Get("Content-Type")))
			return
		}
		for pcm, err := range pcmChunks(resp.Body) {
			if !yield(pcm, err) {
				return
			}
		}
	}
}

type request struct {
	Text     string `json:"text"`
	Voice    string `json:"voice"`
	Language string `json:"language,omitzero"`
}
