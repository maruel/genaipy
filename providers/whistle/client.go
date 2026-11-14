// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package whistle transcribes complete utterances through a Cactus Whistle
// worker, such as one whistlesrv runs, in genai transcription sessions.
package whistle

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"slices"
	"strings"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/genaipy/internal/batch"
	"golang.org/x/text/language"
)

// DefaultModel is the Hugging Face model the worker loads without local
// weights.
const DefaultModel = "Cactus-Compute/whistle"

// Client is a genai.Provider whose ConnectRealtime opens transcription
// sessions; other provider methods return base.ErrNotSupported.
type Client struct {
	base.NotImplemented

	client *http.Client
	remote string
	model  string
}

// New configures a Whistle worker at remote, its base URL.
//
// client is required. model names the weights the worker loaded for ModelID;
// empty means DefaultModel.
func New(client *http.Client, remote, model string) (*Client, error) {
	if client == nil {
		return nil, errors.New("whistle HTTP client is required")
	}
	u, err := url.Parse(remote)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" || u.User != nil || u.RawQuery != "" || u.Fragment != "" {
		return nil, errors.New("whistle worker must be an HTTP base URL without credentials, query or fragment")
	}
	if model == "" {
		model = DefaultModel
	}
	return &Client{client: client, remote: strings.TrimRight(remote, "/"), model: model}, nil
}

// ConnectRealtime implements genai.Provider. It requires
// genai.RealtimeOptionTranscription and transcribes one utterance of 16 kHz
// PCM, buffering up to 10 minutes until Commit; it yields no drafts. Whistle
// has no language input and detects the languages listed in the
// Cactus-Compute/whistle model card, so RealtimeOptionAudio.InputLanguage is
// only checked against them; another is base.ErrNotSupported.
func (w *Client) ConnectRealtime(_ context.Context, opts ...genai.RealtimeOption) (genai.RealtimeSession, error) {
	transcription := false
	var unsupported []string
	for _, opt := range opts {
		if opt == nil {
			return nil, errors.New("realtime option must not be nil")
		}
		if err := opt.Validate(); err != nil {
			return nil, fmt.Errorf("realtime option %T: %w", opt, err)
		}
		switch o := opt.(type) {
		case genai.RealtimeOptionTranscription:
			transcription = true
		case *genai.RealtimeOptionAudio:
			if o.Voice != "" {
				unsupported = append(unsupported, "voice")
			}
			if o.OutputTranscription {
				unsupported = append(unsupported, "output transcription")
			}
			if r := o.InputFormat.SampleRate; r != 0 && r != sampleRate {
				unsupported = append(unsupported, fmt.Sprintf("input sample rate %d Hz", r))
			}
			if o.InputLanguage != "" {
				// Validate parsed it.
				b, _ := language.MustParse(o.InputLanguage).Base()
				if !slices.Contains(languages, b.String()) {
					unsupported = append(unsupported, "input language "+o.InputLanguage)
				}
			}
		case genai.RealtimeTurnDetection:
			if o == genai.RealtimeTurnServer {
				unsupported = append(unsupported, "server turn detection")
			}
		default:
			unsupported = append(unsupported, fmt.Sprintf("%T", opt))
		}
	}
	if !transcription {
		unsupported = append(unsupported, "conversation")
	}
	if len(unsupported) != 0 {
		return nil, fmt.Errorf("whistle realtime: %w", &base.ErrNotSupported{Options: unsupported})
	}
	return batch.NewSession(sampleRate, sampleRate*2*600, w.transcribe), nil
}

// Name implements genai.Provider.
func (w *Client) Name() string { return "whistle" }

// ModelID implements genai.Provider.
func (w *Client) ModelID() string { return w.model }

// OutputModalities implements genai.Provider.
func (w *Client) OutputModalities() genai.Modalities { return genai.Modalities{genai.ModalityText} }

// Capabilities implements genai.Provider.
func (w *Client) Capabilities() genai.ProviderCapabilities {
	return genai.ProviderCapabilities{Realtime: true}
}

// Scoreboard implements genai.Provider. No smoke qualification exists.
func (w *Client) Scoreboard() scoreboard.Score { return scoreboard.Score{} }

// HTTPClient implements genai.Provider.
func (w *Client) HTTPClient() *http.Client { return w.client }

// Close implements genai.Provider. The client owns no resources.
func (w *Client) Close() error { return nil }

// sampleRate is Whistle's only input rate.
const sampleRate = 16000

// languages are the ISO 639 base languages in the Cactus-Compute/whistle model
// card.
var languages = []string{"de", "en", "es", "fr", "it", "nl", "pl"}

// transcribe recognizes complete 16 kHz PCM. Long clips are split into
// 30-second windows. Cancelling a request leaves the worker available.
func (w *Client) transcribe(ctx context.Context, pcm []byte) (string, error) {
	var parts []string
	// The encoder accepts at most 30 seconds. Long utterances use consecutive
	// windows; a word crossing the boundary can lose accuracy.
	const window = 16000 * 2 * 30
	for len(pcm) > 0 {
		n := min(len(pcm), window)
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, w.remote+"/transcribe", bytes.NewReader(pcm[:n]))
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

var _ genai.Provider = (*Client)(nil)
