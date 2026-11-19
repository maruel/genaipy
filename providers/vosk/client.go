// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package vosk transcribes speech through a Vosk server, such as one vosksrv
// runs or alphacep/vosk-server's WebSocket server.
//
// See https://alphacephei.com/vosk/server.
package vosk

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"net/http"
	"net/url"
	"slices"
	"strings"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/scoreboard"
	"golang.org/x/text/language"
)

// Model names a Vosk model archive in https://alphacephei.com/vosk/models.
//
// A Vosk model recognizes one language and has no language input.
type Model string

// Languages returns the ISO 639 base language that m transcribes;
// ConnectRealtime rejects another input language. It returns nil for a model
// New rejects.
//
// Immutable: callers share the result.
func (m Model) Languages() []string { return modelLanguages[m] }

// English is the 1.8 GB generic US English server model. The worker serving
// it peaked at 6.7 GiB resident on Linux x86-64.
const English Model = "vosk-model-en-us-0.22"

// Models are the models New accepts, sorted.
//
// Immutable.
var Models = slices.Sorted(maps.Keys(modelLanguages))

// modelLanguages owns the model set: each model New accepts and its language.
var modelLanguages = map[Model][]string{
	English: {"en"},
}

// Client transcribes mono S16LE PCM through a Vosk server in genai
// transcription sessions; other provider methods return base.ErrNotSupported.
type Client struct {
	base.NotImplemented

	client *http.Client
	remote string
	model  Model
}

// New configures a Vosk server that loaded model.
//
// client is required and can supply authentication through its transport.
// remote is the server's HTTP or WebSocket base URL, including any path. The
// server serves its loaded model; model only declares its language.
func New(client *http.Client, remote string, model Model) (*Client, error) {
	if client == nil {
		return nil, errors.New("vosk HTTP client is required")
	}
	u, err := url.Parse(remote)
	if err != nil || !slices.Contains([]string{"http", "https", "ws", "wss"}, u.Scheme) || u.Host == "" || u.User != nil || u.RawQuery != "" || u.Fragment != "" {
		return nil, errors.New("vosk server must be an HTTP or WebSocket URL without credentials, query or fragment")
	}
	if model.Languages() == nil {
		return nil, fmt.Errorf("unsupported Vosk model %q", model)
	}
	return &Client{client: client, remote: remote, model: model}, nil
}

// ConnectRealtime implements genai.Provider. It requires
// genai.RealtimeOptionTranscription and streams one utterance of 16 to 96 kHz
// PCM; RealtimeOptionAudio.InputFormat defaults to 16 kHz. Vosk downsamples to
// the models' 16 kHz and never upsamples. RealtimeOptionAudio.InputLanguage is
// only checked against the model's language; another is base.ErrNotSupported.
func (c *Client) ConnectRealtime(ctx context.Context, opts ...genai.RealtimeOption) (genai.RealtimeSession, error) {
	transcription := false
	format := genai.AudioFormat{SampleRate: 16000}
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
			if o.InputFormat.SampleRate != 0 {
				format = o.InputFormat
			}
			if o.InputLanguage != "" {
				// Validate parsed it.
				b, _ := language.MustParse(o.InputLanguage).Base()
				if !slices.Contains(c.model.Languages(), b.String()) {
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
	if format.SampleRate < 16000 || format.SampleRate > 96000 {
		unsupported = append(unsupported, fmt.Sprintf("input sample rate %d Hz", format.SampleRate))
	}
	if len(unsupported) != 0 {
		return nil, fmt.Errorf("vosk realtime: %w", &base.ErrNotSupported{Options: unsupported})
	}
	return dialSession(ctx, c.client, strings.TrimRight(c.remote, "/"), format)
}

// Name implements genai.Provider.
func (c *Client) Name() string { return "vosk" }

// ModelID implements genai.Provider.
func (c *Client) ModelID() string { return string(c.model) }

// OutputModalities implements genai.Provider.
func (c *Client) OutputModalities() genai.Modalities { return genai.Modalities{genai.ModalityText} }

// Capabilities implements genai.Provider.
func (c *Client) Capabilities() genai.ProviderCapabilities {
	return genai.ProviderCapabilities{Realtime: true}
}

// Scoreboard implements genai.Provider. No smoke qualification exists.
func (c *Client) Scoreboard() scoreboard.Score { return scoreboard.Score{} }

// HTTPClient implements genai.Provider.
func (c *Client) HTTPClient() *http.Client { return c.client }

// Close implements genai.Provider. The client owns no resources.
func (c *Client) Close() error { return nil }

var _ genai.Provider = (*Client)(nil)
