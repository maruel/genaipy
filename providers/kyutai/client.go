// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package kyutai transcribes live speech with Kyutai STT through moshi-server's
// /api/asr-streaming WebSocket, served by a stock moshi-server or by
// kyutaisrv.
//
// The protocol is defined by rust/moshi-server/src/asr.rs in kyutai-labs/moshi.
// The models are documented in kyutai-labs/delayed-streams-modeling.
package kyutai

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/scoreboard"
	"golang.org/x/text/language"
)

// SampleRate is the mono PCM rate in Hz that the models transcribe.
const SampleRate = 24000

// Model is a Kyutai STT model, named by its PyTorch Hugging Face repository.
type Model string

// Languages returns the sorted ISO 639 base languages that m transcribes,
// from its model card; ConnectRealtime rejects another input language. It
// returns nil for a model New rejects.
//
// Immutable: callers share the result.
func (m Model) Languages() []string { return models[m].languages }

const (
	// STT1B is kyutai/stt-1b-en_fr: English and French with a 0.5 s delay.
	STT1B Model = "kyutai/stt-1b-en_fr"
	// STT2B6 is kyutai/stt-2.6b-en: English with a 2.5 s delay.
	STT2B6 Model = "kyutai/stt-2.6b-en"
)

// Models are the models New accepts, sorted.
//
// Immutable.
var Models = slices.Sorted(maps.Keys(models))

// Client transcribes speech in genai transcription sessions; other provider
// methods return base.ErrNotSupported.
type Client struct {
	base.NotImplemented

	client *http.Client
	remote string
	model  Model
	apiKey string
}

// New configures a server at remote, its HTTP base URL, that loaded model.
//
// client is required. apiKey goes in the kyutai-api-key header, which
// moshi-server checks against its authorized_ids; kyutaisrv ignores it. An
// empty apiKey sends no header.
func New(client *http.Client, remote string, model Model, apiKey string) (*Client, error) {
	if client == nil {
		return nil, errors.New("kyutai HTTP client is required")
	}
	u, err := url.Parse(remote)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" || u.User != nil || u.RawQuery != "" || u.Fragment != "" {
		return nil, errors.New("kyutai server must be an HTTP base URL without credentials, query or fragment")
	}
	if model.Languages() == nil {
		return nil, fmt.Errorf("unsupported Kyutai STT model %q; use one of %q", model, Models)
	}
	return &Client{client: client, remote: strings.TrimRight(remote, "/"), model: model, apiKey: apiKey}, nil
}

// ConnectRealtime implements genai.Provider. It requires
// genai.RealtimeOptionTranscription and transcribes one utterance of
// SampleRate PCM, the default RealtimeOptionAudio.InputFormat. The model has
// no language input: RealtimeOptionAudio.InputLanguage is only checked against
// the model's Languages, and another language is base.ErrNotSupported.
func (a *Client) ConnectRealtime(ctx context.Context, opts ...genai.RealtimeOption) (genai.RealtimeSession, error) {
	transcription := false
	var unsupported []string
	lang := language.Und
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
			if r := o.InputFormat.SampleRate; r != 0 && r != SampleRate {
				unsupported = append(unsupported, fmt.Sprintf("input sample rate %d Hz", r))
			}
			if o.InputLanguage != "" {
				// Validate parsed it.
				lang = language.MustParse(o.InputLanguage)
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
		return nil, fmt.Errorf("kyutai realtime: %w", &base.ErrNotSupported{Options: unsupported})
	}
	if b, _ := lang.Base(); lang != language.Und && !slices.Contains(a.model.Languages(), b.String()) {
		return nil, fmt.Errorf("kyutai model %s does not support spoken language %q: %w", a.model, lang, &base.ErrNotSupported{Options: []string{"input language " + lang.String()}})
	}
	return dialSession(ctx, a.client, a.remote, a.apiKey, models[a.model])
}

// Name implements genai.Provider.
func (a *Client) Name() string { return "kyutai" }

// ModelID implements genai.Provider.
func (a *Client) ModelID() string { return string(a.model) }

// OutputModalities implements genai.Provider.
func (a *Client) OutputModalities() genai.Modalities { return genai.Modalities{genai.ModalityText} }

// Capabilities implements genai.Provider.
func (a *Client) Capabilities() genai.ProviderCapabilities {
	return genai.ProviderCapabilities{Realtime: true}
}

// Scoreboard implements genai.Provider. It declares the transcription session
// at SampleRate, whose transcript streams word by word.
func (a *Client) Scoreboard() scoreboard.Score {
	names := make([]string, len(Models))
	for i, m := range Models {
		names[i] = string(m)
	}
	return scoreboard.Score{
		Country: "Local",
		Realtime: []scoreboard.Realtime{{

			Models: names,
			In:     map[scoreboard.Modality]scoreboard.ModalCapability{scoreboard.ModalityAudio: {Inline: true}},
			Out:    map[scoreboard.Modality]scoreboard.ModalCapability{scoreboard.ModalityText: {Inline: true}},
			Realtime: &scoreboard.RealtimeFunctionality{
				InputSampleRate: SampleRate,
				Transcription:   &scoreboard.RealtimeTranscription{Drafts: true, InputLanguage: true},
			},
		}},
	}
}

// HTTPClient implements genai.Provider.
func (a *Client) HTTPClient() *http.Client { return a.client }

// Close implements genai.Provider. The client owns no resources.
func (a *Client) Close() error { return nil }

var _ genai.Provider = (*Client)(nil)

// modelInfo describes a model; delay is the audio_delay_seconds of
// stt_config in its config.json at the revision kyutaisrv pins.
type modelInfo struct {
	languages []string
	// delay separates speech from its transcript.
	delay time.Duration
}

var models = map[Model]modelInfo{
	STT1B:  {languages: []string{"en", "fr"}, delay: 500 * time.Millisecond},
	STT2B6: {languages: []string{"en"}, delay: 2500 * time.Millisecond},
}
