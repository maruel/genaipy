// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package nemospeech transcribes speech through an NVIDIA NeMo-Speech.cpp
// server, such as one nemospeechsrv runs.
//
// See https://github.com/NVIDIA/NeMo-Speech.cpp/blob/main/docs/api.md.
package nemospeech

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"maps"
	"mime/multipart"
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

// Model names an ASR model in NeMo-Speech.cpp's model index
// (models/index.json in NVIDIA/NeMo-Speech.cpp v0.3.0).
//
// NeMo-Speech.cpp ignores a request language for models without language
// prompts and silently falls back to detection for an unknown locale, so the
// language support of each model is declared here from its model card.
type Model string

// Languages returns the sorted ISO 639 base languages that m transcribes,
// from its model card; ConnectRealtime rejects another input language. It
// returns nil for a model New rejects.
//
// Immutable: callers share the result.
func (m Model) Languages() []string { return modelLanguages[m] }

const (
	// NemotronASR is nvidia/nemotron-3.5-asr-streaming-0.6b. A language selects
	// its closest language-ID prompt among the 32 transcription-ready and
	// broad-coverage locales; the 8 adaptation-ready locales need fine-tuning.
	// Without one, it detects the language.
	//
	// Measured on NeMo-Speech.cpp 0.3.0 (CPU, q8_0): detection identified
	// English, French and German clips; a matching locale left transcripts
	// unchanged, while a mismatched one returned empty or truncated text.
	// English speech forced to fr-FR returned "".
	NemotronASR Model = "nemotron-3.5"
	// NemotronASREnglish is nvidia/nemotron-speech-streaming-en-0.6b, English only.
	NemotronASREnglish Model = "nemotron-en"
	// ParakeetCTC is nvidia/parakeet-ctc-1.1b, English only.
	ParakeetCTC Model = "parakeet-ctc"
	// ParakeetTDT is nvidia/parakeet-tdt-0.6b-v3. It detects 25 European
	// languages and has no language input. Measured on NeMo-Speech.cpp 0.3.0
	// (CPU, q8_0): English, French and German clips were transcribed in their
	// spoken language.
	ParakeetTDT Model = "parakeet-tdt"
)

// Models are the models New accepts, sorted.
//
// Immutable.
var Models = slices.Sorted(maps.Keys(modelLanguages))

// Client transcribes mono S16LE PCM through a NeMo-Speech.cpp server in
// genai transcription sessions; other provider methods return
// base.ErrNotSupported.
type Client struct {
	base.NotImplemented

	client *http.Client
	remote string
	model  Model
	lang   languageMapper
}

// New configures a NeMo-Speech.cpp server that loaded model.
//
// client is required and can supply authentication through its transport.
// remote is the server's base URL, including any path prefix. The server
// serves its loaded model; model selects the language mapping and whether
// ConnectRealtime is available.
func New(client *http.Client, remote string, model Model) (*Client, error) {
	if err := validateHTTP(client, remote); err != nil {
		return nil, err
	}
	r := &Client{client: client, remote: strings.TrimRight(remote, "/"), model: model}
	switch langs := model.Languages(); {
	case langs == nil:
		return nil, fmt.Errorf("unsupported NeMo-Speech model %q", model)
	case model == NemotronASR:
		r.lang = localeLanguages(nemotronLocales...)
	default:
		r.lang = detectedLanguages(langs...)
	}
	return r, nil
}

// ConnectRealtime implements genai.Provider. It requires
// genai.RealtimeOptionTranscription and transcribes one utterance of 8 to
// 96 kHz PCM; RealtimeOptionAudio.InputFormat defaults to 16 kHz.
// RealtimeOptionAudio.InputLanguage goes through the model's language mapping,
// and a language the model does not support is base.ErrNotSupported.
// Nemotron and Parakeet CTC stream over the server's realtime WebSocket.
// ParakeetTDT cannot stream: its session buffers up to 10 minutes of audio
// and posts it to the batch endpoint at Commit, so it yields no drafts.
func (a *Client) ConnectRealtime(ctx context.Context, opts ...genai.RealtimeOption) (genai.RealtimeSession, error) {
	transcription := false
	format := genai.AudioFormat{SampleRate: 16000}
	lang := language.Und
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
	// The server's accepted input range.
	if format.SampleRate < 8000 || format.SampleRate > 96000 {
		unsupported = append(unsupported, fmt.Sprintf("input sample rate %d Hz", format.SampleRate))
	}
	if len(unsupported) != 0 {
		return nil, fmt.Errorf("NeMo-Speech realtime: %w", &base.ErrNotSupported{Options: unsupported})
	}
	code, err := a.lang(lang)
	if err != nil {
		return nil, err
	}
	if a.model == ParakeetTDT {
		rate := format.SampleRate
		return batch.NewSession(rate, rate*2*600, func(ctx context.Context, pcm []byte) (string, error) {
			return a.transcribe(ctx, pcm, rate, code)
		}), nil
	}
	return dialSession(ctx, a.client, a.remote, format, code)
}

// Name implements genai.Provider.
func (a *Client) Name() string { return "nemospeech" }

// ModelID implements genai.Provider. It is the model index alias.
func (a *Client) ModelID() string { return string(a.model) }

// OutputModalities implements genai.Provider.
func (a *Client) OutputModalities() genai.Modalities { return genai.Modalities{genai.ModalityText} }

// Capabilities implements genai.Provider.
func (a *Client) Capabilities() genai.ProviderCapabilities {
	return genai.ProviderCapabilities{Realtime: true}
}

// Scoreboard implements genai.Provider. No smoke qualification exists.
func (a *Client) Scoreboard() scoreboard.Score { return scoreboard.Score{} }

// HTTPClient implements genai.Provider.
func (a *Client) HTTPClient() *http.Client { return a.client }

// Close implements genai.Provider. The client owns no resources.
func (a *Client) Close() error { return nil }

// transcribe posts complete audio to the batch endpoint with the mapped
// language code. Empty audio returns empty text without making a request.
// Malformed responses and server failures return errors.
func (a *Client) transcribe(ctx context.Context, pcm []byte, sampleRate int, code string) (string, error) {
	if len(pcm) == 0 {
		return "", nil
	}
	wav := pcmWAV(pcm, sampleRate)
	// An empty model field selects the loaded model; Model is an index
	// alias, not the served model id.
	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	file, err := form.CreateFormFile("file", "speech.wav")
	if err != nil {
		return "", err
	}
	if _, err := file.Write(wav); err != nil {
		return "", err
	}
	if code != "" {
		if err := form.WriteField("language", code); err != nil {
			return "", err
		}
	}
	if err := form.WriteField("response_format", "json"); err != nil {
		return "", err
	}
	if err := form.Close(); err != nil {
		return "", err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, a.remote+"/v1/audio/transcriptions", &body)
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", form.FormDataContentType())
	resp, err := a.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("transcribe audio: %w", err)
	}
	defer closeResponse(ctx, resp.Body)
	if resp.StatusCode != http.StatusOK {
		return "", statusError(resp)
	}
	data, err := io.ReadAll(io.LimitReader(resp.Body, (1<<20)+1))
	if err != nil {
		return "", fmt.Errorf("read transcription: %w", err)
	}
	if len(data) > 1<<20 {
		return "", errors.New("transcription response exceeds 1 MiB")
	}
	var result struct {
		Text  *string `json:"text"`
		Error *struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(data, &result); err != nil {
		return "", fmt.Errorf("decode transcription: %w", err)
	}
	if result.Error != nil {
		return "", fmt.Errorf("transcription server error: %s", result.Error.Message)
	}
	if result.Text == nil {
		return "", errors.New("transcription response has no text")
	}
	return strings.TrimSpace(*result.Text), nil
}

func validateHTTP(client *http.Client, remote string) error {
	if client == nil {
		return errors.New("NeMo-Speech HTTP client is required")
	}
	u, err := url.Parse(remote)
	if err != nil {
		return fmt.Errorf("parse speech server URL: %w", err)
	}
	if (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" || u.User != nil || u.RawQuery != "" || u.Fragment != "" {
		return errors.New("NeMo-Speech server must be an HTTP base URL without credentials, query or fragment")
	}
	return nil
}

func closeResponse(ctx context.Context, body io.Closer) {
	if err := body.Close(); err != nil {
		slog.WarnContext(ctx, "speech: close HTTP response", "err", err)
	}
}

func statusError(resp *http.Response) error {
	data, err := io.ReadAll(io.LimitReader(resp.Body, 4096))
	return errors.Join(fmt.Errorf("audio server HTTP %s: %s", resp.Status, strings.TrimSpace(string(data))), err)
}

func pcmWAV(pcm []byte, sampleRate int) []byte {
	// ConnectRealtime bounds the rate to 96 kHz and the audio to 10 minutes of
	// whole samples, which fit the WAV header's 32-bit fields.
	out := make([]byte, 44+len(pcm))
	copy(out, "RIFF")
	binary.LittleEndian.PutUint32(out[4:], uint32(36+len(pcm)))
	copy(out[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(out[16:], 16)
	binary.LittleEndian.PutUint16(out[20:], 1)
	binary.LittleEndian.PutUint16(out[22:], 1)
	binary.LittleEndian.PutUint32(out[24:], uint32(sampleRate))
	binary.LittleEndian.PutUint32(out[28:], uint32(sampleRate*2))
	binary.LittleEndian.PutUint16(out[32:], 2)
	binary.LittleEndian.PutUint16(out[34:], 16)
	copy(out[36:], "data")
	binary.LittleEndian.PutUint32(out[40:], uint32(len(pcm)))
	copy(out[44:], pcm)
	return out
}

var _ genai.Provider = (*Client)(nil)
