// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// HTTP recognition and synthesis clients for local audio servers.

package speech

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"log/slog"
	"mime"
	"mime/multipart"
	"net/http"
	"net/url"
	"strings"
)

// TranscriptionProtocol selects the server's transcription request contract.
type TranscriptionProtocol string

const (
	// OpenAI uses /v1/audio/transcriptions and a model form field.
	OpenAI TranscriptionProtocol = "openai"
	// WhisperCPP uses /inference; the server selects its model.
	WhisperCPP TranscriptionProtocol = "whispercpp"
)

// HTTPRecognizer transcribes mono S16LE PCM over HTTP.
type HTTPRecognizer struct {
	client   *http.Client
	remote   string
	model    string
	protocol TranscriptionProtocol
}

// NewHTTPRecognizer configures a transcription server and its request protocol.
//
// client is required and can supply authentication through its transport.
// remote is the server's base URL, including any path prefix. OpenAI requires
// a model; WhisperCPP rejects one because the server owns model selection.
func NewHTTPRecognizer(client *http.Client, remote, model string, protocol TranscriptionProtocol) (*HTTPRecognizer, error) {
	if err := validateHTTP(client, remote); err != nil {
		return nil, err
	}
	switch protocol {
	case OpenAI:
		if model == "" {
			return nil, errors.New("transcription model is required for OpenAI")
		}
	case WhisperCPP:
		if model != "" {
			return nil, errors.New("WhisperCPP model is selected by the server")
		}
	default:
		return nil, fmt.Errorf("unsupported transcription protocol %q", protocol)
	}
	return &HTTPRecognizer{client: client, remote: strings.TrimRight(remote, "/"), model: model, protocol: protocol}, nil
}

// Transcribe recognizes speech captured at sampleRate Hz.
//
// Empty audio returns empty text without making a request. Invalid PCM or
// sample rates, malformed responses, and server failures return errors.
func (a *HTTPRecognizer) Transcribe(ctx context.Context, pcm []byte, sampleRate int) (string, error) {
	wav, err := pcmWAV(pcm, sampleRate)
	if err != nil {
		return "", err
	}
	if len(pcm) == 0 {
		return "", nil
	}
	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	file, err := form.CreateFormFile("file", "speech.wav")
	if err != nil {
		return "", err
	}
	if _, err := file.Write(wav); err != nil {
		return "", err
	}
	path := "/v1/audio/transcriptions"
	if a.protocol == WhisperCPP {
		path = "/inference"
	} else if err := form.WriteField("model", a.model); err != nil {
		return "", err
	}
	if err := form.WriteField("response_format", "json"); err != nil {
		return "", err
	}
	if err := form.Close(); err != nil {
		return "", err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, a.remote+path, &body)
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

// HTTPSynthesizer streams 24 kHz mono S16LE PCM from an OpenAI audio server.
type HTTPSynthesizer struct {
	client *http.Client
	remote string
	model  string
	voice  string
}

// NewHTTPSynthesizer configures an OpenAI-compatible speech server.
//
// client, model and voice are required. The server must return raw signed
// 16-bit little-endian mono PCM at SynthesisSampleRate; no resampling occurs.
func NewHTTPSynthesizer(client *http.Client, remote, model, voice string) (*HTTPSynthesizer, error) {
	if err := validateHTTP(client, remote); err != nil {
		return nil, err
	}
	if model == "" || voice == "" {
		return nil, errors.New("speech model and voice are required")
	}
	return &HTTPSynthesizer{client: client, remote: strings.TrimRight(remote, "/"), model: model, voice: voice}, nil
}

// Synthesize streams aligned PCM chunks with independent owned storage.
//
// Empty text produces no request. Iteration can stop early; the response body
// is closed in all cases. Request cancellation terminates the HTTP stream.
func (a *HTTPSynthesizer) Synthesize(ctx context.Context, text string) iter.Seq2[[]byte, error] {
	return func(yield func([]byte, error) bool) {
		if text == "" {
			return
		}
		data, err := json.Marshal(struct {
			Model          string `json:"model"`
			Input          string `json:"input"`
			Voice          string `json:"voice"`
			ResponseFormat string `json:"response_format"`
			Stream         bool   `json:"stream"`
		}{a.model, text, a.voice, "pcm", true})
		if err != nil {
			yield(nil, err)
			return
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, a.remote+"/v1/audio/speech", bytes.NewReader(data))
		if err != nil {
			yield(nil, err)
			return
		}
		req.Header.Set("Content-Type", "application/json")
		resp, err := a.client.Do(req)
		if err != nil {
			yield(nil, fmt.Errorf("synthesize audio: %w", err))
			return
		}
		defer closeResponse(ctx, resp.Body)
		if resp.StatusCode != http.StatusOK {
			yield(nil, statusError(resp))
			return
		}
		contentType, _, err := mime.ParseMediaType(resp.Header.Get("Content-Type"))
		if err != nil || contentType != "audio/pcm" && contentType != "application/octet-stream" {
			yield(nil, fmt.Errorf("speech server returned %q instead of PCM", resp.Header.Get("Content-Type")))
			return
		}
		for pcm, err := range PCMChunks(resp.Body) {
			if !yield(pcm, err) {
				return
			}
		}
	}
}

func validateHTTP(client *http.Client, remote string) error {
	if client == nil {
		return errors.New("speech HTTP client is required")
	}
	u, err := url.Parse(remote)
	if err != nil {
		return fmt.Errorf("parse speech server URL: %w", err)
	}
	if (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" || u.User != nil || u.RawQuery != "" || u.Fragment != "" {
		return errors.New("speech server must be an HTTP base URL without credentials, query or fragment")
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
