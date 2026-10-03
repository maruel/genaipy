// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for genai speech recognition requests and transcript metadata.

package speech

import (
	"context"
	"encoding/binary"
	"errors"
	"net/http"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/scoreboard"
)

func TestProviderRecognizer(t *testing.T) {
	t.Parallel()
	t.Run("Transcribe", func(t *testing.T) {
		t.Run("input validation", testProviderRecognizerInputs)
		t.Run("error", testProviderRecognizerFailure)
		t.Run("WAV request", func(t *testing.T) {
			t.Parallel()
			p := &fakeASRProvider{reply: "hello world"}
			text, err := mustProviderRecognizer(t, p).Transcribe(t.Context(), []byte{1, 0, 2, 0}, 44100)
			if err != nil {
				t.Fatal(err)
			}
			if text != "hello world" {
				t.Errorf("text = %q, want hello world", text)
			}
			if p.mimeType != "audio/wav" {
				t.Errorf("mimeType = %q, want audio/wav", p.mimeType)
			}
			if string(p.wav[:4]) != "RIFF" || string(p.wav[8:12]) != "WAVE" {
				t.Fatalf("wav header = %q/%q, want RIFF/WAVE", p.wav[:4], p.wav[8:12])
			}
			if got := binary.LittleEndian.Uint32(p.wav[24:]); got != 44100 {
				t.Errorf("wav sample rate = %d, want %d", got, 44100)
			}
			if got := binary.LittleEndian.Uint32(p.wav[40:]); got != 4 {
				t.Errorf("wav data size = %d, want 4", got)
			}
		})
		t.Run("transcript", func(t *testing.T) {
			t.Parallel()
			// Raw outputs follow the formats parse_asr_output accepts in
			// QwenLM/Qwen3-ASR qwen_asr/inference/utils.py.
			for _, tc := range []struct{ name, reply, want string }{
				{"Qwen3-ASR tag", "language English<asr_text>Set a timer.", "Set a timer."},
				{"Qwen3-ASR metadata lines", "language English\n\n<asr_text> Set a timer. \n", "Set a timer."},
				{"Qwen3-ASR no speech", "language None<asr_text>", ""},
				{"plain text", " Set a timer.\n", "Set a timer."},
			} {
				t.Run(tc.name, func(t *testing.T) {
					t.Parallel()
					text, err := mustProviderRecognizer(t, &fakeASRProvider{reply: tc.reply}).Transcribe(t.Context(), []byte{1, 0}, 16000)
					if err != nil {
						t.Fatal(err)
					}
					if text != tc.want {
						t.Errorf("text = %q, want %q", text, tc.want)
					}
				})
			}
		})
	})
}

type fakeASRProvider struct {
	base.NotImplemented

	reply    string
	mimeType string
	wav      []byte
}

func (p *fakeASRProvider) Close() error { return nil }

func (p *fakeASRProvider) Name() string { return "fake-asr" }

func (p *fakeASRProvider) ModelID() string { return "fake-model" }

func (p *fakeASRProvider) OutputModalities() genai.Modalities {
	return genai.Modalities{scoreboard.ModalityText}
}

func (p *fakeASRProvider) Scoreboard() scoreboard.Score { return scoreboard.Score{} }

func (p *fakeASRProvider) HTTPClient() *http.Client { return nil }

func (p *fakeASRProvider) GenSync(_ context.Context, msgs genai.Messages, _ ...genai.GenOption) (genai.Result, error) {
	doc := msgs[0].Requests[1].Doc
	mimeType, data, err := doc.Read(10 * 1024 * 1024)
	if err != nil {
		return genai.Result{}, err
	}
	p.mimeType = mimeType
	p.wav = data
	return genai.Result{Replies: []genai.Reply{{Text: p.reply}}}, nil
}

func mustProviderRecognizer(t *testing.T, p genai.Provider) *ProviderRecognizer {
	a, err := NewProviderRecognizer(p)
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func TestNewProviderRecognizer(t *testing.T) {
	t.Parallel()
	if _, err := NewProviderRecognizer(nil); err == nil {
		t.Fatal("nil provider accepted")
	}
}

func testProviderRecognizerFailure(t *testing.T) {
	t.Parallel()
	a := mustProviderRecognizer(t, &failingProvider{})
	if _, err := a.Transcribe(t.Context(), []byte{0, 0}, 16000); !errors.Is(err, context.Canceled) {
		t.Fatalf("error = %v", err)
	}
}

type failingProvider struct{ fakeASRProvider }

func (p *failingProvider) GenSync(context.Context, genai.Messages, ...genai.GenOption) (genai.Result, error) {
	return genai.Result{}, context.Canceled
}

func testProviderRecognizerInputs(t *testing.T) {
	t.Parallel()
	p := &fakeASRProvider{reply: "must not run"}
	a := mustProviderRecognizer(t, p)
	for _, tc := range []struct {
		pcm  []byte
		rate int
	}{{[]byte{0}, 16000}, {[]byte{0, 0}, 0}, {[]byte{0, 0}, -1}} {
		if _, err := a.Transcribe(t.Context(), tc.pcm, tc.rate); err == nil {
			t.Fatal("invalid PCM/rate accepted")
		}
	}
	if text, err := a.Transcribe(t.Context(), nil, 16000); err != nil || text != "" {
		t.Fatalf("empty = %q, %v", text, err)
	}
	if p.wav != nil {
		t.Fatal("input validation called provider")
	}
}
