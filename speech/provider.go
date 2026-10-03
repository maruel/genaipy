// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Speech recognition through an existing genai provider.

package speech

import (
	"bytes"
	"context"
	"errors"
	"strings"

	"github.com/maruel/genai"
)

// ProviderRecognizer transcribes mono S16LE PCM through a genai provider.
//
// The caller owns the provider and its lifetime.
type ProviderRecognizer struct{ provider genai.Provider }

// NewProviderRecognizer binds speech recognition to an existing provider.
func NewProviderRecognizer(p genai.Provider) (*ProviderRecognizer, error) {
	if p == nil {
		return nil, errors.New("speech recognition provider is required")
	}
	return &ProviderRecognizer{provider: p}, nil
}

// Transcribe recognizes speech captured at sampleRate Hz.
//
// Empty audio returns empty text. Qwen ASR language metadata is removed from
// tagged output; other models' plain text is preserved without surrounding space.
func (a *ProviderRecognizer) Transcribe(ctx context.Context, pcm []byte, sampleRate int) (string, error) {
	wav, err := pcmWAV(pcm, sampleRate)
	if err != nil {
		return "", err
	}
	if len(pcm) == 0 {
		return "", nil
	}
	msg := genai.Message{Requests: []genai.Request{
		{Text: "Transcribe the attached audio. Return only the transcript text, with no commentary."},
		{Doc: genai.Doc{Filename: "speech.wav", Src: bytes.NewReader(wav)}},
	}}
	res, err := a.provider.GenSync(ctx, genai.Messages{msg}, &genai.GenOptionText{SystemPrompt: "You are a speech recognition engine. Return only the spoken words."})
	if err != nil {
		return "", err
	}
	raw := res.String()
	// QwenLM/Qwen3-ASR's parse_asr_output uses this tag to separate language
	// metadata from the transcript: qwen_asr/inference/utils.py.
	if _, text, ok := strings.Cut(raw, "<asr_text>"); ok {
		raw = text
	}
	return strings.TrimSpace(raw), nil
}
