// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for Kyutai client configuration, session options and scoreboard.

package kyutai

import (
	"errors"
	"net/http"
	"slices"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
)

func TestNew(t *testing.T) {
	t.Parallel()
	t.Run("error", func(t *testing.T) {
		t.Parallel()
		for name, tc := range map[string]struct {
			client *http.Client
			remote string
			model  Model
		}{
			"no client":   {nil, "http://127.0.0.1:1", STT1B},
			"websocket":   {http.DefaultClient, "ws://127.0.0.1:1", STT1B},
			"credentials": {http.DefaultClient, "http://u:p@127.0.0.1:1", STT1B},
			"model":       {http.DefaultClient, "http://127.0.0.1:1", "kyutai/stt-1b-en_fr-mlx"},
		} {
			if _, err := New(tc.client, tc.remote, tc.model, ""); err == nil {
				t.Errorf("%s: accepted", name)
			}
		}
	})
}

func TestClient(t *testing.T) {
	t.Parallel()
	t.Run("ConnectRealtime", func(t *testing.T) {
		t.Parallel()
		// Each rejection happens before dialing the unreachable server.
		for name, tc := range map[string]struct {
			model Model
			opts  []genai.RealtimeOption
			want  string
		}{
			"conversation": {STT1B, nil, "conversation"},
			"rate":         {STT1B, []genai.RealtimeOption{genai.RealtimeOptionTranscription{}, &genai.RealtimeOptionAudio{InputFormat: genai.AudioFormat{SampleRate: 16000}}}, "input sample rate 16000 Hz"},
			"turn":         {STT1B, []genai.RealtimeOption{genai.RealtimeOptionTranscription{}, genai.RealtimeTurnServer}, "server turn detection"},
			"language":     {STT2B6, []genai.RealtimeOption{genai.RealtimeOptionTranscription{}, &genai.RealtimeOptionAudio{InputLanguage: "fr-CA"}}, "input language fr-CA"},
		} {
			c, err := New(http.DefaultClient, "http://127.0.0.1:1", tc.model, "")
			if err != nil {
				t.Fatal(err)
			}
			_, err = c.ConnectRealtime(t.Context(), tc.opts...)
			if e, ok := errors.AsType[*base.ErrNotSupported](err); !ok || !slices.Contains(e.Options, tc.want) {
				t.Errorf("%s: %v", name, err)
			}
		}
	})
	t.Run("Scoreboard", func(t *testing.T) {
		t.Parallel()
		c, err := New(http.DefaultClient, "http://127.0.0.1:1", STT1B, "")
		if err != nil {
			t.Fatal(err)
		}
		score := c.Scoreboard()
		if err := score.Validate(); err != nil {
			t.Fatal(err)
		}
		if r := score.Realtime[0]; r.Realtime.InputSampleRate != 24000 || !slices.Contains(r.Models, c.ModelID()) {
			t.Errorf("realtime row = %+v", r)
		}
	})
}
