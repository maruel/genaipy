// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for NeMo-Speech.cpp transcription requests, responses and language support.

package nemospeech

import (
	"errors"
	"io"
	"maps"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"

	"encoding/binary"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"golang.org/x/text/language"
)

func TestClient(t *testing.T) {
	t.Parallel()
	t.Run("language", func(t *testing.T) {
		t.Parallel()
		for _, tc := range []struct {
			model      Model
			lang, want string
		}{
			{ParakeetTDT, "fr", ""},
			{ParakeetCTC, "en-US", ""},
			{NemotronASR, "und", ""},
			{NemotronASR, "fr-CA", "fr-CA"},
			{NemotronASR, "en-AU", "en-GB"},
			{NemotronASR, "no", "nb-NO"},
		} {
			a, err := New(http.DefaultClient, "http://127.0.0.1:1", tc.model)
			if err != nil {
				t.Fatal(err)
			}
			if got, err := a.lang(language.MustParse(tc.lang)); err != nil || got != tc.want {
				t.Errorf("%s %s = %q, %v; want %q", tc.model, tc.lang, got, err, tc.want)
			}
		}
		for _, tc := range []struct {
			model Model
			lang  string
		}{
			{ParakeetTDT, "th"},
			{ParakeetCTC, "fr"},
			{NemotronASREnglish, "de"},
			// Adaptation-ready locales need fine-tuning.
			{NemotronASR, "el"},
			// Traditional Chinese matches zh-CN below language.High.
			{NemotronASR, "zh-TW"},
		} {
			a, err := New(http.DefaultClient, "http://127.0.0.1:1", tc.model)
			if err != nil {
				t.Fatal(err)
			}
			_, err = a.ConnectRealtime(t.Context(), genai.RealtimeOptionTranscription{}, &genai.RealtimeOptionAudio{InputLanguage: tc.lang})
			if !errors.As(err, new(*base.ErrNotSupported)) {
				t.Errorf("%s %s: error = %v", tc.model, tc.lang, err)
			}
		}
	})
	t.Run("Parakeet TDT", func(t *testing.T) {
		t.Parallel()
		t.Run("request", func(t *testing.T) {
			t.Parallel()
			var got map[string]string
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/v1/audio/transcriptions" || r.Method != http.MethodPost {
					t.Errorf("request = %s %s", r.Method, r.URL.Path)
				}
				got = readForm(t, w, r)
				_, _ = io.WriteString(w, `{"text":"  turn left  "}`)
			}))
			t.Cleanup(srv.Close)
			text, err := transcribeTDT(t, srv, []byte{1, 0, 2, 0}, &genai.RealtimeOptionAudio{InputFormat: genai.AudioFormat{SampleRate: 22050}, InputLanguage: "fr"})
			if err != nil || text != "turn left" {
				t.Fatalf("transcribe = %q, %v", text, err)
			}
			if wav := got["file"]; len(wav) < 44 || wav[:4] != "RIFF" || binary.LittleEndian.Uint32([]byte(wav)[24:]) != 22050 {
				t.Errorf("WAV = %q", wav)
			}
			delete(got, "file")
			if !maps.Equal(got, map[string]string{"response_format": "json"}) {
				t.Errorf("fields = %v", got)
			}
		})
		t.Run("empty", func(t *testing.T) {
			t.Parallel()
			srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { t.Error("request for empty audio") }))
			t.Cleanup(srv.Close)
			if text, err := transcribeTDT(t, srv, nil); err != nil || text != "" {
				t.Fatalf("empty = %q, %v", text, err)
			}
		})
		for _, tc := range []struct {
			name   string
			status int
			body   string
			want   string
		}{
			{"malformed", 200, "not JSON", "decode"},
			{"HTTP error", 500, "failed", "HTTP 500"},
			{"missing text", 200, `{}`, "no text"},
			{"in-band error", 200, `{"error":{"message":"model failed"}}`, "model failed"},
			{"oversized", 200, `{"text":"` + strings.Repeat("x", 1<<20) + `"}`, "exceeds"},
		} {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()
				srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
					w.WriteHeader(tc.status)
					_, _ = io.WriteString(w, tc.body)
				}))
				t.Cleanup(srv.Close)
				if _, err := transcribeTDT(t, srv, []byte{0, 0}); err == nil || !strings.Contains(err.Error(), tc.want) {
					t.Fatalf("error = %v, want %s", err, tc.want)
				}
			})
		}
	})
}

// transcribeTDT transcribes pcm in a Parakeet TDT session against srv.
func transcribeTDT(t *testing.T, srv *httptest.Server, pcm []byte, opts ...genai.RealtimeOption) (string, error) {
	a, err := New(srv.Client(), srv.URL, ParakeetTDT)
	if err != nil {
		t.Fatal(err)
	}
	s := connect(t, a, opts...)
	send(t, s, genai.RealtimeInputContent{Audio: pcm, Commit: true})
	text := ""
	for e, err := range s.Receive(t.Context()) {
		if err != nil {
			return text, err
		}
		if e.Content.Final {
			text = e.Content.Text
		}
		if e.Kind == genai.RealtimeEventTurnDone {
			break
		}
	}
	return text, nil
}

// readForm returns the multipart fields of a transcription request.
func readForm(t *testing.T, w http.ResponseWriter, r *http.Request) map[string]string {
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20)
	form, err := r.MultipartReader()
	if err != nil {
		t.Error(err)
		return nil
	}
	fields := map[string]string{}
	for {
		part, err := form.NextPart()
		if errors.Is(err, io.EOF) {
			return fields
		}
		if err != nil {
			t.Error(err)
			return nil
		}
		data, err := io.ReadAll(part)
		if err != nil {
			t.Error(err)
			return nil
		}
		fields[part.FormName()] = string(data)
	}
}

func TestModel(t *testing.T) {
	t.Parallel()
	t.Run("Languages", func(t *testing.T) {
		t.Parallel()
		for _, m := range Models {
			a, err := New(http.DefaultClient, "http://127.0.0.1:1", m)
			if err != nil {
				t.Fatal(err)
			}
			langs := m.Languages()
			if len(langs) == 0 || !slices.IsSorted(langs) {
				t.Errorf("%s languages %q, want a sorted nonempty list", m, langs)
			}
			// Each declared language is one the session accepts.
			for _, l := range langs {
				if _, err := a.lang(language.MustParse(l)); err != nil {
					t.Errorf("%s %s: %v", m, l, err)
				}
			}
		}
		if langs := Model("nemotron-x").Languages(); langs != nil {
			t.Errorf("unknown model languages %q, want nil", langs)
		}
	})
}

func TestNew(t *testing.T) {
	t.Parallel()
	t.Run("error", func(t *testing.T) {
		t.Parallel()
		for _, tc := range []struct {
			name, remote string
			model        Model
			client       *http.Client
		}{
			{"nil client", "http://localhost", ParakeetTDT, nil},
			{"invalid URL", "ftp://localhost", ParakeetTDT, http.DefaultClient},
			{"query", "http://localhost?q=1", ParakeetTDT, http.DefaultClient},
			{"credentials", "http://user:pass@localhost", ParakeetTDT, http.DefaultClient},
			{"no model", "http://localhost", "", http.DefaultClient},
			{"unknown model", "http://localhost", "nemotron-x", http.DefaultClient},
		} {
			t.Run(tc.name, func(t *testing.T) {
				if _, err := New(tc.client, tc.remote, tc.model); err == nil {
					t.Fatal("expected configuration error")
				}
			})
		}
	})
}
