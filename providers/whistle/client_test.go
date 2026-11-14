// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for Whistle transcription sessions, options and ASR windowing.

package whistle

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
)

func TestClient(t *testing.T) {
	// The fake worker returns the sample count of each window.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		data, _ := io.ReadAll(r.Body)
		switch {
		case r.URL.Path != "/transcribe":
			http.NotFound(w, r)
		case len(data) == 2 && data[1] == 0x80:
			<-r.Context().Done()
		case len(data) == 4:
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte(`{"error":"native decoding failed"}`))
		case len(data) == 6:
			_, _ = w.Write([]byte(`not json`))
		default:
			_, _ = w.Write([]byte(`{"text":" ` + strconv.Itoa(len(data)/2) + ` "}`))
		}
	}))
	t.Cleanup(srv.Close)
	w, err := New(srv.Client(), srv.URL+"/", "")
	if err != nil {
		t.Fatal(err)
	}
	t.Run("new", func(t *testing.T) {
		if w.ModelID() != DefaultModel {
			t.Errorf("ModelID = %q", w.ModelID())
		}
		if c, err := New(srv.Client(), srv.URL, "/m.cact"); err != nil || c.ModelID() != "/m.cact" {
			t.Errorf("ModelID = %v, %v", c, err)
		}
		for _, remote := range []string{"", "ftp://x", "http://u@x", "http://x?q", ":"} {
			if _, err := New(srv.Client(), remote, ""); err == nil {
				t.Errorf("%q accepted", remote)
			}
		}
		if _, err := New(nil, srv.URL, ""); err == nil {
			t.Error("nil client accepted")
		}
	})
	t.Run("long utterance", func(t *testing.T) {
		text, err := transcribe(t.Context(), t, w, make([]byte, 16000*2*31))
		if err != nil || text != "480000 16000" {
			t.Fatalf("got %q, %v", text, err)
		}
	})
	t.Run("failures", func(t *testing.T) {
		for _, n := range []int{4, 6} {
			if _, err := transcribe(t.Context(), t, w, make([]byte, n)); err == nil {
				t.Errorf("%d bytes: no error", n)
			}
		}
	})
	t.Run("options", func(t *testing.T) {
		if _, err := w.ConnectRealtime(t.Context(), genai.RealtimeOptionTranscription{}, &genai.RealtimeOptionAudio{InputLanguage: "de-CH"}); err != nil {
			t.Fatal(err)
		}
		for _, opts := range [][]genai.RealtimeOption{
			{},
			{genai.RealtimeOptionTranscription{}, &genai.RealtimeOptionAudio{InputLanguage: "ja"}},
			{genai.RealtimeOptionTranscription{}, &genai.RealtimeOptionAudio{InputFormat: genai.AudioFormat{SampleRate: 24000}, Voice: "v", OutputTranscription: true}},
			{genai.RealtimeOptionTranscription{}, genai.RealtimeTurnServer},
			{genai.RealtimeOptionTranscription{}, genai.RealtimeOptionInstructions("x")},
		} {
			if _, err := w.ConnectRealtime(t.Context(), opts...); !errors.As(err, new(*base.ErrNotSupported)) {
				t.Errorf("%+v: %v", opts, err)
			}
		}
		if _, err := w.ConnectRealtime(t.Context(), nil); err == nil {
			t.Error("nil option accepted")
		}
	})
	t.Run("cancellation and reuse", func(t *testing.T) {
		c, done := context.WithTimeout(t.Context(), 50*time.Millisecond)
		defer done()
		if _, err := transcribe(c, t, w, []byte{0, 0x80}); !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("got %v", err)
		}
		text, err := transcribe(t.Context(), t, w, []byte{0, 0})
		if err != nil || text != "1" {
			t.Fatalf("got %q, %v", text, err)
		}
	})
}

// transcribe sends pcm in one session and returns the final transcript.
func transcribe(ctx context.Context, t *testing.T, w *Client, pcm []byte) (string, error) {
	s, err := w.ConnectRealtime(ctx, genai.RealtimeOptionTranscription{})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = s.Close() }()
	if err := s.Send(ctx, genai.RealtimeInputContent{Audio: pcm, Commit: true}); err != nil {
		t.Fatal(err)
	}
	text := ""
	for e, err := range s.Receive(ctx) {
		if err != nil {
			return "", err
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
