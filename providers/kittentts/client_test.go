// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for KittenTTS synthesis requests and streamed responses.

package kittentts

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestClient(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req request
		if r.URL.Path != "/synthesize" || json.NewDecoder(r.Body).Decode(&req) != nil {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/octet-stream")
		switch req.Text {
		case "voice":
			_, _ = w.Write([]byte(req.Voice[:2]))
		case "error":
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte(`{"error":"synthesis failed"}`))
		case "json":
			w.Header().Set("Content-Type", "application/json")
		case "empty":
		case "cancel":
			_, _ = w.Write([]byte{1, 2})
			w.(http.Flusher).Flush() //nolint:forcetypeassert // httptest writers flush.
			<-r.Context().Done()
		default:
			_, _ = w.Write([]byte{1, 2})
			w.(http.Flusher).Flush() //nolint:forcetypeassert // httptest writers flush.
			_, _ = w.Write([]byte{3, 4})
		}
	}))
	t.Cleanup(srv.Close)
	c, err := New(srv.Client(), srv.URL+"/", "")
	if err != nil {
		t.Fatal(err)
	}
	t.Run("new", func(t *testing.T) {
		t.Parallel()
		for _, remote := range []string{"", "ftp://x", "http://u@x", "http://x?q", ":"} {
			if _, err := New(srv.Client(), remote, ""); err == nil {
				t.Errorf("%q accepted", remote)
			}
		}
		if _, err := New(nil, srv.URL, ""); err == nil {
			t.Error("nil client accepted")
		}
	})
	t.Run("audio", func(t *testing.T) {
		t.Parallel()
		if pcm, err := collect(t.Context(), c, "hello"); err != nil || !bytes.Equal(pcm, []byte{1, 2, 3, 4}) {
			t.Errorf("pcm = %v, %v", pcm, err)
		}
		for _, text := range []string{"", "empty"} {
			if pcm, err := collect(t.Context(), c, text); err != nil || len(pcm) != 0 {
				t.Errorf("%q: pcm = %v, %v", text, pcm, err)
			}
		}
	})
	t.Run("voice", func(t *testing.T) {
		t.Parallel()
		if pcm, err := collect(t.Context(), c, "voice"); err != nil || string(pcm) != DefaultVoice[:2] {
			t.Errorf("default voice = %q, %v", pcm, err)
		}
		luna, err := New(srv.Client(), srv.URL, "Luna")
		if err != nil {
			t.Fatal(err)
		}
		if pcm, err := collect(t.Context(), luna, "voice"); err != nil || string(pcm) != "Lu" {
			t.Errorf("configured voice = %q, %v", pcm, err)
		}
	})
	t.Run("failures", func(t *testing.T) {
		t.Parallel()
		if _, err := collect(t.Context(), c, "error"); err == nil || !strings.Contains(err.Error(), "synthesis failed") {
			t.Errorf("status error = %v", err)
		}
		if _, err := collect(t.Context(), c, "json"); err == nil || !strings.Contains(err.Error(), "instead of PCM") {
			t.Errorf("content type error = %v", err)
		}
		down, err := New(srv.Client(), "http://127.0.0.1:1", "")
		if err != nil {
			t.Fatal(err)
		}
		if _, err := collect(t.Context(), down, "hello"); err == nil {
			t.Error("unreachable worker accepted")
		}
	})
	t.Run("cancel", func(t *testing.T) {
		t.Parallel()
		ctx, cancel := context.WithCancel(t.Context())
		defer cancel()
		var got error
		for pcm, err := range c.Synthesize(ctx, "cancel") {
			if len(pcm) > 0 {
				cancel()
			}
			if err != nil {
				got = err
			}
		}
		if !errors.Is(got, context.Canceled) {
			t.Errorf("error = %v", got)
		}
	})
}

func collect(ctx context.Context, c *Client, text string) ([]byte, error) {
	var out []byte
	for pcm, err := range c.Synthesize(ctx, text) {
		if err != nil {
			return nil, err
		}
		out = append(out, pcm...)
	}
	return out, nil
}
