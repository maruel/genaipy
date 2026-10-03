// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for local audio HTTP engine request and response contracts.

package speech

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"
	"time"

	"encoding/binary"
)

func TestHTTPRecognizer(t *testing.T) {
	t.Parallel()
	t.Run("Transcribe", func(t *testing.T) {
		t.Run("limits", testHTTPRecognizerInputAndResponseLimits)
		for _, tc := range []struct {
			engine TranscriptionProtocol
			path   string
		}{
			{OpenAI, "/v1/audio/transcriptions"},
			{WhisperCPP, "/inference"},
		} {
			t.Run(string(tc.engine), func(t *testing.T) {
				t.Parallel()
				srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					if r.URL.Path != tc.path || r.Method != http.MethodPost {
						t.Errorf("request = %s %s", r.Method, r.URL.Path)
					}
					r.Body = http.MaxBytesReader(w, r.Body, 1<<20)
					form, err := r.MultipartReader()
					if err != nil {
						t.Error(err)
						return
					}
					fields := map[string]string{}
					for {
						part, err := form.NextPart()
						if errors.Is(err, io.EOF) {
							break
						}
						if err != nil {
							t.Error(err)
							return
						}
						data, err := io.ReadAll(part)
						if err != nil {
							t.Error(err)
							return
						}
						fields[part.FormName()] = string(data)
					}
					if tc.engine == OpenAI && fields["model"] != "asr-model" {
						t.Errorf("model = %q", fields["model"])
					}
					if fields["response_format"] != "json" {
						t.Errorf("response_format = %q", fields["response_format"])
					}
					if wav := fields["file"]; len(wav) < 44 || wav[:4] != "RIFF" || binary.LittleEndian.Uint32([]byte(wav)[24:]) != 22050 {
						t.Errorf("WAV = %q", wav)
					}
					_, _ = io.WriteString(w, `{"text":"  turn left  "}`)
				}))
				t.Cleanup(srv.Close)
				a := mustHTTPRecognizer(t, srv.Client(), srv.URL, tc.engine)
				got, err := a.Transcribe(t.Context(), []byte{1, 0, 2, 0}, 22050)
				if err != nil || got != "turn left" {
					t.Fatalf("transcribe = %q, %v", got, err)
				}
			})
		}
		t.Run("bad response", func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				_, _ = io.WriteString(w, "not JSON")
			}))
			t.Cleanup(srv.Close)
			a := mustHTTPRecognizer(t, srv.Client(), srv.URL, OpenAI)
			if _, err := a.Transcribe(t.Context(), []byte{0, 0}, 16000); err == nil {
				t.Fatal("expected malformed response error")
			}
		})
		for _, tc := range []struct {
			name   string
			status int
			body   string
			want   string
		}{
			{"HTTP error", 500, "failed", "HTTP 500"},
			{"missing text", 200, `{}`, "no text"},
			{"in-band error", 200, `{"error":{"message":"model failed"}}`, "model failed"},
		} {
			t.Run(tc.name, func(t *testing.T) {
				srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
					w.WriteHeader(tc.status)
					_, _ = io.WriteString(w, tc.body)
				}))
				t.Cleanup(srv.Close)
				a := mustHTTPRecognizer(t, srv.Client(), srv.URL, OpenAI)
				if _, err := a.Transcribe(t.Context(), []byte{0, 0}, 16000); err == nil || !strings.Contains(err.Error(), tc.want) {
					t.Fatalf("error = %v, want %s", err, tc.want)
				}
			})
		}
		t.Run("cancelled", func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			cancel()
			a := mustHTTPRecognizer(t, http.DefaultClient, "http://127.0.0.1:1", OpenAI)
			if _, err := a.Transcribe(ctx, []byte{0, 0}, 16000); !errors.Is(err, context.Canceled) {
				t.Fatalf("error = %v", err)
			}
		})
	})
}

func TestHTTPSynthesizer(t *testing.T) {
	t.Parallel()
	t.Run("Synthesize", func(t *testing.T) {
		t.Run("early stop", testHTTPSynthesizerEarlyStop)
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != "/v1/audio/speech" {
				t.Errorf("path = %q", r.URL.Path)
			}
			var request map[string]any
			if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
				t.Error(err)
			}
			if request["model"] != "tts-model" || request["voice"] != "voice-a" || request["response_format"] != "pcm" || request["stream"] != true {
				t.Errorf("request = %#v", request)
			}
			w.Header().Set("Content-Type", "audio/pcm")
			_, _ = w.Write([]byte{1, 0, 2})
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
			_, _ = w.Write([]byte{0, 3, 0})
		}))
		t.Cleanup(srv.Close)
		a := mustHTTPSynthesizer(t, srv.Client(), srv.URL)
		var pcm []byte
		for chunk, err := range a.Synthesize(t.Context(), "hello") {
			if err != nil {
				t.Fatal(err)
			}
			pcm = append(pcm, chunk...)
		}
		if !slices.Equal(pcm, []byte{1, 0, 2, 0, 3, 0}) {
			t.Fatalf("PCM = %v", pcm)
		}
		for _, tc := range []struct {
			name   string
			status int
			body   string
			want   string
		}{
			{"server error", 500, "failed", "HTTP 500"},
			{"odd PCM", 200, "x", "odd-length"},
			{"wrong content type", 200, "xx", "instead of PCM"},
		} {
			t.Run(tc.name, func(t *testing.T) {
				bad := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
					if tc.name != "wrong content type" {
						w.Header().Set("Content-Type", "audio/pcm")
					}
					w.WriteHeader(tc.status)
					_, _ = io.WriteString(w, tc.body)
				}))
				t.Cleanup(bad.Close)
				a := mustHTTPSynthesizer(t, bad.Client(), bad.URL)
				var got error
				for _, err := range a.Synthesize(t.Context(), "hello") {
					got = err
				}
				if got == nil || !strings.Contains(got.Error(), tc.want) {
					t.Fatalf("error = %v, want %s", got, tc.want)
				}
			})
		}
		t.Run("empty PCM", func(t *testing.T) {
			empty := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "audio/pcm")
				w.WriteHeader(http.StatusOK)
			}))
			t.Cleanup(empty.Close)
			a := mustHTTPSynthesizer(t, empty.Client(), empty.URL)
			for chunk, err := range a.Synthesize(t.Context(), "silent") {
				t.Fatalf("empty synthesis yielded %v, %v", chunk, err)
			}
		})
		t.Run("cancelled", func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			cancel()
			var got error
			for _, err := range a.Synthesize(ctx, "hello") {
				got = err
			}
			if got == nil || !strings.Contains(got.Error(), "canceled") {
				t.Fatalf("error = %v", got)
			}
		})
		t.Run("cancel mid-stream", func(t *testing.T) {
			stream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "audio/pcm")
				_, _ = w.Write([]byte{1, 0})
				w.(http.Flusher).Flush()
				<-r.Context().Done()
			}))
			t.Cleanup(stream.Close)
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			a := mustHTTPSynthesizer(t, stream.Client(), stream.URL)
			var got error
			for chunk, err := range a.Synthesize(ctx, "hello") {
				if len(chunk) > 0 {
					cancel()
				}
				if err != nil {
					got = err
				}
			}
			if !errors.Is(got, context.Canceled) {
				t.Fatalf("error = %v", got)
			}
		})
	})
}

func mustHTTPRecognizer(t *testing.T, client *http.Client, remote string, protocol TranscriptionProtocol) *HTTPRecognizer {
	model := "asr-model"
	if protocol == WhisperCPP {
		model = ""
	}
	a, err := NewHTTPRecognizer(client, remote, model, protocol)
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func mustHTTPSynthesizer(t *testing.T, client *http.Client, remote string) *HTTPSynthesizer {
	a, err := NewHTTPSynthesizer(client, remote, "tts-model", "voice-a")
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func TestNewHTTPRecognizer(t *testing.T) {
	t.Parallel()
	t.Run("error", func(t *testing.T) {
		for _, tc := range []struct {
			name, remote, model string
			protocol            TranscriptionProtocol
			client              *http.Client
		}{
			{"nil client", "http://localhost", "model", OpenAI, nil},
			{"invalid URL", "ftp://localhost", "model", OpenAI, http.DefaultClient},
			{"query", "http://localhost?q=1", "model", OpenAI, http.DefaultClient},
			{"credentials", "http://user:pass@localhost", "model", OpenAI, http.DefaultClient},
			{"no model", "http://localhost", "", OpenAI, http.DefaultClient},
			{"whisper model", "http://localhost", "model", WhisperCPP, http.DefaultClient},
			{"unknown protocol", "http://localhost", "model", "bogus", http.DefaultClient},
		} {
			t.Run(tc.name, func(t *testing.T) {
				if _, err := NewHTTPRecognizer(tc.client, tc.remote, tc.model, tc.protocol); err == nil {
					t.Fatal("expected configuration error")
				}
			})
		}
	})
}

func TestNewHTTPSynthesizer(t *testing.T) {
	t.Parallel()
	t.Run("error", func(t *testing.T) {
		for _, tc := range []struct{ remote, model, voice string }{
			{"file:///speech", "model", "voice"},
			{"http://localhost", "", "voice"},
			{"http://localhost", "model", ""},
		} {
			if _, err := NewHTTPSynthesizer(http.DefaultClient, tc.remote, tc.model, tc.voice); err == nil {
				t.Fatal("expected configuration error")
			}
		}
	})
}

func testHTTPSynthesizerEarlyStop(t *testing.T) {
	t.Parallel()
	ended := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "audio/pcm")
		_, _ = w.Write([]byte{1, 0})
		w.(http.Flusher).Flush()
		<-r.Context().Done()
		close(ended)
	}))
	t.Cleanup(srv.Close)
	a := mustHTTPSynthesizer(t, srv.Client(), srv.URL)
	for pcm, err := range a.Synthesize(t.Context(), "hello") {
		if err != nil || len(pcm) == 0 {
			t.Fatalf("first chunk = %v, %v", pcm, err)
		}
		break
	}
	select {
	case <-ended:
	case <-time.After(3 * time.Second):
		t.Fatal("early iterator stop did not close HTTP stream")
	}
}

func testHTTPRecognizerInputAndResponseLimits(t *testing.T) {
	t.Parallel()
	a := mustHTTPRecognizer(t, http.DefaultClient, "http://127.0.0.1:1", OpenAI)
	if text, err := a.Transcribe(t.Context(), nil, 16000); err != nil || text != "" {
		t.Fatalf("empty = %q, %v", text, err)
	}
	for _, tc := range []struct {
		pcm  []byte
		rate int
	}{{[]byte{0}, 16000}, {[]byte{0, 0}, 0}} {
		if _, err := a.Transcribe(t.Context(), tc.pcm, tc.rate); err == nil {
			t.Fatal("invalid PCM/rate accepted")
		}
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, `{"text":"`+strings.Repeat("x", 1<<20)+`"}`)
	}))
	t.Cleanup(srv.Close)
	a = mustHTTPRecognizer(t, srv.Client(), srv.URL, OpenAI)
	if _, err := a.Transcribe(t.Context(), []byte{0, 0}, 16000); err == nil || !strings.Contains(err.Error(), "exceeds") {
		t.Fatalf("error = %v", err)
	}
}
