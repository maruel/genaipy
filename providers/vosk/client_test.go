// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for Vosk transcription sessions against a fake vosk-server.

package vosk

import (
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/coder/websocket"
	"github.com/coder/websocket/wsjson"
	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
)

// fakeServer follows alphacep/vosk-server: it reads the configuration, then
// answers each audio message with the next reply and eofMessage with final.
// A reply of "" closes the connection with an internal error instead.
type fakeServer struct {
	replies []string
	final   string

	config chan map[string]map[string]int
	audio  chan int
}

func (f *fakeServer) start(t *testing.T) string {
	f.config = make(chan map[string]map[string]int, 1)
	f.audio = make(chan int, 100)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			t.Error(err)
			return
		}
		defer func() { _ = conn.CloseNow() }()
		ctx := r.Context()
		var config map[string]map[string]int
		if wsjson.Read(ctx, conn, &config) != nil {
			return
		}
		f.config <- config
		for i := 0; ; i++ {
			typ, data, err := conn.Read(ctx)
			if err != nil {
				return
			}
			reply := f.final
			if typ == websocket.MessageBinary {
				f.audio <- len(data)
				reply = f.replies[i]
			} else if string(data) != eofMessage {
				t.Errorf("text message %q", data)
				return
			}
			if reply == "" {
				_ = conn.Close(websocket.StatusInternalError, "native failure")
				return
			}
			if conn.Write(ctx, websocket.MessageText, []byte(reply)) != nil || typ == websocket.MessageText {
				return
			}
		}
	}))
	t.Cleanup(srv.Close)
	return srv.URL
}

func newClient(t *testing.T, remote string) *Client {
	c, err := New(http.DefaultClient, remote, English)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

// run sends each chunk then commits, and returns the events until the
// session's end, which must be io.EOF.
func run(t *testing.T, s genai.RealtimeSession, chunks ...[]byte) []genai.RealtimeEvent {
	t.Cleanup(func() { _ = s.Close() })
	go func() {
		for _, c := range chunks {
			if err := s.Send(t.Context(), genai.RealtimeInputContent{Audio: c}); err != nil {
				t.Error(err)
				return
			}
		}
		if err := s.Send(t.Context(), genai.RealtimeInputContent{Commit: true}); err != nil {
			t.Error(err)
		}
	}()
	var got []genai.RealtimeEvent
	for e, err := range s.Receive(t.Context()) {
		if err != nil {
			if !errors.Is(err, io.EOF) {
				t.Fatal(err)
			}
			break
		}
		got = append(got, e)
	}
	return got
}

func interim(s string) genai.RealtimeEvent {
	return genai.RealtimeEvent{Kind: genai.RealtimeEventTranscriptInputInterim, Content: genai.RealtimeContent{Text: s}}
}

func input(s string, final bool) genai.RealtimeEvent {
	return genai.RealtimeEvent{Kind: genai.RealtimeEventTranscriptInput, Content: genai.RealtimeContent{Text: s, Final: final}}
}

func TestNew(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		for _, remote := range []string{"http://127.0.0.1:2700", "ws://vosk:2700/asr"} {
			c, err := New(http.DefaultClient, remote, English)
			if err != nil {
				t.Fatal(err)
			}
			if c.ModelID() != string(English) {
				t.Errorf("ModelID = %q", c.ModelID())
			}
		}
	})
	t.Run("error", func(t *testing.T) {
		for _, tc := range []struct {
			name   string
			client *http.Client
			remote string
			model  Model
		}{
			{"nil client", nil, "http://127.0.0.1", English},
			{"scheme", http.DefaultClient, "ftp://127.0.0.1", English},
			{"credentials", http.DefaultClient, "http://u:p@127.0.0.1", English},
			{"query", http.DefaultClient, "http://127.0.0.1/?a=1", English},
			{"model", http.DefaultClient, "http://127.0.0.1", "vosk-model-fr-0.22"},
		} {
			t.Run(tc.name, func(t *testing.T) {
				if _, err := New(tc.client, tc.remote, tc.model); err == nil {
					t.Fatal("accepted")
				}
			})
		}
	})
}

func TestModels(t *testing.T) {
	if !slices.IsSorted(Models) || len(Models) == 0 {
		t.Fatalf("Models = %q", Models)
	}
	for _, m := range Models {
		if len(m.Languages()) != 1 {
			t.Errorf("%s languages = %q, want one", m, m.Languages())
		}
	}
}

func TestClient(t *testing.T) {
	t.Run("ConnectRealtime", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			t.Run("phrases", func(t *testing.T) {
				f := &fakeServer{
					replies: []string{
						`{"partial" : ""}`,
						`{"partial" : "turn on"}`,
						`{"partial" : "turn on"}`,
						`{"text" : "turn on the light"}`,
						`{"partial" : "in the"}`,
					},
					final: `{"text" : "in the kitchen"}`,
				}
				c := newClient(t, f.start(t))
				s, err := c.ConnectRealtime(t.Context(), genai.RealtimeOptionTranscription{}, &genai.RealtimeOptionAudio{InputFormat: genai.AudioFormat{SampleRate: 48000}, InputLanguage: "en-CA"})
				if err != nil {
					t.Fatal(err)
				}
				pcm := make([]byte, 640)
				got := run(t, s, pcm, pcm, pcm, pcm, pcm)
				// Unchanged partials publish nothing.
				want := []genai.RealtimeEvent{
					interim("turn on"),
					input("turn on the light", false),
					interim(" in the"),
					input(" in the kitchen", false),
					input("turn on the light in the kitchen", true),
					{Kind: genai.RealtimeEventTurnDone},
				}
				if !reflect.DeepEqual(got, want) {
					t.Errorf("events = %+v\nwant %+v", got, want)
				}
				if config := <-f.config; config["config"]["sample_rate"] != 48000 {
					t.Errorf("config = %v", config)
				}
				for range 5 {
					if n := <-f.audio; n != 640 {
						t.Errorf("audio message = %d bytes", n)
					}
				}
			})
			t.Run("silence", func(t *testing.T) {
				f := &fakeServer{replies: []string{`{"partial" : ""}`}, final: `{"text" : ""}`}
				s, err := newClient(t, f.start(t)).ConnectRealtime(t.Context(), genai.RealtimeOptionTranscription{})
				if err != nil {
					t.Fatal(err)
				}
				got := run(t, s, make([]byte, 640))
				if want := []genai.RealtimeEvent{{Kind: genai.RealtimeEventTurnDone}}; !reflect.DeepEqual(got, want) {
					t.Errorf("events = %+v", got)
				}
				if config := <-f.config; config["config"]["sample_rate"] != 16000 {
					t.Errorf("config = %v", config)
				}
			})
			t.Run("alternatives", func(t *testing.T) {
				f := &fakeServer{final: `{"alternatives" : [{"text" : "lights on", "confidence" : 210.5}, {"text" : "light son", "confidence" : 200.1}]}`}
				s, err := newClient(t, f.start(t)).ConnectRealtime(t.Context(), genai.RealtimeOptionTranscription{})
				if err != nil {
					t.Fatal(err)
				}
				got := run(t, s)
				want := []genai.RealtimeEvent{input("lights on", false), input("lights on", true), {Kind: genai.RealtimeEventTurnDone}}
				if !reflect.DeepEqual(got, want) {
					t.Errorf("events = %+v", got)
				}
			})
		})
		t.Run("error", func(t *testing.T) {
			c := newClient(t, "http://127.0.0.1:1")
			for _, tc := range []struct {
				name string
				opts []genai.RealtimeOption
				want string
			}{
				{"conversation", nil, "conversation"},
				{"language", []genai.RealtimeOption{genai.RealtimeOptionTranscription{}, &genai.RealtimeOptionAudio{InputLanguage: "fr"}}, "input language fr"},
				{"upsampling", []genai.RealtimeOption{genai.RealtimeOptionTranscription{}, &genai.RealtimeOptionAudio{InputFormat: genai.AudioFormat{SampleRate: 8000}}}, "input sample rate 8000 Hz"},
				{"server turn detection", []genai.RealtimeOption{genai.RealtimeOptionTranscription{}, genai.RealtimeTurnServer}, "server turn detection"},
			} {
				t.Run(tc.name, func(t *testing.T) {
					_, err := c.ConnectRealtime(t.Context(), tc.opts...)
					e, ok := errors.AsType[*base.ErrNotSupported](err)
					if !ok || !slices.Contains(e.Options, tc.want) {
						t.Fatalf("error = %v, want unsupported %q", err, tc.want)
					}
				})
			}
			t.Run("server failure", func(t *testing.T) {
				f := &fakeServer{replies: []string{""}}
				s, err := newClient(t, f.start(t)).ConnectRealtime(t.Context(), genai.RealtimeOptionTranscription{})
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() { _ = s.Close() })
				if err := s.Send(t.Context(), genai.RealtimeInputContent{Audio: make([]byte, 640)}); err != nil {
					t.Fatal(err)
				}
				for _, err := range s.Receive(t.Context()) {
					if err == nil || !strings.Contains(err.Error(), "native failure") {
						t.Fatalf("error = %v", err)
					}
					break
				}
			})
			t.Run("odd audio", func(t *testing.T) {
				f := &fakeServer{}
				s, err := newClient(t, f.start(t)).ConnectRealtime(t.Context(), genai.RealtimeOptionTranscription{})
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() { _ = s.Close() })
				if err := s.Send(t.Context(), genai.RealtimeInputContent{Audio: make([]byte, 3)}); err == nil {
					t.Fatal("accepted a partial sample")
				}
			})
			t.Run("send after commit", func(t *testing.T) {
				f := &fakeServer{final: `{"text" : ""}`}
				s, err := newClient(t, f.start(t)).ConnectRealtime(t.Context(), genai.RealtimeOptionTranscription{})
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() { _ = s.Close() })
				if err := s.Send(t.Context(), genai.RealtimeInputContent{Commit: true}); err != nil {
					t.Fatal(err)
				}
				if err := s.Send(t.Context(), genai.RealtimeInputContent{Audio: make([]byte, 2)}); err == nil {
					t.Fatal("accepted audio after commit")
				}
			})
		})
	})
}
