// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for NeMo-Speech.cpp realtime transcription sessions: setup, segments and errors.

package nemospeech

import (
	"context"
	"encoding/json"
	"errors"
	"iter"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/coder/websocket"
	"github.com/coder/websocket/wsjson"
	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
)

// realtimeServer fakes NeMo-Speech.cpp's realtime endpoint. It records the
// session.update, then answers each binary frame with the next events of
// script and the commit with the events of commit.
type realtimeServer struct {
	created json.RawMessage
	script  [][]string
	commit  []string

	update chan map[string]any
	frames chan int
}

func (f *realtimeServer) start(t *testing.T) string {
	f.update = make(chan map[string]any, 1)
	f.frames = make(chan int, 100)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/audio/transcriptions/realtime" {
			t.Errorf("path = %s", r.URL.Path)
			http.NotFound(w, r)
			return
		}
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			t.Error(err)
			return
		}
		defer func() { _ = conn.CloseNow() }()
		f.serve(r.Context(), conn)
	}))
	t.Cleanup(srv.Close)
	return srv.URL
}

func (f *realtimeServer) serve(ctx context.Context, conn *websocket.Conn) {
	created := f.created
	if created == nil {
		created = json.RawMessage(`{"type":"session.created","session":{"sample_rate":16000}}`)
	}
	if conn.Write(ctx, websocket.MessageText, created) != nil || !strings.Contains(string(created), "session.created") {
		return
	}
	var update map[string]any
	if wsjson.Read(ctx, conn, &update) != nil {
		return
	}
	f.update <- update
	if conn.Write(ctx, websocket.MessageText, []byte(`{"type":"session.updated"}`)) != nil {
		return
	}
	for i := 0; ; {
		typ, data, err := conn.Read(ctx)
		if err != nil {
			return
		}
		events := f.commit
		if typ == websocket.MessageBinary {
			f.frames <- len(data)
			if i >= len(f.script) {
				continue
			}
			events = f.script[i]
			i++
		}
		for _, ev := range events {
			if conn.Write(ctx, websocket.MessageText, []byte(ev)) != nil {
				return
			}
		}
	}
}

func realtimeRecognizer(t *testing.T, remote string, model Model) *Client {
	r, err := New(http.DefaultClient, remote, model)
	if err != nil {
		t.Fatal(err)
	}
	return r
}

func sessionOf(t *testing.T, update map[string]any) map[string]any {
	session, ok := update["session"].(map[string]any)
	if !ok {
		t.Fatalf("update = %v", update)
	}
	return session
}

func delta(s string) string {
	return `{"type":"conversation.item.input_audio_transcription.delta","delta":"` + s + `"}`
}

func completed(s string) string {
	return `{"type":"conversation.item.input_audio_transcription.completed","transcript":"` + s + `"}`
}

const committed = `{"type":"input_audio_buffer.committed"}`

func TestClientConnectRealtime(t *testing.T) {
	t.Parallel()
	t.Run("valid", func(t *testing.T) {
		t.Parallel()
		t.Run("segments", func(t *testing.T) {
			t.Parallel()
			f := &realtimeServer{
				script: [][]string{
					{delta("hello")},
					{delta(" world"), completed("Hello world.")},
					{delta("how")},
				},
				commit: []string{completed("How?"), committed},
			}
			s := connect(t, realtimeRecognizer(t, f.start(t), NemotronASR))
			if update := <-f.update; update["type"] != "session.update" || len(sessionOf(t, update)) != 1 {
				t.Errorf("update = %v", update)
			}
			want := genai.RealtimeSessionInfo{InputFormat: genai.AudioFormat{SampleRate: 16000}, InputModalities: []genai.Modality{genai.ModalityAudio}, TurnDetection: genai.RealtimeTurnManual, Capabilities: genai.RealtimeCapabilities{AudioCommit: true}}
			if info := s.Info(); !reflect.DeepEqual(info, want) {
				t.Errorf("info = %+v", info)
			}
			// Each event is read by a new Receive call, which resumes where the
			// previous one stopped, mid-frame included.
			next := func() (genai.RealtimeEvent, error) {
				n, stop := iter.Pull2(s.Receive(t.Context()))
				defer stop()
				e, err, _ := n()
				return e, err
			}
			// Each frame's events arrive before the next frame.
			for _, events := range [][]genai.RealtimeEvent{
				{interim("hello")},
				{interim("hello world"), transcript("Hello world.")},
				{interim(" how")},
			} {
				send(t, s, genai.RealtimeInputContent{Audio: make([]byte, 320)})
				if n := <-f.frames; n != 320 {
					t.Errorf("frame = %d bytes, want 320", n)
				}
				for _, w := range events {
					if e, err := next(); err != nil || !reflect.DeepEqual(e, w) {
						t.Fatalf("event = %+v, %v, want %+v", e, err, w)
					}
				}
			}
			send(t, s, genai.RealtimeInputContent{Commit: true})
			if err := s.Send(t.Context(), genai.RealtimeInputContent{Audio: make([]byte, 2)}); err == nil {
				t.Error("audio after commit accepted")
			}
			for _, w := range []genai.RealtimeEvent{
				transcript(" How?"),
				{Kind: genai.RealtimeEventTranscriptInput, Content: genai.RealtimeContent{Text: "Hello world. How?", Final: true}},
				{Kind: genai.RealtimeEventTurnDone},
			} {
				if e, err := next(); err != nil || !reflect.DeepEqual(e, w) {
					t.Fatalf("event = %+v, %v, want %+v", e, err, w)
				}
			}
		})
		t.Run("language", func(t *testing.T) {
			t.Parallel()
			f := &realtimeServer{commit: []string{completed(""), committed}}
			s := connect(t, realtimeRecognizer(t, f.start(t), NemotronASR), &genai.RealtimeOptionAudio{InputFormat: genai.AudioFormat{SampleRate: 48000}, InputLanguage: "fr-CA"})
			session := sessionOf(t, <-f.update)
			if session["sample_rate"] != 48000.0 || session["language"] != "fr-CA" {
				t.Errorf("session = %v", session)
			}
			send(t, s, genai.RealtimeInputContent{Commit: true})
			if events := <-collect(t, s); !reflect.DeepEqual(events, []genai.RealtimeEvent{{Kind: genai.RealtimeEventTurnDone}}) {
				t.Errorf("events = %+v", events)
			}
		})
	})
	t.Run("error", func(t *testing.T) {
		t.Parallel()
		t.Run("streaming unsupported", func(t *testing.T) {
			t.Parallel()
			f := &realtimeServer{created: json.RawMessage(`{"type":"error","error":{"message":"complete audio only","code":"streaming_unsupported"}}`)}
			_, err := realtimeRecognizer(t, f.start(t), ParakeetCTC).ConnectRealtime(t.Context(), genai.RealtimeOptionTranscription{})
			if err == nil || !strings.Contains(err.Error(), "streaming_unsupported") {
				t.Fatalf("error = %v", err)
			}
		})
		t.Run("server error", func(t *testing.T) {
			t.Parallel()
			f := &realtimeServer{script: [][]string{{`{"type":"error","error":{"message":"too long","code":"invalid_argument"}}`}}}
			s := connect(t, realtimeRecognizer(t, f.start(t), NemotronASREnglish), &genai.RealtimeOptionAudio{InputLanguage: "en"})
			send(t, s, genai.RealtimeInputContent{Audio: make([]byte, 32)})
			for _, err := range s.Receive(t.Context()) {
				if err == nil || !strings.Contains(err.Error(), "too long") {
					t.Errorf("error = %v", err)
				}
				break
			}
		})
		t.Run("cancelled", func(t *testing.T) {
			t.Parallel()
			f := &realtimeServer{}
			s := connect(t, realtimeRecognizer(t, f.start(t), NemotronASR))
			ctx, cancel := context.WithCancel(t.Context())
			cancel()
			for _, err := range s.Receive(ctx) {
				if !errors.Is(err, context.Canceled) {
					t.Errorf("error = %v", err)
				}
				break
			}
		})
		t.Run("rejected", func(t *testing.T) {
			t.Parallel()
			for _, tc := range []struct {
				name  string
				model Model
				opts  []genai.RealtimeOption
			}{
				{"conversation", NemotronASR, nil},
				{"sample rate", NemotronASR, []genai.RealtimeOption{genai.RealtimeOptionTranscription{}, &genai.RealtimeOptionAudio{InputFormat: genai.AudioFormat{SampleRate: 4000}}}},
				{"instructions", NemotronASR, []genai.RealtimeOption{genai.RealtimeOptionTranscription{}, genai.RealtimeOptionInstructions("x")}},
			} {
				if _, err := realtimeRecognizer(t, "http://127.0.0.1:1", tc.model).ConnectRealtime(t.Context(), tc.opts...); !errors.As(err, new(*base.ErrNotSupported)) {
					t.Errorf("%s: error = %v", tc.name, err)
				}
			}
			_, err := realtimeRecognizer(t, "http://127.0.0.1:1", ParakeetCTC).ConnectRealtime(t.Context(), genai.RealtimeOptionTranscription{}, &genai.RealtimeOptionAudio{InputLanguage: "fr"})
			if !errors.As(err, new(*base.ErrNotSupported)) {
				t.Errorf("unsupported language: error = %v", err)
			}
		})
		t.Run("input", func(t *testing.T) {
			t.Parallel()
			s := connect(t, realtimeRecognizer(t, (&realtimeServer{}).start(t), NemotronASR))
			for _, in := range []genai.RealtimeInput{genai.RealtimeCancel{}, genai.RealtimeInputContent{Text: "x"}, genai.RealtimeInputContent{Audio: make([]byte, 2), Generate: true}} {
				if err := s.Send(t.Context(), in); !errors.As(err, new(*base.ErrNotSupported)) {
					t.Errorf("%+v: error = %v", in, err)
				}
			}
		})
	})
}

func TestJoinSegments(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		segments []string
		want     string
	}{
		{[]string{"Hello world.", " how"}, "Hello world. how"},
		{[]string{"", "Hi.", "  "}, "Hi."},
		{[]string{"今天", "很好"}, "今天很好"}, //nolint:gosmopolitan // Han script joins without spaces.
		{nil, ""},
	} {
		if got := joinSegments(tc.segments...); got != tc.want {
			t.Errorf("joinSegments(%q) = %q, want %q", tc.segments, got, tc.want)
		}
	}
}

// connect opens a transcription session closed at cleanup.
func connect(t *testing.T, c *Client, opts ...genai.RealtimeOption) genai.RealtimeSession {
	s, err := c.ConnectRealtime(t.Context(), append([]genai.RealtimeOption{genai.RealtimeOptionTranscription{}}, opts...)...)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := s.Close(); err != nil {
			t.Error(err)
		}
	})
	return s
}

func send(t *testing.T, s genai.RealtimeSession, in genai.RealtimeInput) {
	if err := s.Send(t.Context(), in); err != nil {
		t.Fatal(err)
	}
}

// collect receives events on another goroutine until TurnDone.
func collect(t *testing.T, s genai.RealtimeSession) <-chan []genai.RealtimeEvent {
	out := make(chan []genai.RealtimeEvent, 1)
	go func() {
		var events []genai.RealtimeEvent
		for e, err := range s.Receive(t.Context()) {
			if err != nil {
				t.Error(err)
				break
			}
			events = append(events, e)
			if e.Kind == genai.RealtimeEventTurnDone {
				break
			}
		}
		out <- events
	}()
	return out
}

func interim(s string) genai.RealtimeEvent {
	return genai.RealtimeEvent{Kind: genai.RealtimeEventTranscriptInputInterim, Content: genai.RealtimeContent{Text: s}}
}

func transcript(s string) genai.RealtimeEvent {
	return genai.RealtimeEvent{Kind: genai.RealtimeEventTranscriptInput, Content: genai.RealtimeContent{Text: s}}
}
