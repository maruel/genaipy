// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for Kyutai transcription sessions against a scripted moshi-server.

package kyutai

import (
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"

	"github.com/coder/websocket"
	"github.com/maruel/genai"
	"github.com/vmihailenco/msgpack/v5"
)

// fakeServer accepts /api/asr-streaming connections and hands each to the
// test, which plays moshi-server.
type fakeServer struct {
	conns   chan *websocket.Conn
	headers chan http.Header
}

func startFake(t *testing.T) (*fakeServer, string) {
	f := &fakeServer{conns: make(chan *websocket.Conn, 1), headers: make(chan http.Header, 1)}
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/asr-streaming" {
			http.NotFound(w, r)
			return
		}
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			t.Error(err)
			return
		}
		// The flush of the 2.6B model is 33 frames of float32.
		conn.SetReadLimit(1 << 20)
		f.headers <- r.Header
		f.conns <- conn
		<-release
		_ = conn.CloseNow()
	}))
	t.Cleanup(srv.Close)
	t.Cleanup(func() { close(release) })
	return f, srv.URL
}

type clientMsg struct {
	Type string    `msgpack:"type"`
	PCM  []float32 `msgpack:"pcm"`
	ID   *int64    `msgpack:"id"`
}

func read(t *testing.T, conn *websocket.Conn) clientMsg {
	typ, data, err := conn.Read(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	var m clientMsg
	if err := msgpack.Unmarshal(data, &m); typ != websocket.MessageBinary || err != nil {
		t.Fatalf("message %v: %v", typ, err)
	}
	return m
}

func write(t *testing.T, conn *websocket.Conn, m map[string]any) {
	data, err := msgpack.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	if err := conn.Write(t.Context(), websocket.MessageBinary, data); err != nil {
		t.Fatal(err)
	}
}

// silent reports whether m is Audio of n samples of silence.
func silent(m clientMsg, n int) bool {
	return m.Type == "Audio" && len(m.PCM) == n && !slices.ContainsFunc(m.PCM, func(v float32) bool { return v != 0 })
}

func connect(t *testing.T, remote string, model Model, opts ...genai.RealtimeOption) genai.RealtimeSession {
	c, err := New(http.DefaultClient, remote, model, "key")
	if err != nil {
		t.Fatal(err)
	}
	rt, err := c.ConnectRealtime(t.Context(), append([]genai.RealtimeOption{genai.RealtimeOptionTranscription{}}, opts...)...)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := rt.Close(); err != nil {
			t.Error(err)
		}
	})
	return rt
}

// next returns the next n events.
func next(t *testing.T, rt genai.RealtimeSession, n int) []genai.RealtimeEvent {
	var out []genai.RealtimeEvent
	for e, err := range rt.Receive(t.Context()) {
		if err != nil {
			t.Fatal(err)
		}
		if out = append(out, e); len(out) == n {
			break
		}
	}
	return out
}

func TestSession(t *testing.T) {
	t.Parallel()
	t.Run("transcript", func(t *testing.T) {
		t.Parallel()
		f, remote := startFake(t)
		rt := connect(t, remote, STT2B6, &genai.RealtimeOptionAudio{InputLanguage: "en-US"})
		if h := <-f.headers; h.Get("Kyutai-Api-Key") != "key" {
			t.Errorf("headers = %v", h)
		}
		conn := <-f.conns
		// The stream starts with 1 s of silence, rounded up to a frame.
		if m := read(t, conn); !silent(m, 13*frameSize) {
			t.Errorf("start = %s of %d samples", m.Type, len(m.PCM))
		}
		if err := rt.Send(t.Context(), genai.RealtimeInputContent{Audio: []byte{0x00, 0x40, 0x00, 0x80}}); err != nil {
			t.Fatal(err)
		}
		if m := read(t, conn); m.Type != "Audio" || !slices.Equal(m.PCM, []float32{0.5, -1}) {
			t.Errorf("audio = %+v", m)
		}
		write(t, conn, map[string]any{"type": "Word", "text": "hello", "start_time": 0.0})
		write(t, conn, map[string]any{"type": "Step", "step_idx": 1, "prs": []float32{0.1}, "buffered_pcm": 0})
		write(t, conn, map[string]any{"type": "Word", "text": "world", "start_time": 0.5})
		events := next(t, rt, 2)
		if events[0].Content.Text != "hello" || events[1].Content.Text != " world" || events[1].Content.Final {
			t.Errorf("drafts = %+v", events)
		}
		if err := rt.Send(t.Context(), genai.RealtimeInputContent{Commit: true}); err != nil {
			t.Fatal(err)
		}
		// Silence ends the speech so its last word comes before the Marker.
		if m := read(t, conn); !silent(m, 15*frameSize) {
			t.Errorf("end = %s of %d samples", m.Type, len(m.PCM))
		}
		if m := read(t, conn); m.Type != "Marker" || m.ID == nil {
			t.Errorf("commit = %+v", m)
		}
		// The 2.5 s delay rounds up to 32 frames, plus one for a partial frame.
		if m := read(t, conn); !silent(m, 33*frameSize) {
			t.Errorf("flush = %s of %d samples", m.Type, len(m.PCM))
		}
		if err := rt.Send(t.Context(), genai.RealtimeInputContent{Audio: []byte{0, 0}}); err == nil {
			t.Error("audio accepted after commit")
		}
		write(t, conn, map[string]any{"type": "Marker", "id": 0})
		// A consumer that stops at the final transcript still receives TurnDone.
		if e := next(t, rt, 1)[0]; e.Kind != genai.RealtimeEventTranscriptInput || !e.Content.Final || e.Content.Text != "hello world" {
			t.Errorf("final = %+v", e)
		}
		if e := next(t, rt, 1)[0]; e.Kind != genai.RealtimeEventTurnDone {
			t.Errorf("done = %+v", e)
		}
	})
	t.Run("error", func(t *testing.T) {
		t.Parallel()
		f, remote := startFake(t)
		rt := connect(t, remote, STT1B)
		write(t, <-f.conns, map[string]any{"type": "Error", "message": "no free channels"})
		var got error
		for _, err := range rt.Receive(t.Context()) {
			got = err
			break
		}
		if got == nil || !strings.Contains(got.Error(), "no free channels") {
			t.Errorf("err = %v", got)
		}
		if err := rt.Send(t.Context(), genai.RealtimeInputContent{Audio: []byte{0}}); err == nil {
			t.Error("partial sample accepted")
		}
	})
}
