// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the complete-utterance transcription session.

package batch

import (
	"bytes"
	"context"
	"errors"
	"io"
	"iter"
	"net"
	"reflect"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
)

func TestSession(t *testing.T) {
	// next reads one event with a new Receive call, which resumes where the
	// previous one stopped.
	next := func(ctx context.Context, s genai.RealtimeSession) (genai.RealtimeEvent, error) {
		n, stop := iter.Pull2(s.Receive(ctx))
		defer stop()
		e, err, _ := n()
		return e, err
	}
	t.Run("utterance", func(t *testing.T) {
		var got []byte
		s := NewSession(16000, 8, func(_ context.Context, pcm []byte) (string, error) {
			got = pcm
			return "hello", nil
		})
		want := genai.RealtimeSessionInfo{InputFormat: genai.AudioFormat{SampleRate: 16000}, InputModalities: []genai.Modality{genai.ModalityAudio}, TurnDetection: genai.RealtimeTurnManual, Capabilities: genai.RealtimeCapabilities{AudioCommit: true}}
		if info := s.Info(); !reflect.DeepEqual(info, want) {
			t.Errorf("info = %+v", info)
		}
		for _, in := range []genai.RealtimeInput{nil, genai.RealtimeInputContent{Audio: []byte{0, 0}, Generate: true}, genai.RealtimeCancel{}} {
			if err := s.Send(t.Context(), in); err == nil {
				t.Errorf("%+v accepted", in)
			}
		}
		if _, ok := errors.AsType[*base.ErrNotSupported](s.Send(t.Context(), genai.RealtimeCancel{})); !ok {
			t.Error("cancel is not ErrNotSupported")
		}
		if err := s.Send(t.Context(), genai.RealtimeInputContent{Audio: []byte{1, 0, 2, 0}}); err != nil {
			t.Fatal(err)
		}
		if err := s.Send(t.Context(), genai.RealtimeInputContent{Audio: make([]byte, 6)}); err == nil {
			t.Error("audio beyond the bound accepted")
		}
		if err := s.Send(t.Context(), genai.RealtimeInputContent{Audio: []byte{3, 0}, Commit: true}); err != nil {
			t.Fatal(err)
		}
		if err := s.Send(t.Context(), genai.RealtimeInputContent{Audio: []byte{0, 0}}); err == nil {
			t.Error("audio after commit accepted")
		}
		for _, w := range []genai.RealtimeEvent{
			{Kind: genai.RealtimeEventTranscriptInput, Content: genai.RealtimeContent{Text: "hello", Final: true}},
			{Kind: genai.RealtimeEventTurnDone},
		} {
			if e, err := next(t.Context(), s); err != nil || !reflect.DeepEqual(e, w) {
				t.Fatalf("event = %+v, %v, want %+v", e, err, w)
			}
		}
		if _, err := next(t.Context(), s); !errors.Is(err, io.EOF) {
			t.Errorf("after the utterance: %v", err)
		}
		if !bytes.Equal(got, []byte{1, 0, 2, 0, 3, 0}) {
			t.Errorf("pcm = %v", got)
		}
	})
	t.Run("empty transcript", func(t *testing.T) {
		s := NewSession(16000, 8, func(context.Context, []byte) (string, error) { return "", nil })
		if err := s.Send(t.Context(), genai.RealtimeInputContent{Commit: true}); err != nil {
			t.Fatal(err)
		}
		if e, err := next(t.Context(), s); err != nil || e.Kind != genai.RealtimeEventTurnDone {
			t.Fatalf("event = %+v, %v", e, err)
		}
	})
	t.Run("failure", func(t *testing.T) {
		calls := 0
		want := errors.New("boom")
		s := NewSession(16000, 8, func(context.Context, []byte) (string, error) {
			calls++
			return "", want
		})
		if err := s.Send(t.Context(), genai.RealtimeInputContent{Commit: true}); err != nil {
			t.Fatal(err)
		}
		for range 2 {
			if _, err := next(t.Context(), s); !errors.Is(err, want) {
				t.Fatal(err)
			}
		}
		if calls != 1 {
			t.Errorf("transcribed %d times", calls)
		}
	})
	t.Run("ends", func(t *testing.T) {
		s := NewSession(16000, 8, func(context.Context, []byte) (string, error) { return "", nil })
		ctx, cancel := context.WithCancel(t.Context())
		cancel()
		if _, err := next(ctx, s); !errors.Is(err, context.Canceled) {
			t.Errorf("canceled: %v", err)
		}
		if err := s.Close(); err != nil {
			t.Fatal(err)
		}
		if _, err := next(t.Context(), s); !errors.Is(err, net.ErrClosed) {
			t.Errorf("closed: %v", err)
		}
		if err := s.Send(t.Context(), genai.RealtimeInputContent{Commit: true}); !errors.Is(err, net.ErrClosed) {
			t.Errorf("send after close: %v", err)
		}
	})
}
