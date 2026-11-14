// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package batch adapts a recognizer of complete utterances to a transcription
// genai.RealtimeSession.
package batch

import (
	"context"
	"errors"
	"fmt"
	"io"
	"iter"
	"net"
	"sync"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
)

// Transcribe recognizes one complete utterance of mono S16LE PCM.
type Transcribe func(ctx context.Context, pcm []byte) (string, error)

// NewSession returns a session that buffers up to maxBytes of PCM at
// sampleRate and calls transcribe on Commit. Receive then yields the final
// transcript, unless it is empty, and TurnDone, then io.EOF. The session
// transcribes one utterance and never yields drafts.
func NewSession(sampleRate, maxBytes int, transcribe Transcribe) genai.RealtimeSession {
	return &session{
		info: genai.RealtimeSessionInfo{
			InputFormat:     genai.AudioFormat{SampleRate: sampleRate},
			InputModalities: []genai.Modality{genai.ModalityAudio},
			TurnDetection:   genai.RealtimeTurnManual,
			Capabilities:    genai.RealtimeCapabilities{AudioCommit: true},
		},
		maxBytes:   maxBytes,
		transcribe: transcribe,
		commit:     make(chan struct{}),
		closed:     make(chan struct{}),
	}
}

type session struct {
	info       genai.RealtimeSessionInfo
	maxBytes   int
	transcribe Transcribe
	commit     chan struct{}
	closed     chan struct{}
	closeOnce  sync.Once

	mu        sync.Mutex
	pcm       []byte
	committed bool

	// Receiver-owned: the untaken events and the utterance's end, io.EOF or
	// the transcription error, so a consumer that stops early and receives
	// again loses nothing.
	pending []genai.RealtimeEvent
	end     error
}

// Info implements genai.RealtimeSession.
func (s *session) Info() genai.RealtimeSessionInfo { return s.info }

// Close implements genai.RealtimeSession.
func (s *session) Close() error {
	s.closeOnce.Do(func() { close(s.closed) })
	return nil
}

// Send implements genai.RealtimeSession. It accepts audio until Commit.
func (s *session) Send(_ context.Context, in genai.RealtimeInput) error {
	if in == nil {
		return errors.New("realtime input is nil")
	}
	if err := in.Validate(); err != nil {
		return err
	}
	v, ok := in.(genai.RealtimeInputContent)
	if !ok || v.Audio == nil && !v.Commit || v.Generate {
		return fmt.Errorf("transcription accepts only audio and commits: %w", &base.ErrNotSupported{Options: []string{fmt.Sprintf("%T", in)}})
	}
	select {
	case <-s.closed:
		return net.ErrClosed
	default:
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.committed {
		return errors.New("transcription already committed its utterance")
	}
	if len(s.pcm)+len(v.Audio) > s.maxBytes {
		return fmt.Errorf("utterance exceeds %d bytes", s.maxBytes)
	}
	s.pcm = append(s.pcm, v.Audio...)
	if v.Commit {
		s.committed = true
		close(s.commit)
	}
	return nil
}

// Receive implements genai.RealtimeSession. Transcription runs in the first
// Receive after Commit, under its ctx.
func (s *session) Receive(ctx context.Context) iter.Seq2[genai.RealtimeEvent, error] {
	return func(yield func(genai.RealtimeEvent, error) bool) {
		for {
			if len(s.pending) != 0 {
				e := s.pending[0]
				s.pending[0] = genai.RealtimeEvent{}
				s.pending = s.pending[1:]
				if !yield(e, nil) {
					return
				}
				continue
			}
			if s.end != nil {
				yield(genai.RealtimeEvent{}, s.end)
				return
			}
			select {
			case <-ctx.Done():
				yield(genai.RealtimeEvent{}, ctx.Err())
				return
			case <-s.closed:
				yield(genai.RealtimeEvent{}, net.ErrClosed)
				return
			case <-s.commit:
			}
			s.mu.Lock()
			pcm := s.pcm
			s.pcm = nil
			s.mu.Unlock()
			text, err := s.transcribe(ctx, pcm)
			if err != nil {
				s.end = err
				continue
			}
			if text != "" {
				s.pending = append(s.pending, genai.RealtimeEvent{Kind: genai.RealtimeEventTranscriptInput, Content: genai.RealtimeContent{Text: text, Final: true}})
			}
			s.pending = append(s.pending, genai.RealtimeEvent{Kind: genai.RealtimeEventTurnDone})
			s.end = io.EOF
		}
	}
}

var _ genai.RealtimeSession = (*session)(nil)
