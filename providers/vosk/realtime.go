// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Vosk WebSocket transcription as a genai.RealtimeSession.

package vosk

import (
	"context"
	"errors"
	"fmt"
	"io"
	"iter"
	"net"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/coder/websocket"
	"github.com/coder/websocket/wsjson"
	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
)

// eofMessage ends the utterance. alphacep/vosk-server compares the message
// text exactly (websocket/asr_server.py).
const eofMessage = `{"eof" : 1}`

// result is a Vosk recognizer result: PartialResult, or Result at an endpoint
// and FinalResult at eofMessage. A server configured for alternatives sends
// them, best first, instead of text.
type result struct {
	Partial      *string       `json:"partial"`
	Text         *string       `json:"text"`
	Alternatives []alternative `json:"alternatives"`
}

type alternative struct {
	Text string `json:"text"`
}

// session transcribes one utterance over the vosk-server WebSocket protocol.
// The server replies to each audio message and to eofMessage with exactly one
// result, in order, so the reply to eofMessage is the final one.
//
// Vosk emits a result whenever its endpointer detects the end of a phrase,
// then partials of the next phrase. Phrase results become input transcript
// deltas and the current partial an interim transcript, each prefixed with
// the space that joins it to the text before.
type session struct {
	conn *websocket.Conn
	info genai.RealtimeSessionInfo
	// last is the 1-based index of the reply to eofMessage, zero before
	// Commit. Send stores it before writing eofMessage, so Receive reads it
	// without waiting behind a blocked write.
	last atomic.Int64

	// Receiver-owned, so a consumer that stops early and receives again loses
	// nothing: text is the completed phrases, partial the current phrase's
	// draft, replies the results read, pending the untaken events, and end
	// the session's end after TurnDone.
	text, partial string
	replies       int64
	pending       []genai.RealtimeEvent
	end           error

	mu   sync.Mutex
	sent int64
}

// dialSession opens a session for format PCM.
func dialSession(ctx context.Context, client *http.Client, remote string, format genai.AudioFormat) (*session, error) {
	conn, _, err := websocket.Dial(ctx, remote, &websocket.DialOptions{HTTPClient: client}) //nolint:bodyclose // coder/websocket owns the response body.
	if err != nil {
		return nil, fmt.Errorf("open vosk transcription: %w", err)
	}
	// A result carries at most an utterance's transcript.
	conn.SetReadLimit(1 << 20)
	// The configuration precedes audio and has no reply.
	config := map[string]map[string]int{"config": {"sample_rate": format.SampleRate}}
	if err := wsjson.Write(ctx, conn, config); err != nil {
		_ = conn.CloseNow()
		return nil, fmt.Errorf("configure vosk transcription: %w", err)
	}
	info := genai.RealtimeSessionInfo{
		InputFormat:     format,
		InputModalities: []genai.Modality{genai.ModalityAudio},
		TurnDetection:   genai.RealtimeTurnManual,
		Capabilities:    genai.RealtimeCapabilities{AudioCommit: true},
	}
	return &session{conn: conn, info: info}, nil
}

// Info implements genai.RealtimeSession.
func (s *session) Info() genai.RealtimeSessionInfo { return s.info }

// Close implements genai.RealtimeSession.
func (s *session) Close() error {
	if err := s.conn.CloseNow(); err != nil && !errors.Is(err, net.ErrClosed) {
		return err
	}
	return nil
}

// Send implements genai.RealtimeSession. It accepts audio until Commit. The
// server decodes each message before reading the next, so Send waits for
// recognition to catch up.
func (s *session) Send(ctx context.Context, in genai.RealtimeInput) error {
	if in == nil {
		return errors.New("realtime input is nil")
	}
	if err := in.Validate(); err != nil {
		return err
	}
	v, ok := in.(genai.RealtimeInputContent)
	if !ok || v.Audio == nil && !v.Commit || v.Generate {
		return fmt.Errorf("vosk transcription accepts only audio and commits: %w", &base.ErrNotSupported{Options: []string{fmt.Sprintf("%T", in)}})
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.last.Load() != 0 {
		return errors.New("vosk transcription already committed its utterance")
	}
	if len(v.Audio) != 0 {
		if len(v.Audio)%2 != 0 {
			return errors.New("vosk audio must be whole S16LE samples")
		}
		s.sent++
		if err := s.conn.Write(ctx, websocket.MessageBinary, v.Audio); err != nil {
			return fmt.Errorf("send vosk audio: %w", err)
		}
	}
	if v.Commit {
		s.sent++
		s.last.Store(s.sent)
		if err := s.conn.Write(ctx, websocket.MessageText, []byte(eofMessage)); err != nil {
			return fmt.Errorf("commit vosk audio: %w", err)
		}
	}
	return nil
}

// Receive implements genai.RealtimeSession. The final transcript and TurnDone
// follow the reply to Commit, then io.EOF.
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
			var r result
			if err := wsjson.Read(ctx, s.conn, &r); err != nil {
				yield(genai.RealtimeEvent{}, fmt.Errorf("read vosk transcription: %w", err))
				return
			}
			s.replies++
			switch {
			case r.Partial != nil:
				if p := strings.TrimSpace(*r.Partial); p != s.partial {
					s.partial = p
					s.pending = append(s.pending, genai.RealtimeEvent{Kind: genai.RealtimeEventTranscriptInputInterim, Content: genai.RealtimeContent{Text: s.join(p)}})
				}
			case r.Text != nil || len(r.Alternatives) != 0:
				var t string
				if r.Text != nil {
					t = *r.Text
				} else {
					t = r.Alternatives[0].Text
				}
				if d := s.join(strings.TrimSpace(t)); d != "" {
					s.pending = append(s.pending, genai.RealtimeEvent{Kind: genai.RealtimeEventTranscriptInput, Content: genai.RealtimeContent{Text: d}})
					s.text += d
				}
				s.partial = ""
			default:
				yield(genai.RealtimeEvent{}, errors.New("vosk sent a result without text"))
				return
			}
			if s.replies == s.last.Load() {
				if s.text != "" {
					s.pending = append(s.pending, genai.RealtimeEvent{Kind: genai.RealtimeEventTranscriptInput, Content: genai.RealtimeContent{Text: s.text, Final: true}})
				}
				s.pending = append(s.pending, genai.RealtimeEvent{Kind: genai.RealtimeEventTurnDone})
				s.end = io.EOF
			}
		}
	}
}

// join returns phrase as it extends the completed text: with a separating
// space unless either is empty. Vosk separates words with spaces in every
// language.
func (s *session) join(phrase string) string {
	if s.text == "" || phrase == "" {
		return phrase
	}
	return " " + phrase
}

var _ genai.RealtimeSession = (*session)(nil)
