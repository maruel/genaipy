// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// NeMo-Speech.cpp realtime transcription as a genai.RealtimeSession.

package nemospeech

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"net"
	"net/http"
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"

	"github.com/coder/websocket"
	"github.com/coder/websocket/wsjson"
	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
)

// realtimeEvent is a server event of NeMo-Speech.cpp's
// /v1/audio/transcriptions/realtime protocol (docs/api.md, and the realtime
// handler in server/http/http_server.cpp, in NVIDIA/NeMo-Speech.cpp v0.3.0).
type realtimeEvent struct {
	Type       string `json:"type"`
	Delta      string `json:"delta"`
	Transcript string `json:"transcript"`
	Error      *struct {
		Message string `json:"message"`
		Code    string `json:"code"`
	} `json:"error"`
}

func (ev *realtimeEvent) err() error {
	return fmt.Errorf("realtime transcription error %s: %s", ev.Error.Code, ev.Error.Message)
}

// session transcribes one utterance over NeMo-Speech.cpp's realtime
// WebSocket. The server decodes each audio frame before reading the next, so
// Send waits for recognition to catch up.
//
// The server emits a final transcript per segment whenever its endpointer
// detects the end of speech, then partials of the next segment. Segment finals
// become input transcript deltas and the current partial an interim
// transcript, each prefixed with the space that joins it to the text before.
type session struct {
	conn *websocket.Conn
	info genai.RealtimeSessionInfo

	// Receiver-owned, so a consumer that stops early and receives again loses
	// nothing: text is the completed segments, partial the current segment's
	// draft, and pending the untaken events of one server message.
	text, partial string
	pending       []genai.RealtimeEvent

	mu        sync.Mutex
	committed bool
}

// dialSession opens a session for format PCM in language code; an empty code
// lets the model choose.
func dialSession(ctx context.Context, client *http.Client, remote string, format genai.AudioFormat, code string) (*session, error) {
	conn, _, err := websocket.Dial(ctx, remote+"/v1/audio/transcriptions/realtime", &websocket.DialOptions{HTTPClient: client}) //nolint:bodyclose // coder/websocket owns the response body.
	if err != nil {
		return nil, fmt.Errorf("open realtime transcription: %w", err)
	}
	// Each event carries at most a segment's transcript.
	conn.SetReadLimit(1 << 20)
	if err := startRealtime(ctx, conn, format.SampleRate, code); err != nil {
		_ = conn.CloseNow()
		return nil, err
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

// Send implements genai.RealtimeSession. It accepts audio until Commit.
func (s *session) Send(ctx context.Context, in genai.RealtimeInput) error {
	if in == nil {
		return errors.New("realtime input is nil")
	}
	if err := in.Validate(); err != nil {
		return err
	}
	v, ok := in.(genai.RealtimeInputContent)
	if !ok || v.Audio == nil && !v.Commit || v.Generate {
		return fmt.Errorf("NeMo-Speech transcription accepts only audio and commits: %w", &base.ErrNotSupported{Options: []string{fmt.Sprintf("%T", in)}})
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.committed {
		return errors.New("NeMo-Speech transcription already committed its utterance")
	}
	if len(v.Audio) != 0 {
		if err := s.conn.Write(ctx, websocket.MessageBinary, v.Audio); err != nil {
			return fmt.Errorf("send realtime audio: %w", err)
		}
	}
	if v.Commit {
		if err := wsjson.Write(ctx, s.conn, map[string]string{"type": "input_audio_buffer.commit"}); err != nil {
			return fmt.Errorf("commit realtime audio: %w", err)
		}
		s.committed = true
	}
	return nil
}

// Receive implements genai.RealtimeSession. The final transcript and TurnDone
// follow the server's acknowledgement of the commit. A server error ends the
// session.
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
			var ev realtimeEvent
			if err := wsjson.Read(ctx, s.conn, &ev); err != nil {
				yield(genai.RealtimeEvent{}, fmt.Errorf("read realtime transcription: %w", err))
				return
			}
			switch ev.Type {
			case "conversation.item.input_audio_transcription.delta":
				// TODO: The server sends the whole partial instead of its
				// extension when the new partial does not extend the previous
				// one (emit in server/http/http_server.cpp, v0.3.0), and the
				// event does not say which it is. Partials are detokenized from
				// an append-only greedy token list, so a revision is not
				// expected for the supported models; none was observed. A
				// replaced partial shows as repeated text until the segment's
				// final. Fix upstream by flagging replacements, then honor it.
				s.partial += ev.Delta
				s.pending = append(s.pending, genai.RealtimeEvent{Kind: genai.RealtimeEventTranscriptInputInterim, Content: genai.RealtimeContent{Text: joinSegments(s.text, s.partial)[len(s.text):]}})
			case "conversation.item.input_audio_transcription.completed":
				joined := joinSegments(s.text, ev.Transcript)
				if d := joined[len(s.text):]; d != "" {
					s.pending = append(s.pending, genai.RealtimeEvent{Kind: genai.RealtimeEventTranscriptInput, Content: genai.RealtimeContent{Text: d}})
				}
				s.text, s.partial = joined, ""
			case "input_audio_buffer.committed":
				if s.text != "" {
					s.pending = append(s.pending, genai.RealtimeEvent{Kind: genai.RealtimeEventTranscriptInput, Content: genai.RealtimeContent{Text: s.text, Final: true}})
				}
				s.pending = append(s.pending, genai.RealtimeEvent{Kind: genai.RealtimeEventTurnDone})
			case "error":
				if ev.Error == nil {
					yield(genai.RealtimeEvent{}, errors.New("realtime transcription error without details"))
				} else {
					yield(genai.RealtimeEvent{}, ev.err())
				}
				return
			default:
			}
		}
	}
}

// startRealtime configures the session before audio, which the server
// requires.
func startRealtime(ctx context.Context, conn *websocket.Conn, sampleRate int, code string) error {
	if err := expectEvent(ctx, conn, "session.created"); err != nil {
		return err
	}
	update := struct {
		Type    string `json:"type"`
		Session struct {
			SampleRate int    `json:"sample_rate"`
			Language   string `json:"language,omitempty"`
		} `json:"session"`
	}{Type: "session.update"}
	update.Session.SampleRate = sampleRate
	update.Session.Language = code
	if err := wsjson.Write(ctx, conn, update); err != nil {
		return fmt.Errorf("configure realtime transcription: %w", err)
	}
	return expectEvent(ctx, conn, "session.updated")
}

func expectEvent(ctx context.Context, conn *websocket.Conn, want string) error {
	var ev realtimeEvent
	if err := wsjson.Read(ctx, conn, &ev); err != nil {
		return fmt.Errorf("read realtime transcription: %w", err)
	}
	if ev.Error != nil {
		return ev.err()
	}
	if ev.Type != want {
		return fmt.Errorf("realtime transcription sent %q instead of %q", ev.Type, want)
	}
	return nil
}

// joinSegments joins transcript segments with a space, except next to a
// script written without spaces.
func joinSegments(segments ...string) string {
	var b strings.Builder
	for _, seg := range segments {
		if seg = strings.TrimSpace(seg); seg == "" {
			continue
		}
		if b.Len() > 0 {
			prev, _ := utf8.DecodeLastRuneInString(b.String())
			next, _ := utf8.DecodeRuneInString(seg)
			if !unspaced(prev) && !unspaced(next) {
				b.WriteByte(' ')
			}
		}
		b.WriteString(seg)
	}
	return b.String()
}

func unspaced(r rune) bool {
	return unicode.In(r, unicode.Han, unicode.Hiragana, unicode.Katakana, unicode.Thai, unicode.Lao, unicode.Khmer, unicode.Myanmar)
}

var _ genai.RealtimeSession = (*session)(nil)
