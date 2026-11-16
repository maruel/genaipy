// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// moshi-server's /api/asr-streaming protocol as a genai.RealtimeSession.

package kyutai

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"iter"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/coder/websocket"
	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/vmihailenco/msgpack/v5"
)

// frameSize is the samples of one 80 ms model step.
const frameSize = 1920

// startSilence starts every stream, as upstream's
// stt_from_file_rust_server.py does for both models. The 2.6B model's
// stt_config asks for 1 s; the 1B model's asks for none, yet without it the
// 1B model dropped a first word spoken in the first frame: "I need a taxi at
// six." became "need a taxi at six." One frame of silence restored "I"; 12
// frames also restored the capitalization of a first "Please" and "Okay,".
const startSilence = time.Second

// endSilence ends the speech before the commit's Marker, so that the
// padding token that ends the last word comes before the Marker.
//
// moshi-core's asr.rs emits a word only at the padding token after it, and
// moshi-server's batched_asr.rs returns a Marker asr_delay_in_tokens steps
// after it, counting the frames still buffered before it; its configs set 6
// steps for the 1B model and 32 for the 2.6B model. Its step counter also
// advances for other connections on the step that receives the Marker. The
// Marker thus follows the end of speech by the silence's frames plus 5 or 31
// steps. No upstream source bounds when the last word ends: tokens follow
// their audio by the model delay, but a word's tokens run on after that.
// Measured from the end of English and French clips: the 1B model ended its
// last word up to 11 steps later, 18 when the clip was cut inside the word,
// and the 2.6B model up to 34 steps later when cut inside a word. 1.2 s,
// 15 steps, leaves them at least 20 and 46 steps.
const endSilence = 1200 * time.Millisecond

// audioMsg is the Audio message of asr.rs: SampleRate PCM in [-1, 1].
type audioMsg struct {
	Type string    `msgpack:"type"`
	PCM  []float32 `msgpack:"pcm"`
}

// markerMsg is the Marker message of asr.rs: the server returns it once the
// audio before it is transcribed.
type markerMsg struct {
	Type string `msgpack:"type"`
	ID   int64  `msgpack:"id"`
}

// outMsg is a server message of asr.rs. The client reads Word, Marker and
// Error; it ignores EndWord, Step and Ready.
type outMsg struct {
	Type    string `msgpack:"type"`
	Text    string `msgpack:"text"`
	Message string `msgpack:"message"`
}

// session transcribes one utterance.
//
// Each Word becomes an input transcript delta, prefixed with the space that
// joins it to the words before. Commit sends endSilence, a Marker,
// then the model delay of silence plus one frame for a partial frame the
// server buffered: the server returns the Marker once the delay has passed
// over the audio before it, and steps the silence as fast as the model runs.
// The returned Marker yields the final transcript and TurnDone.
type session struct {
	conn  *websocket.Conn
	info  genai.RealtimeSessionInfo
	flush int

	// Receiver-owned, so a consumer that stops early and receives again loses
	// nothing: text is the transcript so far and pending the untaken events
	// of one server message.
	text    string
	pending []genai.RealtimeEvent

	mu        sync.Mutex
	committed bool
}

func dialSession(ctx context.Context, client *http.Client, remote, apiKey string, m modelInfo) (*session, error) {
	h := http.Header{}
	if apiKey != "" {
		h.Set("Kyutai-Api-Key", apiKey)
	}
	conn, _, err := websocket.Dial(ctx, remote+"/api/asr-streaming", &websocket.DialOptions{HTTPClient: client, HTTPHeader: h}) //nolint:bodyclose // coder/websocket owns the response body.
	if err != nil {
		return nil, fmt.Errorf("open Kyutai transcription: %w", err)
	}
	// Server messages carry one word, VAD probabilities or an error.
	conn.SetReadLimit(1 << 20)
	s := &session{
		conn:  conn,
		flush: (steps(m.delay) + 1) * frameSize,
		info: genai.RealtimeSessionInfo{
			InputFormat:     genai.AudioFormat{SampleRate: SampleRate},
			InputModalities: []genai.Modality{genai.ModalityAudio},
			TurnDetection:   genai.RealtimeTurnManual,
			Capabilities:    genai.RealtimeCapabilities{AudioCommit: true},
		},
	}
	if err := s.write(ctx, &audioMsg{Type: "Audio", PCM: make([]float32, steps(startSilence)*frameSize)}); err != nil {
		_ = conn.CloseNow()
		return nil, err
	}
	return s, nil
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

// Send implements genai.RealtimeSession. It accepts S16LE audio until Commit.
func (s *session) Send(ctx context.Context, in genai.RealtimeInput) error {
	if in == nil {
		return errors.New("realtime input is nil")
	}
	if err := in.Validate(); err != nil {
		return err
	}
	v, ok := in.(genai.RealtimeInputContent)
	if !ok || v.Audio == nil && !v.Commit || v.Generate {
		return fmt.Errorf("kyutai transcription accepts only audio and commits: %w", &base.ErrNotSupported{Options: []string{fmt.Sprintf("%T", in)}})
	}
	if len(v.Audio)%2 != 0 {
		return errors.New("kyutai transcription audio must be whole S16LE samples")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.committed {
		return errors.New("kyutai transcription already committed its utterance")
	}
	if len(v.Audio) != 0 {
		pcm := make([]float32, len(v.Audio)/2)
		for i := range pcm {
			pcm[i] = float32(int16(binary.LittleEndian.Uint16(v.Audio[2*i:]))) / 32768
		}
		if err := s.write(ctx, &audioMsg{Type: "Audio", PCM: pcm}); err != nil {
			return err
		}
	}
	if v.Commit {
		if err := s.write(ctx, &audioMsg{Type: "Audio", PCM: make([]float32, steps(endSilence)*frameSize)}); err != nil {
			return err
		}
		if err := s.write(ctx, &markerMsg{Type: "Marker"}); err != nil {
			return err
		}
		if err := s.write(ctx, &audioMsg{Type: "Audio", PCM: make([]float32, s.flush)}); err != nil {
			return err
		}
		s.committed = true
	}
	return nil
}

// Receive implements genai.RealtimeSession. The final transcript and TurnDone
// follow the server's return of the commit's Marker. A server error ends the
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
			typ, data, err := s.conn.Read(ctx)
			if err != nil {
				yield(genai.RealtimeEvent{}, fmt.Errorf("read Kyutai transcription: %w", err))
				return
			}
			if typ != websocket.MessageBinary {
				continue
			}
			var m outMsg
			if err := msgpack.Unmarshal(data, &m); err != nil {
				yield(genai.RealtimeEvent{}, fmt.Errorf("decode Kyutai message: %w", err))
				return
			}
			switch m.Type {
			case "Word":
				w := strings.TrimSpace(m.Text)
				if w == "" {
					continue
				}
				if s.text != "" {
					w = " " + w
				}
				s.text += w
				s.pending = append(s.pending, genai.RealtimeEvent{Kind: genai.RealtimeEventTranscriptInput, Content: genai.RealtimeContent{Text: w}})
			case "Marker":
				if s.text != "" {
					s.pending = append(s.pending, genai.RealtimeEvent{Kind: genai.RealtimeEventTranscriptInput, Content: genai.RealtimeContent{Text: s.text, Final: true}})
				}
				s.pending = append(s.pending, genai.RealtimeEvent{Kind: genai.RealtimeEventTurnDone})
			case "Error":
				yield(genai.RealtimeEvent{}, fmt.Errorf("kyutai transcription error: %s", m.Message))
				return
			default:
			}
		}
	}
}

// write sends m, an audioMsg or a markerMsg.
func (s *session) write(ctx context.Context, m any) error {
	data, err := msgpack.Marshal(m)
	if err != nil {
		return err
	}
	if err := s.conn.Write(ctx, websocket.MessageBinary, data); err != nil {
		return fmt.Errorf("send Kyutai message: %w", err)
	}
	return nil
}

// steps rounds d up to whole model steps.
func steps(d time.Duration) int {
	step := time.Duration(frameSize) * time.Second / SampleRate
	return int((d + step - 1) / step)
}

var _ genai.RealtimeSession = (*session)(nil)
