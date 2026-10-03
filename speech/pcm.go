// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package speech provides local speech recognition and synthesis adapters.
package speech

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"iter"
	"math"
)

// Recognizer transcribes mono S16LE PCM captured at the supplied sample rate.
type Recognizer interface {
	Transcribe(context.Context, []byte, int) (string, error)
}

// Synthesizer streams mono S16LE PCM at SynthesisSampleRate.
type Synthesizer interface {
	Synthesize(context.Context, string) iter.Seq2[[]byte, error]
}

// SynthesisSampleRate is the output sample rate of the synthesis adapters.
//
// Output is signed 16-bit little-endian mono PCM at 24 kHz. HTTP speech
// servers must be configured to produce this format; no resampling occurs.
const SynthesisSampleRate = 24000

// PCMChunks frames signed 16-bit PCM into aligned chunks with owned storage.
//
// An odd trailing byte or a read failure yields an error and ends iteration.
// The caller owns r and must close it when applicable, including on early exit.
func PCMChunks(r io.Reader) iter.Seq2[[]byte, error] {
	return func(yield func([]byte, error) bool) {
		var pending byte
		hasPending := false
		buf := make([]byte, 32*1024)
		for {
			n, err := r.Read(buf)
			if n > 0 {
				pcm := make([]byte, n+1)
				start := 0
				if hasPending {
					pcm[0] = pending
					start = 1
				}
				copy(pcm[start:], buf[:n])
				length := start + n
				even := length - length%2
				hasPending = length != even
				if hasPending {
					pending = pcm[even]
				}
				if even > 0 && !yield(pcm[:even], nil) {
					return
				}
			}
			if err != nil {
				if errors.Is(err, io.EOF) {
					if hasPending {
						yield(nil, errors.New("PCM stream returned odd-length audio"))
					}
				} else {
					yield(nil, fmt.Errorf("read PCM stream: %w", err))
				}
				return
			}
		}
	}
}

func pcmWAV(pcm []byte, sampleRate int) ([]byte, error) {
	if sampleRate <= 0 || uint64(sampleRate) > math.MaxUint32/2 {
		return nil, errors.New("sample rate must be positive and fit a WAV byte rate")
	}
	if len(pcm)%2 != 0 {
		return nil, errors.New("PCM input has odd length")
	}
	if uint64(len(pcm)) > math.MaxUint32-36 {
		return nil, errors.New("PCM input exceeds WAV size limit")
	}
	out := make([]byte, 44+len(pcm))
	copy(out, "RIFF")
	binary.LittleEndian.PutUint32(out[4:], uint32(36+len(pcm)))
	copy(out[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(out[16:], 16)
	binary.LittleEndian.PutUint16(out[20:], 1)
	binary.LittleEndian.PutUint16(out[22:], 1)
	binary.LittleEndian.PutUint32(out[24:], uint32(sampleRate))
	binary.LittleEndian.PutUint32(out[28:], uint32(sampleRate*2))
	binary.LittleEndian.PutUint16(out[32:], 2)
	binary.LittleEndian.PutUint16(out[34:], 16)
	copy(out[36:], "data")
	binary.LittleEndian.PutUint32(out[40:], uint32(len(pcm)))
	copy(out[44:], pcm)
	return out, nil
}
