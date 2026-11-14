// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Aligned framing of the worker's streamed PCM.

package kittentts

import (
	"errors"
	"fmt"
	"io"
	"iter"
)

// SampleRate is the output sample rate: Synthesize yields signed 16-bit
// little-endian mono PCM at 24 kHz.
const SampleRate = 24000

// pcmChunks frames signed 16-bit PCM into aligned chunks with owned storage.
//
// An odd trailing byte or a read failure yields an error and ends iteration.
// The caller owns r and must close it when applicable, including on early exit.
func pcmChunks(r io.Reader) iter.Seq2[[]byte, error] {
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
