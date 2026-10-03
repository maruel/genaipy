// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for aligned PCM streaming, owned storage and stream errors.

package speech

import (
	"bytes"
	"errors"
	"io"
	"reflect"
	"testing"
)

func TestPCMChunks(t *testing.T) {
	t.Parallel()
	t.Run("valid", func(t *testing.T) {
		t.Run("fragmented samples", func(t *testing.T) {
			r := &fragmentReader{parts: [][]byte{{1}, {0, 2}, {0}, {3, 0}}}
			var chunks [][]byte
			for pcm, err := range PCMChunks(r) {
				if err != nil {
					t.Fatal(err)
				}
				chunks = append(chunks, pcm)
			}
			if !reflect.DeepEqual(chunks, [][]byte{{1, 0}, {2, 0}, {3, 0}}) {
				t.Fatalf("chunks = %v", chunks)
			}
		})
		t.Run("early stop", func(t *testing.T) {
			r := &fragmentReader{parts: [][]byte{{1, 0}, {2, 0}}}
			for range PCMChunks(r) {
				break
			}
			if len(r.parts) != 1 {
				t.Fatalf("read %d extra parts", 1-len(r.parts))
			}
		})
		t.Run("empty", func(t *testing.T) {
			for pcm, err := range PCMChunks(bytes.NewReader(nil)) {
				t.Fatalf("empty stream yielded %v, %v", pcm, err)
			}
		})
	})
	t.Run("error", func(t *testing.T) {
		t.Run("odd trailing byte", func(t *testing.T) {
			var got error
			var pcm []byte
			for chunk, err := range PCMChunks(bytes.NewReader([]byte{1, 0, 2})) {
				pcm = append(pcm, chunk...)
				got = err
			}
			if got == nil || !bytes.Equal(pcm, []byte{1, 0}) {
				t.Fatalf("stream = %v, %v", pcm, got)
			}
		})
		t.Run("read failure with data", func(t *testing.T) {
			sentinel := errors.New("reader failed")
			var got error
			var pcm []byte
			for chunk, err := range PCMChunks(&fragmentReader{parts: [][]byte{{1, 0}}, err: sentinel}) {
				pcm = append(pcm, chunk...)
				got = err
			}
			if !errors.Is(got, sentinel) || !bytes.Equal(pcm, []byte{1, 0}) {
				t.Fatalf("stream = %v, %v", pcm, got)
			}
		})
	})
}

type fragmentReader struct {
	parts [][]byte
	err   error
}

func (r *fragmentReader) Read(p []byte) (int, error) {
	if len(r.parts) == 0 {
		return 0, io.EOF
	}
	n := copy(p, r.parts[0])
	r.parts = r.parts[1:]
	if len(r.parts) == 0 {
		return n, r.err
	}
	return n, nil
}
