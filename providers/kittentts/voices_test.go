// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the voice tables' ordering, uniqueness and model lookup errors.

package kittentts

import "testing"

func TestVoices(t *testing.T) {
	t.Parallel()
	for _, model := range Models {
		voices, err := Voices(model)
		if err != nil {
			t.Fatal(err)
		}
		if len(voices) == 0 {
			t.Errorf("%s has no voices", model)
		}
		// Callers take the first voice of a language as its default, and
		// select a voice, which speaks one language, by name.
		for i := 1; i < len(voices); i++ {
			if voices[i-1].Name >= voices[i].Name {
				t.Errorf("%s: %v is not before %v", model, voices[i-1], voices[i])
			}
		}
		for _, v := range voices {
			if len(v.Language) != 2 {
				t.Errorf("%s: %v has no ISO 639-1 language", model, v)
			}
		}
	}
	if _, err := Voices("KittenML/kitten-tts-mini-0.8"); err == nil {
		t.Error("unsupported model accepted")
	}
}
