// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Spoken-language mappings from a BCP 47 tag to a model's request value.

package nemospeech

import (
	"fmt"
	"slices"

	"github.com/maruel/genai/base"
	"golang.org/x/text/language"
)

// languageMapper resolves a spoken language to the value a request carries.
//
// language.Und resolves to "" for every model: the model chooses. A supported
// language resolves to "" when the model has no language input and detects
// the language itself.
type languageMapper func(lang language.Tag) (string, error)

// detectedLanguages accepts the listed ISO 639 base languages and sends
// nothing: the model detects the language from the audio.
func detectedLanguages(bases ...string) languageMapper {
	return func(lang language.Tag) (string, error) {
		if lang == language.Und {
			return "", nil
		}
		if b, _ := lang.Base(); !slices.Contains(bases, b.String()) {
			return "", unsupportedLanguage(lang)
		}
		return "", nil
	}
}

// localeLanguages sends the closest listed locale. A match below
// language.High fails, such as a different script or an unrelated language.
func localeLanguages(locales ...string) languageMapper {
	tags := make([]language.Tag, len(locales))
	for i, l := range locales {
		tags[i] = language.MustParse(l)
	}
	m := language.NewMatcher(tags)
	return func(lang language.Tag) (string, error) {
		if lang == language.Und {
			return "", nil
		}
		if _, i, c := m.Match(lang); c >= language.High {
			return locales[i], nil
		}
		return "", unsupportedLanguage(lang)
	}
}

// unsupportedLanguage reports a spoken language the model cannot transcribe.
func unsupportedLanguage(lang language.Tag) error {
	return fmt.Errorf("NeMo-Speech model does not support spoken language %q: %w", lang, &base.ErrNotSupported{Options: []string{"input language " + lang.String()}})
}
