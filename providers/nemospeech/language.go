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

// nemotronLocales are the transcription-ready and broad-coverage locales in the
// nvidia/nemotron-3.5-asr-streaming-0.6b model card.
var nemotronLocales = []string{
	"en-US", "en-GB", "es-US", "es-ES", "fr-FR", "fr-CA", "it-IT", "pt-BR",
	"pt-PT", "nl-NL", "de-DE", "tr-TR", "ru-RU", "ar-AR", "hi-IN", "ja-JP",
	"ko-KR", "vi-VN", "uk-UA", "pl-PL", "sv-SE", "cs-CZ", "nb-NO", "da-DK",
	"bg-BG", "fi-FI", "hr-HR", "sk-SK", "zh-CN", "hu-HU", "ro-RO", "et-EE",
}

// nemotronLanguages are the base languages of nemotronLocales and "no", the
// Norwegian macrolanguage that the locale matcher resolves to nb-NO.
var nemotronLanguages = func() []string {
	out := make([]string, 0, len(nemotronLocales)+1)
	out = append(out, "no")
	for _, l := range nemotronLocales {
		b, _ := language.MustParse(l).Base()
		out = append(out, b.String())
	}
	slices.Sort(out)
	return slices.Compact(out)
}()

// modelLanguages owns the model set: each model New accepts and its sorted
// ISO 639 base languages.
var modelLanguages = map[Model][]string{
	NemotronASR:        nemotronLanguages,
	NemotronASREnglish: {"en"},
	ParakeetCTC:        {"en"},
	// Supported languages in the nvidia/parakeet-tdt-0.6b-v3 model card.
	ParakeetTDT: {
		"bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hr", "hu", "it",
		"lt", "lv", "mt", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "uk",
	},
}

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
