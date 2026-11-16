// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Built-in voices of the supported KittenTTS models and the language each speaks.

package kittentts

import (
	"fmt"
	"maps"
	"slices"
)

// Voice is a built-in voice and the language it speaks.
type Voice struct {
	Name string
	// Language is the ISO 639-1 code of the language the voice speaks; empty
	// means English. Text in another language than English disables the
	// worker's English text normalization.
	Language string
}

// Voices returns the built-in voices of model in name order.
//
// KittenML/kitten-tts-2 has no language argument: each voice carries its
// language, and an English voice reading German "will not work well"
// (docs/voices-and-expression.md, "Languages", in KittenML/KittenTTS at the
// revision kittenttssrv pins).
//
// The KittenTTS 2 nano, pico and mini models select a language with a
// language argument, and each language has its own voices. Mini also lists
// its English voices under German, Spanish, Italian, Portuguese and Russian;
// they are listed here only as English, so that each name speaks one
// language. Mini's Japanese is omitted: it needs kittenml's "ja" extra, whose
// pyopenjtalk and unidic-lite build from source with a C++ toolchain.
func Voices(model string) ([]Voice, error) {
	m, err := lookup(model)
	if err != nil {
		return nil, err
	}
	return slices.Clone(m.voices), nil
}

// Revision returns the Hugging Face commit of model whose config Voices
// lists. kittenttssrv loads the model at this commit, so the listed voices and
// languages cannot drift with the repository's main.
func Revision(model string) (string, error) {
	m, err := lookup(model)
	if err != nil {
		return "", err
	}
	return m.revision, nil
}

// Models are the models Voices supports, sorted.
var Models = slices.Sorted(maps.Keys(models))

type modelInfo struct {
	revision string
	voices   []Voice
}

var models = map[string]modelInfo{
	"KittenML/kitten-tts-2":      {"c93cb7c271baa51d9ae8204d5277988906f235a2", kittenTTS2Voices},
	"KittenML/kitten-tts-2-mini": {"c01797caec3967b400da457656e011eb81f530a4", kittenTTS2MiniVoices},
	"KittenML/kitten-tts-2-nano": {"54526b8d8e45333cd720d3a082c106aacf416064", kittenTTS2NanoVoices},
	"KittenML/kitten-tts-2-pico": {"da8c9d029e16d9302f1827adb303d92316e25f73", kittenTTS2PicoVoices},
}

func lookup(model string) (modelInfo, error) {
	m, ok := models[model]
	if !ok {
		return modelInfo{}, fmt.Errorf("KittenTTS model %q is not supported; use one of %q", model, Models)
	}
	return m, nil
}

// kittenTTS2Voices lists voices/voices.json of KittenML/kitten-tts-2 at its
// revision. Each non-English voice_id carries its ISO 639-1 code as a "fr_"
// prefix or a "_fr" suffix, and its name starts with the English language name.
var kittenTTS2Voices = []Voice{
	{Name: "AirportAnnouncer", Language: "en"},
	{Name: "Alfred", Language: "en"},
	{Name: "Arabic", Language: "ar"},
	{Name: "ArabicFalconer", Language: "ar"},
	{Name: "ArabicMother", Language: "ar"},
	{Name: "ArabicNews", Language: "ar"},
	{Name: "AudiobookNarratorFemale", Language: "en"},
	{Name: "Bella", Language: "en"},
	{Name: "BoredTeenager", Language: "en"},
	{Name: "Bruno", Language: "en"},
	{Name: "CalmHostageNegotiator", Language: "en"},
	{Name: "Chinese", Language: "zh"},
	{Name: "ChineseLivestreamSeller", Language: "zh"},
	{Name: "ChineseNewsCctv", Language: "zh"},
	{Name: "ChineseXianxiaMaster", Language: "zh"},
	{Name: "Claire", Language: "en"},
	{Name: "Dante", Language: "en"},
	{Name: "Diana", Language: "en"},
	{Name: "Dolores", Language: "en"},
	{Name: "DutchBaker", Language: "nl"},
	{Name: "DutchGrandpaStory", Language: "nl"},
	{Name: "DutchWeatherPresenter", Language: "nl"},
	{Name: "Edith", Language: "en"},
	{Name: "Eleanor", Language: "en"},
	{Name: "Elliot", Language: "en"},
	{Name: "FairytaleNarratorWhimsical", Language: "en"},
	{Name: "Frank", Language: "en"},
	{Name: "French", Language: "fr"},
	{Name: "FrenchDocumentary", Language: "fr"},
	{Name: "FrenchGrandmereRecette", Language: "fr"},
	{Name: "FrenchTeacher", Language: "fr"},
	{Name: "FrenchWaiter", Language: "fr"},
	{Name: "German", Language: "de"},
	{Name: "GermanKneipe", Language: "de"},
	{Name: "GermanProfessor", Language: "de"},
	{Name: "GermanTrainAnnouncement", Language: "de"},
	{Name: "Grace", Language: "en"},
	{Name: "GrandmaBedtimeStory", Language: "en"},
	{Name: "GriefEulogy", Language: "en"},
	{Name: "Herbert", Language: "en"},
	{Name: "Hindi", Language: "hi"},
	{Name: "HindiBollywoodDrama", Language: "hi"},
	{Name: "HindiCollegeFriends", Language: "hi"},
	{Name: "HindiDadiKahani", Language: "hi"},
	{Name: "HindiYogaTeacher", Language: "hi"},
	{Name: "Hugo", Language: "en"},
	{Name: "IndonesianNewsAnchor", Language: "id"},
	{Name: "IndonesianTourGuide", Language: "id"},
	{Name: "Iris", Language: "en"},
	{Name: "Italian", Language: "it"},
	{Name: "ItalianFashionDesigner", Language: "it"},
	{Name: "ItalianFootballFan", Language: "it"},
	{Name: "ItalianGondolier", Language: "it"},
	{Name: "ItalianNonnaKitchen", Language: "it"},
	{Name: "ItalianOperaTenor", Language: "it"},
	{Name: "JapaneseAnimeHeroine", Language: "ja"},
	{Name: "JapaneseGrandma", Language: "ja"},
	{Name: "JapaneseSalarymanApology", Language: "ja"},
	{Name: "JapaneseTrainConductor", Language: "ja"},
	{Name: "Jasper", Language: "en"},
	{Name: "Julian", Language: "en"},
	{Name: "Kiki", Language: "en"},
	{Name: "KoreanGrandpaAdvice", Language: "ko"},
	{Name: "KoreanKdramaConfession", Language: "ko"},
	{Name: "KoreanNewsAnchor", Language: "ko"},
	{Name: "Laurence", Language: "en"},
	{Name: "Leo", Language: "en"},
	{Name: "Luna", Language: "en"},
	{Name: "Maeve", Language: "en"},
	{Name: "Marcus", Language: "en"},
	{Name: "Martha", Language: "en"},
	{Name: "Matthew", Language: "en"},
	{Name: "Miles", Language: "en"},
	{Name: "NervousFirstDate", Language: "en"},
	{Name: "NewsAnchorFemale", Language: "en"},
	{Name: "Otis", Language: "en"},
	{Name: "PilotCabinAnnouncement", Language: "en"},
	{Name: "PolishBabciaPierogi", Language: "pl"},
	{Name: "PolishMountainGuide", Language: "pl"},
	{Name: "PolishNewsAnchor", Language: "pl"},
	{Name: "PolishPoetryStudent", Language: "pl"},
	{Name: "Portuguese", Language: "pt"},
	{Name: "PortugueseBeach", Language: "pt"},
	{Name: "PortugueseBusiness", Language: "pt"},
	{Name: "PortugueseFado", Language: "pt"},
	{Name: "PortugueseGrandpaFarm", Language: "pt"},
	{Name: "PortugueseNewsAnchor", Language: "pt"},
	{Name: "Raven", Language: "en"},
	{Name: "Reginald", Language: "en"},
	{Name: "Rosie", Language: "en"},
	{Name: "Russian", Language: "ru"},
	{Name: "RussianBabushka", Language: "ru"},
	{Name: "RussianFairyTale", Language: "ru"},
	{Name: "RussianHockeyCheer", Language: "ru"},
	{Name: "RussianLiteratureTeacher", Language: "ru"},
	{Name: "RussianSpaceControl", Language: "ru"},
	{Name: "RussianTaxiDriver", Language: "ru"},
	{Name: "Sable", Language: "en"},
	{Name: "Saoirse", Language: "en"},
	{Name: "SarcasticCoworker", Language: "en"},
	{Name: "Serena", Language: "en"},
	{Name: "Spanish", Language: "es"},
	{Name: "SpanishAbuela", Language: "es"},
	{Name: "SpanishBreakupSad", Language: "es"},
	{Name: "SpanishNews", Language: "es"},
	{Name: "SpanishTango", Language: "es"},
	{Name: "TurkishFisherman", Language: "tr"},
	{Name: "TurkishGrandmaCooking", Language: "tr"},
	{Name: "TurkishNewsAnchor", Language: "tr"},
	{Name: "Victor", Language: "en"},
	{Name: "Victoria", Language: "en"},
	{Name: "Vincent", Language: "en"},
	{Name: "Walter", Language: "en"},
	{Name: "WeatherPresenterCheerful", Language: "en"},
	{Name: "Willow", Language: "en"},
}

// kittenTTS2NanoVoices lists the voices of config.json in
// KittenML/kitten-tts-2-nano at its revision.
var kittenTTS2NanoVoices = []Voice{
	{Name: "Alessandro", Language: "it"},
	{Name: "Alexei", Language: "ru"},
	{Name: "Ananya", Language: "hi"},
	{Name: "Anna", Language: "de"},
	{Name: "Arjun", Language: "hi"},
	{Name: "Caio", Language: "pt"},
	{Name: "Danny", Language: "en"},
	{Name: "Davide", Language: "it"},
	{Name: "Edith", Language: "en"},
	{Name: "Eleanor", Language: "en"},
	{Name: "Elena", Language: "es"},
	{Name: "Francesca", Language: "it"},
	{Name: "Gabriel", Language: "fr"},
	{Name: "Giorgio", Language: "it"},
	{Name: "Grace", Language: "en"},
	{Name: "Gustavo", Language: "pt"},
	{Name: "Iris", Language: "en"},
	{Name: "Juliana", Language: "pt"},
	{Name: "Juliette", Language: "fr"},
	{Name: "Larissa", Language: "pt"},
	{Name: "Lukas", Language: "de"},
	{Name: "Martha", Language: "en"},
	{Name: "Mateo", Language: "es"},
	{Name: "Max", Language: "en"},
	{Name: "Mia", Language: "de"},
	{Name: "Nicolas", Language: "fr"},
	{Name: "Olga", Language: "ru"},
	{Name: "Otis", Language: "en"},
	{Name: "Pablo", Language: "es"},
	{Name: "Paul", Language: "de"},
	{Name: "Priya", Language: "hi"},
	{Name: "Serena", Language: "en"},
	{Name: "Sergei", Language: "ru"},
	{Name: "Sofia", Language: "es"},
	{Name: "Tatiana", Language: "ru"},
	{Name: "Vikram", Language: "hi"},
	{Name: "Vincent", Language: "en"},
}

// kittenTTS2PicoVoices lists the voices of config.json in
// KittenML/kitten-tts-2-pico at its revision.
var kittenTTS2PicoVoices = []Voice{
	{Name: "Alessandro", Language: "it"},
	{Name: "Alexei", Language: "ru"},
	{Name: "Amelie", Language: "fr"},
	{Name: "Anna", Language: "de"},
	{Name: "Arjun", Language: "hi"},
	{Name: "Baptiste", Language: "fr"},
	{Name: "Caio", Language: "pt"},
	{Name: "Davide", Language: "it"},
	{Name: "Edith", Language: "en"},
	{Name: "Eleanor", Language: "en"},
	{Name: "Elena", Language: "es"},
	{Name: "Francesca", Language: "it"},
	{Name: "Frank", Language: "en"},
	{Name: "Giorgio", Language: "it"},
	{Name: "Gustavo", Language: "pt"},
	{Name: "Juliana", Language: "pt"},
	{Name: "Larissa", Language: "pt"},
	{Name: "Lea", Language: "fr"},
	{Name: "Lukas", Language: "de"},
	{Name: "Mateo", Language: "es"},
	{Name: "Mia", Language: "de"},
	{Name: "Olga", Language: "ru"},
	{Name: "Otis", Language: "en"},
	{Name: "Pablo", Language: "es"},
	{Name: "Paul", Language: "de"},
	{Name: "Priya", Language: "hi"},
	{Name: "Sergei", Language: "ru"},
	{Name: "Sofia", Language: "es"},
	{Name: "Tatiana", Language: "ru"},
	{Name: "Theo", Language: "fr"},
}

// kittenTTS2MiniVoices lists the voices of config.json in
// KittenML/kitten-tts-2-mini at its revision, without Japanese, and with its
// English voices only in English.
var kittenTTS2MiniVoices = []Voice{
	{Name: "Aino", Language: "fi"},
	{Name: "Ananya", Language: "hi"},
	{Name: "Anastasia", Language: "ru"},
	{Name: "Antoine", Language: "fr"},
	{Name: "Arjun", Language: "hi"},
	{Name: "Astrid", Language: "sv"},
	{Name: "Baraka", Language: "sw"},
	{Name: "Beatriz", Language: "pt"},
	{Name: "Bram", Language: "nl"},
	{Name: "Camila", Language: "pt"},
	{Name: "Camille", Language: "fr"},
	{Name: "Carmen", Language: "es"},
	{Name: "Chiara", Language: "it"},
	{Name: "Chloe", Language: "fr"},
	{Name: "Daan", Language: "nl"},
	{Name: "Danny", Language: "en"},
	{Name: "Diana", Language: "en"},
	{Name: "Diego", Language: "es"},
	{Name: "Dmitri", Language: "ru"},
	{Name: "Edith", Language: "en"},
	{Name: "Eero", Language: "fi"},
	{Name: "Eleanor", Language: "en"},
	{Name: "Elif", Language: "tr"},
	{Name: "Elsa", Language: "sv"},
	{Name: "Etienne", Language: "fr"},
	{Name: "Felix", Language: "de"},
	{Name: "Fenna", Language: "nl"},
	{Name: "Freja", Language: "da"},
	{Name: "Giulia", Language: "it"},
	{Name: "Grace", Language: "en"},
	{Name: "Greta", Language: "de"},
	{Name: "Helmi", Language: "fi"},
	{Name: "Iris", Language: "en"},
	{Name: "Ivan", Language: "ru"},
	{Name: "Javier", Language: "es"},
	{Name: "Jihoon", Language: "ko"},
	{Name: "Jonas", Language: "de"},
	{Name: "Joris", Language: "nl"},
	{Name: "Juho", Language: "fi"},
	{Name: "Julien", Language: "fr"},
	{Name: "Juma", Language: "sw"},
	{Name: "Lena", Language: "de"},
	{Name: "Louis", Language: "fr"},
	{Name: "Luca", Language: "it"},
	{Name: "Lucia", Language: "es"},
	{Name: "Mads", Language: "da"},
	{Name: "Manon", Language: "fr"},
	{Name: "Marco", Language: "it"},
	{Name: "Martha", Language: "en"},
	{Name: "Mathis", Language: "fr"},
	{Name: "Max", Language: "en"},
	{Name: "Minjun", Language: "ko"},
	{Name: "Natasha", Language: "ru"},
	{Name: "Neema", Language: "sw"},
	{Name: "Oskar", Language: "sv"},
	{Name: "Otis", Language: "en"},
	{Name: "Priya", Language: "hi"},
	{Name: "Rafael", Language: "pt"},
	{Name: "Rasmus", Language: "da"},
	{Name: "Sanne", Language: "nl"},
	{Name: "Seoyeon", Language: "ko"},
	{Name: "Serena", Language: "en"},
	{Name: "Thiago", Language: "pt"},
	{Name: "Vikram", Language: "hi"},
	{Name: "Ville", Language: "fi"},
	{Name: "Vincent", Language: "en"},
	{Name: "Zawadi", Language: "sw"},
	{Name: "Zeynep", Language: "tr"},
}
