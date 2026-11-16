// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Built-in voices of the supported KittenTTS models and the language each speaks.

package kittentts

import (
	"fmt"
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
// Only KittenML/kitten-tts-2 is supported. Its voice carries the language: the
// model has no language argument, and an English voice reading German "will
// not work well" (docs/voices-and-expression.md, "Languages", in
// KittenML/KittenTTS at the revision kittenttssrv pins). The smaller KittenTTS
// 2 nano, pico and mini models instead select a language with a language
// argument; the pinned kittenml predates them and rejects their repositories,
// so they have no voices here.
func Voices(model string) ([]Voice, error) {
	if model != "KittenML/kitten-tts-2" {
		return nil, fmt.Errorf("KittenTTS model %q is not supported; use KittenML/kitten-tts-2", model)
	}
	return slices.Clone(kittenTTS2Voices), nil
}

// kittenTTS2Voices lists voices/voices.json of KittenML/kitten-tts-2 at
// Hugging Face revision c93cb7c271baa51d9ae8204d5277988906f235a2. Each
// non-English voice_id carries its ISO 639-1 code as a "fr_" prefix or a
// "_fr" suffix, and its name starts with the English language name.
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
