//go:build cgo || windows || darwin

package ui

import (
	htgotts "github.com/hegedustibor/htgo-tts"
	"github.com/hegedustibor/htgo-tts/handlers"
	"github.com/hegedustibor/htgo-tts/voices"
)

type ttsSpeaker struct {
	speech htgotts.Speech
}

func newSpeaker() *ttsSpeaker {
	return &ttsSpeaker{
		speech: htgotts.Speech{Folder: "audio", Language: voices.English, Handler: &handlers.Native{}},
	}
}

func (s *ttsSpeaker) Speak(text string) {
	s.speech.Speak(text)
}
