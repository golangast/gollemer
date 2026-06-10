//go:build !cgo && !windows && !darwin

package ui

type ttsSpeaker struct{}

func newSpeaker() *ttsSpeaker {
	return &ttsSpeaker{}
}

func (s *ttsSpeaker) Speak(text string) {
	// No-op for systems without CGO audio support
}
