// Package tokenizer defines the special control tokens reserved for
// hardware intent execution.  These tokens are added verbatim to the
// social vocabulary before training so the model never splits them.
package tokenizer

// SpecialHardwareTokens lists every control word that maps 1-to-1 with a
// hardware action.  They are wrapped in angle-brackets to guarantee they
// can never be confused with natural-language words.
var SpecialHardwareTokens = []string{
	// ── Intent tokens ────────────────────────────────────────────────────────
	"<INTENT_CAMERA_CAPTURE>",
	"<INTENT_CAMERA_CAPTURE_ANALYZE>",
	"<INTENT_CAMERA_CAPTURE_SAVE>",
	"<INTENT_CAMERA_DISABLE>",
	"<INTENT_CAMERA_ENABLE>",
	"<INTENT_CAMERA_ANALYZE_ENVIRONMENT>",
	"<INTENT_CAMERA_ANALYZE_COLOR>",
	"<INTENT_CAMERA_FACE_DETECTION>",
	"<INTENT_CAMERA_MOTION_DETECTION>",
	"<INTENT_AUDIO_VOLUME_UP>",
	"<INTENT_AUDIO_VOLUME_DOWN>",
	"<INTENT_AUDIO_VOLUME_MAX>",
	"<INTENT_AUDIO_RECORD>",
	"<INTENT_AUDIO_RECORD_STOP>",
	"<INTENT_AUDIO_PLAYBACK>",
	"<INTENT_AUDIO_PLAYBACK_PAUSE>",
	"<INTENT_AUDIO_SPEECH_SYNTHESIS>",
	"<INTENT_MICROPHONE_MUTE>",
	"<INTENT_MICROPHONE_UNMUTE>",
	"<INTENT_MICROPHONE_STATUS_CHECK>",
	"<INTENT_MICROPHONE_MONITOR>",

	// ── Action role tokens ───────────────────────────────────────────────────
	"<ACT_TAKE>",
	"<ACT_SNAP>",
	"<ACT_CAPTURE>",
	"<ACT_ACTIVATE>",
	"<ACT_INITIALIZE>",
	"<ACT_DISABLE>",
	"<ACT_ENABLE>",
	"<ACT_MUTE>",
	"<ACT_UNMUTE>",
	"<ACT_LOWER>",
	"<ACT_RAISE>",
	"<ACT_MAXIMIZE>",
	"<ACT_PAUSE>",
	"<ACT_RECORD>",
	"<ACT_STOP>",
	"<ACT_BROADCAST>",
	"<ACT_PLAY>",
	"<ACT_LISTEN>",
	"<ACT_CHECK>",
	"<ACT_ANALYZE>",

	// ── Device role tokens ───────────────────────────────────────────────────
	"<DEV_CAMERA>",
	"<DEV_SPEAKER>",
	"<DEV_MICROPHONE>",
	"<DEV_HEADPHONES>",
	"<DEV_AUDIO_JACK>",
}

// IsSpecialToken returns true if tok is a registered hardware control token.
func IsSpecialToken(tok string) bool {
	for _, t := range SpecialHardwareTokens {
		if t == tok {
			return true
		}
	}
	return false
}

// InjectIntoVocab adds every special hardware token to the provided vocabulary
// map (word → id) and its reverse slice (id → word), starting from the next
// available ID.  Call this before training so the embedding matrix has fixed
// slots reserved for each control token.
func InjectIntoVocab(wordToToken map[string]int, tokenToWord *[]string) {
	for _, tok := range SpecialHardwareTokens {
		if _, exists := wordToToken[tok]; exists {
			continue // already injected
		}
		id := len(*tokenToWord)
		wordToToken[tok] = id
		*tokenToWord = append(*tokenToWord, tok)
	}
}
