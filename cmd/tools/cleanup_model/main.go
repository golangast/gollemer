// cleanup_model removes any prototype entries whose name contains a colon or slash,
// which are signs of garbage Whisper debug lines being mis-parsed as intent labels.
package main

import (
	"fmt"
	"strings"

	"github.com/golangast/gollemer/internal/ai/moe"
)

func main() {
	ae, te, headW, headB, classNames, prototypes, err := moe.LoadAudioModel("models/audio_gru.json")
	if err != nil {
		fmt.Printf("❌ Load failed: %v\n", err)
		return
	}

	// Known garbage entries from early bad Whisper sessions or LLM self-listening
	badList := map[string]bool{
		"DO_YOU_ALL_WANT_TO_WAKE_UP": true,
		"SHIT_DONE_SYSTEM":           true,
		"HAVE_THEIR_UPDATES":         true,
		"X_AND_T_ACTION":             true,
		"PARDON_TO_CODE":             true,
		"RUN_THE_TEST":               true,
		"REFECTIVE_THIS":             true,
		"I_NEED_A_JOB_WORK":          true,
		"AND_YOU_DEFILE_THE_WORLD":   true,
		"I_NEED_A_FILE_OF_WORK":      true,
		"I_NEED_A_FOLLOW_UP":         true,
		"COME_HERE_DUKE":             true,
		// LLM self-listening pollution
		"BLANK_AUDIO": true,
		"OKAY":        true,
		"HEALTH":      true,
		"DOWN_I_AM":   true,
	}

	cleaned := make(map[string][]float32)
	var cleanNames []string

	removed := 0
	for _, name := range classNames {
		if strings.Contains(name, ":") || strings.Contains(name, "/") || len(name) > 50 || badList[name] {
			fmt.Printf("🗑️  Removing garbage entry: %s\n", name)
			delete(prototypes, name)
			removed++
			continue
		}
		cleaned[name] = prototypes[name]
		cleanNames = append(cleanNames, name)
	}

	if err := moe.SaveAudioModel("models/audio_gru.json", ae, te, headW, headB, cleanNames, cleaned); err != nil {
		fmt.Printf("❌ Save failed: %v\n", err)
		return
	}

	fmt.Printf("\n✅ Cleaned model: removed %d garbage entries, kept %d valid commands.\n", removed, len(cleanNames))
	for i, n := range cleanNames {
		fmt.Printf("   %2d. %s\n", i+1, n)
	}
}
