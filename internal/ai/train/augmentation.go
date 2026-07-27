package train

import (
	"math/rand"
	"strings"
)

// WarpCommand takes a single command and generates several training variations.
// This forces the MoE architecture to learn the semantic intent rather than
// just memorizing exact token sequences.
func WarpCommand(input string) []string {
	variations := []string{input}
	words := strings.Fields(input)

	// Variation 1: Typo Injection (Simulate fast typing)
	if len(words) > 0 {
		// Common typo: replace "commit" with "comit"
		typoed := strings.Replace(input, "commit", "comit", 1)
		if typoed != input {
			variations = append(variations, typoed)
		}

		// Another typo: double "m" or transposition
		if strings.Contains(input, "create") {
			variations = append(variations, strings.Replace(input, "create", "cretae", 1))
		}
	}

	// Variation 2: Flag Shuffling (CLI specific)
	// Example: "git commit -m 'msg' --amend" -> "git commit --amend -m 'msg'"
	if len(words) > 3 {
		shuffled := make([]string, len(words))
		copy(shuffled, words)

		// Target indices > 1 to preserve leading core command (e.g., "git commit")
		if len(shuffled) > 3 {
			rand.Shuffle(len(shuffled)-2, func(i, j int) {
				shuffled[i+2], shuffled[j+2] = shuffled[j+2], shuffled[i+2]
			})
			variations = append(variations, strings.Join(shuffled, " "))
		}
	}

	// Variation 3: Verb Synonym Swap
	// "show files" -> "list files"
	synonyms := map[string]string{
		"show":   "list",
		"list":   "show",
		"remove": "delete",
		"delete": "remove",
		"make":   "create",
		"create": "make",
	}

	for key, value := range synonyms {
		if strings.Contains(strings.ToLower(input), key) {
			// Replace only the specific instance
			swapped := strings.Replace(strings.ToLower(input), key, value, 1)
			variations = append(variations, swapped)
		}
	}

	return variations
}
