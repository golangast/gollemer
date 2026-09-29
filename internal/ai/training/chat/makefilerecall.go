package chat

// Makefile recall: deterministic command answers from training data.
//
// The makefile brain maps "how do i ..." to exact make commands. The
// tiny model confuses similar inputs ("retrain gocli" vs "retrain the
// code model" vs "open the start guide"), so exact training-pair
// matches return the trained command verbatim instead of generating.
// Unmatched inputs still go to the neural model for generalization.

import (
	"sync"
)

var (
	makefileRecallOnce sync.Once
	makefileRecallMap  map[string]string
)

// initMakefileRecall loads every makefile pair into a
// normalized-input -> command map. Last pair wins on duplicates.
func initMakefileRecall(projectRoot string) {
	makefileRecallOnce.Do(func() {
		makefileRecallMap = map[string]string{}
		for key, out := range loadRecallPairs(projectRoot, MakefileDomain) {
			makefileRecallMap[key] = out
		}
	})
}

// LookupMakefileRecall returns the trained command for an exact
// (normalized) training-input match. Call initMakefileRecall first.
func LookupMakefileRecall(input string) (string, bool) {
	if makefileRecallMap == nil {
		return "", false
	}
	out, ok := makefileRecallMap[normalizeRecallInput(input)]
	return out, ok
}
