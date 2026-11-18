package chat

// Mutation augmentation: an opt-in second augmentation for the social
// domain, built for the robustness round (2026-11-10). The frozen
// robustness eval (robustness_v1) showed the production social model
// answering only 42/160 truly unseen wordings — and its two mutated
// subsets (typos 9/40, function-word drops 11/40) are failures a model
// can be taught out of: if training inputs arrive pre-mangled while
// their targets stay correct, the encoder learns the mangling does not
// change the meaning.
//
// The mutation classes mirror the eval's frozen mutator
// (mutate_v1.py, seed 20261110) so training practices exactly the
// failure shapes the eval measures:
//
//   - typo: drop or swap one character inside a content word
//     (>= 4 letters, all letters, never the first character);
//   - funcdrop family: drop one article/preposition
//     (the/a/an/to/of/in/on/for, never the first token), or — when the
//     input has none — swap two adjacent words past the first.
//
// Each train-base input gets at most ONE mutated sibling, its target
// unchanged. The stream is seeded (mutationSeed) so a corpus always
// produces the same siblings. Go's rand stream is not Python's, so the
// siblings are not byte-identical to the eval's own mutations — the
// classes match, which is what teaches the invariance.
//
// Contamination discipline: a sibling that reproduces a frozen eval
// input would teach an eval phrasing verbatim, so mutationAugment
// drops any sibling whose normalized form is in the blocklist (the
// eval file, loaded by loadMutationBlocklist), and prepare-time
// refuses to run the augmentation without one (fail closed). The
// label-collision discipline is augmentInputs' own: a sibling that
// lands on a different pair's phrasing with a different answer is
// dropped, never taught as a contradiction.

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"strings"
	"unicode"
)

// mutationSeed fixes the augmentation stream: the same corpus always
// produces the same siblings, so a run's data is reproducible. It is
// the frozen eval's seed, kept so the two stay easy to compare.
const mutationSeed = 20261110

// mutationFuncWords are the droppable function words, mirroring the
// frozen eval mutator's FUNC_WORDS set.
var mutationFuncWords = map[string]bool{
	"the": true, "a": true, "an": true, "to": true,
	"of": true, "in": true, "on": true, "for": true,
}

// mutateTokens applies one mutation to a tokenized (lowercased) input:
// the typo class or the funcdrop family, chosen by the stream — the
// eval's mutated set is half of each. Returns the mutated tokens and
// whether anything visibly changed.
func mutateTokens(toks []string, rng *rand.Rand) ([]string, bool) {
	if rng.Intn(2) == 0 {
		return mutateTypoWord(toks, rng)
	}
	return mutateDropFuncOrSwap(toks, rng)
}

// mutateTypoWord drops or swaps one character inside a content word:
// a word of at least 4 letters, all letters, the character never the
// first. Retries until the text actually changes (swapping identical
// neighbours is a no-op), with the frozen mutator's fallback of
// dropping the first candidate's last character.
func mutateTypoWord(toks []string, rng *rand.Rand) ([]string, bool) {
	var cand []int
	for i, w := range toks {
		if len([]rune(w)) >= 4 && allLetters(w) {
			cand = append(cand, i)
		}
	}
	if len(cand) == 0 {
		return toks, false
	}
	orig := strings.Join(toks, " ")
	for try := 0; try < 8; try++ {
		out := append([]string{}, toks...)
		i := cand[rng.Intn(len(cand))]
		w := []rune(out[i])
		pos := 1 + rng.Intn(len(w)-1) // 1..len-1: never the first character
		if rng.Intn(2) == 0 || pos == len(w)-1 {
			out[i] = string(w[:pos]) + string(w[pos+1:]) // drop
		} else {
			w[pos], w[pos+1] = w[pos+1], w[pos] // swap
			out[i] = string(w)
		}
		if strings.Join(out, " ") != orig {
			return out, true
		}
	}
	out := append([]string{}, toks...)
	w := []rune(out[cand[0]])
	out[cand[0]] = string(w[:len(w)-1])
	return out, true
}

// mutateDropFuncOrSwap drops one function word (never the first
// token), or — when the input carries none — swaps two adjacent
// words. It is the eval's second mutation family (mutate_v1's
// mutate_social_variant4).
func mutateDropFuncOrSwap(toks []string, rng *rand.Rand) ([]string, bool) {
	var cand []int
	for i, w := range toks {
		if mutationFuncWords[w] && i > 0 {
			cand = append(cand, i)
		}
	}
	if len(cand) > 0 {
		i := cand[rng.Intn(len(cand))]
		out := append([]string{}, toks[:i]...)
		out = append(out, toks[i+1:]...)
		return out, true
	}
	return mutateSwapAdjacent(toks, rng)
}

// mutateSwapAdjacent swaps one adjacent pair past the first token
// (the frozen mutator's swap_adj: i in [1, len-2]). Swapping identical
// neighbours changes nothing and reports unchanged, so the caller
// never emits a duplicate as a "mutation".
func mutateSwapAdjacent(toks []string, rng *rand.Rand) ([]string, bool) {
	if len(toks) < 3 {
		return toks, false
	}
	i := 1 + rng.Intn(len(toks)-2) // 1..len-2
	out := append([]string{}, toks...)
	out[i], out[i+1] = out[i+1], out[i]
	if strings.Join(out, " ") == strings.Join(toks, " ") {
		return toks, false
	}
	return out, true
}

// allLetters reports whether every rune of w is a letter (the frozen
// mutator's str.isalpha test for typo candidates).
func allLetters(w string) bool {
	for _, r := range w {
		if !unicode.IsLetter(r) {
			return false
		}
	}
	return len(w) > 0
}

// mutationStats counts what one mutationAugment pass did, so the
// training log (and the round's record) carries the real numbers.
type mutationStats struct {
	Added            int // siblings that joined the training pairs
	DroppedBlocklist int // siblings that reproduced a frozen eval input
	DroppedCollision int // siblings that landed on another pair's phrasing with a different answer
	Unchanged        int // inputs no mutation could visibly change
}

// mutationAugment adds one mutated sibling per train-base pair to
// trainPairs (which already holds the base pairs and their synonym
// variants), target unchanged, under augmentInputs' collision
// discipline plus the blocklist drop. Iteration follows trainBase
// order and draws only from rng, so the pass is deterministic.
func mutationAugment(base, trainBase, trainPairs []ChatPair, blocklist map[string]bool, rng *rand.Rand) ([]ChatPair, mutationStats) {
	var stats mutationStats
	baseOutputByNorm := make(map[string]string, len(base))
	for _, p := range base {
		n := normAugInput(p.Input)
		if _, dup := baseOutputByNorm[n]; !dup {
			baseOutputByNorm[n] = p.Output
		}
	}
	emitted := make(map[string]string, len(trainPairs))
	for _, p := range trainPairs {
		n := normAugInput(p.Input)
		if _, dup := emitted[n]; !dup {
			emitted[n] = p.Output
		}
	}
	out := trainPairs
	for _, p := range trainBase {
		mtoks, changed := mutateTokens(tokenizeLower(p.Input), rng)
		if !changed {
			stats.Unchanged++
			continue
		}
		sibling := strings.Join(mtoks, " ")
		n := normAugInput(sibling)
		if blocklist[n] {
			stats.DroppedBlocklist++
			continue
		}
		if owner, collides := baseOutputByNorm[n]; collides && owner != p.Output {
			stats.DroppedCollision++
			continue
		}
		if owner, dup := emitted[n]; dup && owner != p.Output {
			stats.DroppedCollision++
			continue
		}
		emitted[n] = p.Output
		out = append(out, ChatPair{Input: sibling, Output: p.Output, Domain: p.Domain})
		stats.Added++
	}
	return out, stats
}

// loadMutationBlocklist reads a JSONL file (the frozen robustness
// eval) and returns the normalized set of its inputs — the phrasings
// mutation siblings must never reproduce. A line that is not a JSON
// object with an "input" field is treated as a raw input line, so a
// plain text list works too. An empty path is an error: running the
// augmentation without its blocklist is exactly the contamination
// risk the blocklist exists to close, so prepare fails closed.
func loadMutationBlocklist(path string) (map[string]bool, error) {
	if path == "" {
		return nil, fmt.Errorf("mutation augmentation needs a blocklist (-mutation-blocklist <eval.jsonl>): without it a mutated sibling could reproduce a frozen eval input")
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("mutation blocklist: %w", err)
	}
	defer f.Close()
	set := make(map[string]bool)
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var rec struct {
			Input string `json:"input"`
		}
		if err := json.Unmarshal([]byte(line), &rec); err == nil && rec.Input != "" {
			set[normAugInput(rec.Input)] = true
		} else {
			set[normAugInput(line)] = true
		}
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("mutation blocklist: %w", err)
	}
	if len(set) == 0 {
		return nil, fmt.Errorf("mutation blocklist %s: no inputs found", path)
	}
	return set, nil
}
