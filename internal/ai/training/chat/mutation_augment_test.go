package chat

// Mutation augmentation: the robustness round's trainer machinery.
// Pins the mutator classes (mirroring the frozen eval's mutate_v1),
// the determinism of the sibling stream, the target discipline, the
// blocklist and label-collision drops, and — the launch-safety pin —
// that data prepared with the flag OFF is byte-identical to the
// legacy preparation path.

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// oneTypoAway reports whether mut is orig with exactly one character
// dropped, or exactly one adjacent pair swapped.
func oneTypoAway(orig, mut string) bool {
	o, m := []rune(orig), []rune(mut)
	if len(m) == len(o)-1 {
		i := 0
		for i < len(m) && o[i] == m[i] {
			i++
		}
		return string(o[:i])+string(o[i+1:]) == mut
	}
	if len(m) == len(o) {
		var diff []int
		for i := range o {
			if o[i] != m[i] {
				diff = append(diff, i)
			}
		}
		if len(diff) != 2 || diff[1] != diff[0]+1 {
			return false
		}
		return o[diff[0]] == m[diff[1]] && o[diff[1]] == m[diff[0]]
	}
	return false
}

func TestMutateTypoWordChangesOneWord(t *testing.T) {
	toks := tokenizeLower("i am feeling pretty tired")
	rng := rand.New(rand.NewSource(mutationSeed))
	for draw := 0; draw < 20; draw++ {
		out, changed := mutateTypoWord(toks, rng)
		if !changed {
			t.Fatalf("draw %d: expected a change", draw)
		}
		if len(out) != len(toks) {
			t.Fatalf("draw %d: token count changed: %v", draw, out)
		}
		diffs := 0
		for i := range toks {
			if out[i] != toks[i] {
				diffs++
				if !oneTypoAway(toks[i], out[i]) {
					t.Fatalf("draw %d: %q -> %q is not one typo away", draw, toks[i], out[i])
				}
				if len(toks[i]) < 4 {
					t.Fatalf("draw %d: short word %q was mutated", draw, toks[i])
				}
			}
		}
		if diffs != 1 {
			t.Fatalf("draw %d: %d words changed in %v", draw, diffs, out)
		}
	}
}

func TestMutateDropFuncOrSwap(t *testing.T) {
	rng := rand.New(rand.NewSource(mutationSeed))
	// Droppable function words present: exactly one token disappears,
	// and it is one of the function words.
	toks := tokenizeLower("i am going to the park later")
	out, changed := mutateDropFuncOrSwap(toks, rng)
	if !changed || len(out) != len(toks)-1 {
		t.Fatalf("expected one token dropped, got %v (changed=%v)", out, changed)
	}
	removed := ""
	oi := 0
	for _, w := range toks {
		if oi < len(out) && out[oi] == w {
			oi++
		} else {
			removed = w
		}
	}
	if !mutationFuncWords[removed] {
		t.Fatalf("dropped token %q is not a function word", removed)
	}
	// No function words: an adjacent swap past the first token.
	toks2 := tokenizeLower("cats sleep soundly all day")
	out2, changed2 := mutateDropFuncOrSwap(toks2, rng)
	if !changed2 || len(out2) != len(toks2) {
		t.Fatalf("expected an adjacent swap, got %v (changed=%v)", out2, changed2)
	}
	if out2[0] != toks2[0] {
		t.Fatalf("first token moved: %v", out2)
	}
	counts := map[string]int{}
	for _, w := range toks2 {
		counts[w]++
	}
	for _, w := range out2 {
		counts[w]--
	}
	for w, c := range counts {
		if c != 0 {
			t.Fatalf("swap changed the token multiset at %q", w)
		}
	}
	// Nothing mutable: a two-letter greeting has no typo candidate,
	// no function word, and too few tokens to swap.
	if _, changed := mutateTokens(tokenizeLower("hi"), rng); changed {
		t.Fatal("\"hi\" must be unmutable")
	}
}

func mutationFixture() []ChatPair {
	return []ChatPair{
		{Input: "i am feeling pretty tired this evening", Output: "Rest helps. Take it easy tonight.", Domain: "social"},
		{Input: "i am going to the park later", Output: "Enjoy the fresh air!", Domain: "social"},
		{Input: "what a bright sunny morning it is", Output: "A lovely start to the day.", Domain: "social"},
		{Input: "i love this song very much", Output: "It is a great one, isn't it?", Domain: "social"},
		{Input: "the wind outside is blowing hard", Output: "Hold on to your hat out there.", Domain: "social"},
		{Input: "i had a big breakfast today", Output: "A strong start! What did you have?", Domain: "social"},
	}
}

func TestMutationAugmentDeterministic(t *testing.T) {
	base := mutationFixture()
	run := func() ([]ChatPair, mutationStats) {
		rng := rand.New(rand.NewSource(mutationSeed))
		return mutationAugment(base, base, append([]ChatPair{}, base...), map[string]bool{}, rng)
	}
	got1, stats1 := run()
	got2, stats2 := run()
	if !reflect.DeepEqual(got1, got2) {
		t.Fatalf("same seed, different siblings:\n%v\n%v", got1, got2)
	}
	if stats1 != stats2 {
		t.Fatalf("same seed, different stats: %+v vs %+v", stats1, stats2)
	}
	if stats1.Added == 0 {
		t.Fatal("expected at least one sibling from the fixture")
	}
	t.Logf("stats: %+v", stats1)
}

func TestMutationAugmentTargetsUntouched(t *testing.T) {
	base := mutationFixture()
	rng := rand.New(rand.NewSource(mutationSeed))
	got, stats := mutationAugment(base, base, append([]ChatPair{}, base...), map[string]bool{}, rng)
	sourceOutput := map[string]string{}
	for _, p := range base {
		sourceOutput[normAugInput(p.Input)] = p.Output
	}
	siblings := got[len(base):]
	if len(siblings) != stats.Added {
		t.Fatalf("suffix holds %d pairs, stats say %d added", len(siblings), stats.Added)
	}
	for _, s := range siblings {
		// A sibling's answer must be the answer of a base pair — and,
		// paired against the mutation discipline, it must differ in
		// input from every base pair it could be confused with.
		found := false
		for _, p := range base {
			if s.Output == p.Output {
				found = true
				if normAugInput(s.Input) == normAugInput(p.Input) {
					t.Fatalf("sibling input equals its own source input: %q", s.Input)
				}
			}
		}
		if !found {
			t.Fatalf("sibling output %q belongs to no base pair", s.Output)
		}
	}
}

func TestMutationAugmentBlocklistDrop(t *testing.T) {
	one := mutationFixture()[:1]
	rng := rand.New(rand.NewSource(mutationSeed))
	got, stats := mutationAugment(one, one, append([]ChatPair{}, one...), map[string]bool{}, rng)
	if stats.Added != 1 || len(got) != 2 {
		t.Fatalf("fixture must yield exactly one sibling, got stats %+v, %d pairs", stats, len(got))
	}
	sibling := got[1].Input
	// Rerun with that very phrasing blocklisted: the sibling must be
	// dropped, counted, and never emitted.
	block := map[string]bool{normAugInput(sibling): true}
	rng2 := rand.New(rand.NewSource(mutationSeed))
	got2, stats2 := mutationAugment(one, one, append([]ChatPair{}, one...), block, rng2)
	if stats2.DroppedBlocklist != 1 || stats2.Added != 0 {
		t.Fatalf("expected one blocklist drop, got %+v", stats2)
	}
	if len(got2) != 1 {
		t.Fatalf("blocklisted sibling leaked into training pairs: %v", got2)
	}
}

func TestMutationAugmentCollisionDrop(t *testing.T) {
	a := mutationFixture()[0]
	rng := rand.New(rand.NewSource(mutationSeed))
	got, _ := mutationAugment([]ChatPair{a}, []ChatPair{a}, []ChatPair{a}, map[string]bool{}, rng)
	if len(got) != 2 {
		t.Fatalf("fixture must yield one sibling, got %v", got)
	}
	sibling := got[1].Input
	// A second base pair already owns that phrasing with a different
	// answer: the sibling must be dropped as a label collision.
	b := ChatPair{Input: sibling, Output: "A completely different answer.", Domain: "social"}
	base := []ChatPair{a, b}
	rng2 := rand.New(rand.NewSource(mutationSeed))
	got2, stats2 := mutationAugment(base, []ChatPair{a}, append([]ChatPair{}, base...), map[string]bool{}, rng2)
	if stats2.DroppedCollision != 1 || stats2.Added != 0 {
		t.Fatalf("expected one collision drop, got %+v", stats2)
	}
	if len(got2) != len(base) {
		t.Fatalf("colliding sibling leaked: %v", got2)
	}
}

func TestLoadMutationBlocklist(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "block.jsonl")
	content := "{\"input\":\"Hey there how is it going\"}\n" +
		"{\"id\":\"S02c\",\"input\":\"good morning i hoe you slept okay\"}\n" +
		"a raw input line\n\n"
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	set, err := loadMutationBlocklist(path)
	if err != nil {
		t.Fatalf("loadMutationBlocklist: %v", err)
	}
	if len(set) != 3 {
		t.Fatalf("expected 3 blocklisted inputs, got %d: %v", len(set), set)
	}
	if !set[normAugInput("hey there how is it going")] || !set[normAugInput("a raw input line")] {
		t.Fatalf("blocklist contents wrong: %v", set)
	}
	if _, err := loadMutationBlocklist(""); err == nil {
		t.Fatal("empty path must fail closed")
	}
	if _, err := loadMutationBlocklist(filepath.Join(dir, "missing.jsonl")); err == nil {
		t.Fatal("missing file must error")
	}
}

// prepareFixtureRoot writes a tiny project root whose dataset holds
// 15 social pairs and 3 go pairs.
func prepareFixtureRoot(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	dir := filepath.Join(root, "data", "training")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	social := [][2]string{
		{"i am feeling really tired today", "Rest helps. Take it easy this evening."},
		{"the weather is lovely and warm", "Perfect for a walk outside."},
		{"i am going to the market later", "Picking up anything special?"},
		{"what are you doing this weekend", "Nothing big yet. How about you?"},
		{"i had a wonderful dinner with friends", "Good food and good company!"},
		{"music makes the evening feel better", "It really does set the mood."},
		{"i finished a long day at work", "Time to put your feet up."},
		{"do you want to watch a film tonight", "Sure, a film sounds lovely."},
		{"i love walking in the morning light", "Mornings are the best for it."},
		{"the garden looks beautiful in spring", "Everything is waking up out there."},
		{"i am learning to cook new dishes", "That is a tasty hobby to have."},
		{"how was your trip to the coast", "Tell me all about it!"},
		{"i need a quiet evening at home", "Quiet evenings recharge everything."},
		{"the children are playing in the yard", "Let them burn off that energy."},
		{"i am grateful for good friends", "They make all the difference."},
	}
	var b strings.Builder
	for _, p := range social {
		fmt.Fprintf(&b, "{\"input\":%q,\"output\":%q,\"domain\":\"social\"}\n", p[0], p[1])
	}
	for i := 0; i < 3; i++ {
		fmt.Fprintf(&b, "{\"input\":\"what is a goroutine %d\",\"output\":\"a lightweight thread %d\",\"domain\":\"go\"}\n", i, i)
	}
	if err := os.WriteFile(filepath.Join(dir, "chat_pairs.jsonl"), []byte(b.String()), 0o644); err != nil {
		t.Fatal(err)
	}
	return root
}

func TestPrepareMutationsOffByteIdentical(t *testing.T) {
	root := prepareFixtureRoot(t)
	legacy, err := prepareDomainTrainData(root, SocialDomain)
	if err != nil {
		t.Fatalf("legacy prepare: %v", err)
	}
	zero, err := prepareDomainTrainDataOpts(root, SocialDomain, RealTrainOptions{})
	if err != nil {
		t.Fatalf("zero-opts prepare: %v", err)
	}
	if !reflect.DeepEqual(legacy.trainPairs, zero.trainPairs) {
		t.Fatal("flag off: trainPairs differ from the legacy path")
	}
	if !reflect.DeepEqual(legacy.heldBase, zero.heldBase) {
		t.Fatal("flag off: heldBase differs from the legacy path")
	}
	if !reflect.DeepEqual(legacy.probes, zero.probes) {
		t.Fatal("flag off: probes differ from the legacy path")
	}
	if legacy.v.Size() != zero.v.Size() || vocabFingerprint(legacy.v) != vocabFingerprint(zero.v) {
		t.Fatal("flag off: vocabulary differs from the legacy path")
	}
}

func TestPrepareMutationsOn(t *testing.T) {
	root := prepareFixtureRoot(t)
	off, err := prepareDomainTrainData(root, SocialDomain)
	if err != nil {
		t.Fatalf("legacy prepare: %v", err)
	}
	// Fail closed: no blocklist, no augmentation.
	if _, err := prepareDomainTrainDataOpts(root, SocialDomain, RealTrainOptions{AugmentMutations: true}); err == nil {
		t.Fatal("augmentation without a blocklist must error")
	}
	// Social-only: another domain refuses the flag.
	blPath := filepath.Join(root, "block.jsonl")
	bl := "{\"input\":\"zz quantum bananas forever\"}\na raw blocklist entry line\n"
	if err := os.WriteFile(blPath, []byte(bl), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := prepareDomainTrainDataOpts(root, GoDomain, RealTrainOptions{AugmentMutations: true, MutationBlocklist: blPath}); err == nil {
		t.Fatal("augmentation on a non-social domain must error")
	}
	on, err := prepareDomainTrainDataOpts(root, SocialDomain, RealTrainOptions{AugmentMutations: true, MutationBlocklist: blPath})
	if err != nil {
		t.Fatalf("mutations-on prepare: %v", err)
	}
	if len(on.trainPairs) <= len(off.trainPairs) {
		t.Fatalf("expected siblings to grow the training pairs: on=%d off=%d", len(on.trainPairs), len(off.trainPairs))
	}
	// Siblings are a pure suffix: everything the legacy path produced
	// is untouched, in order, at the head.
	if !reflect.DeepEqual(on.trainPairs[:len(off.trainPairs)], off.trainPairs) {
		t.Fatal("mutations-on prepare altered the legacy prefix")
	}
	set, err := loadMutationBlocklist(blPath)
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range on.trainPairs[len(off.trainPairs):] {
		if set[normAugInput(p.Input)] {
			t.Fatalf("blocklisted input was trained on: %q", p.Input)
		}
	}
	t.Logf("train pairs: off=%d on=%d (siblings %d)", len(off.trainPairs), len(on.trainPairs), len(on.trainPairs)-len(off.trainPairs))
}
