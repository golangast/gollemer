package decl

import (
	"encoding/json"
	"fmt"
	"hash/fnv"
	"math/rand"
	"sort"
	"strings"
)

// Generation: specs are sampled combinatorially from the pools
// (kind x name x member counts x types x modifiers), validated by
// the filter, and partitioned by COMBINATION: a combination is
// (kind, decl name, multiset of member types, receiver shape). The
// combination space is split by a seeded hash — a combination whose
// hash lands in the held-out bucket NEVER appears in training, in any
// member naming or order. Phrasing frames are split independently:
// reserved frames never phrase a training pair. The eval matrix is
// the product of the two axes x the two input forms.

// Pair is one training pair (chat_pairs.jsonl shape).
type Pair struct {
	Input  string `json:"input"`
	Output string `json:"output"`
	Domain string `json:"domain"`
}

// EvalItem is one held-out evaluation case.
type EvalItem struct {
	ID        string `json:"id"`
	InputForm string `json:"input_form"` // "spec" | "phrasing"
	Combo     string `json:"combo"`      // "seen" | "unseen"
	Frame     string `json:"frame"`      // "seen" | "unseen" | "" for spec form
	FrameID   string `json:"frame_id,omitempty"`
	Input     string `json:"input"`
	Spec      Spec   `json:"spec"`
	Decl      string `json:"decl"`
}

// GenConfig sizes a generation run.
type GenConfig struct {
	Seed             int64
	TrainQuota       map[Kind]int
	HeldoutQuota     map[Kind]int
	UnseenFrameItems int // heldout specs additionally phrased by a reserved frame
	SeenComboItems   int // train specs phrased by a reserved frame (eval only)
	SanityItems      int // train pairs replayed as the seen/seen cell
}

// DefaultGenConfig is the experiment's sizing: ~3,700 train specs
// (~11k pairs), 620 held-out specs, ~2,000 eval items.
func DefaultGenConfig() GenConfig {
	return GenConfig{
		Seed:             20261026,
		TrainQuota:       map[Kind]int{KindStruct: 1400, KindInterface: 1000, KindFunc: 1300},
		HeldoutQuota:     map[Kind]int{KindStruct: 240, KindInterface: 180, KindFunc: 200},
		UnseenFrameItems: 300,
		SeenComboItems:   300,
		SanityItems:      150,
	}
}

// Dataset is the full generation output.
type Dataset struct {
	TrainSpecs   []Spec
	HeldoutSpecs []Spec
	TrainPairs   []Pair
	EvalItems    []EvalItem
	Stats        *FilterStats
}

// Generate runs the whole pipeline: sample, filter, split, phrase,
// assemble. It errors on coverage violations (a pool atom or frame
// word the held-out items need but training never shows) rather than
// emitting a quietly compromised dataset.
func Generate(cfg GenConfig) (*Dataset, error) {
	if err := AssertPoolInjectivity(); err != nil {
		return nil, err
	}
	rng := rand.New(rand.NewSource(cfg.Seed))
	stats := NewFilterStats()
	s := &sampler{rng: rng}

	ds := &Dataset{}
	trainKeys := map[string]bool{}
	heldKeys := map[string]bool{}
	trainCount := map[Kind]int{}
	heldCount := map[Kind]int{}

	kinds := []Kind{KindStruct, KindInterface, KindFunc}
	maxAttempts := 0
	for _, k := range kinds {
		maxAttempts += (cfg.TrainQuota[k] + cfg.HeldoutQuota[k]) * 60
	}
	attempts := 0
	for attempts < maxAttempts {
		attempts++
		kind := kinds[rng.Intn(len(kinds))]
		needTrain := trainCount[kind] < cfg.TrainQuota[kind]
		needHeld := heldCount[kind] < cfg.HeldoutQuota[kind]
		if !needTrain && !needHeld {
			done := true
			for _, k := range kinds {
				if trainCount[k] < cfg.TrainQuota[k] || heldCount[k] < cfg.HeldoutQuota[k] {
					done = false
				}
			}
			if done {
				break
			}
			continue
		}
		spec := s.spec(kind)
		if !Validate(spec, stats) {
			continue
		}
		canon, err := Canonicalize(spec)
		if err != nil {
			continue
		}
		fullKey := specKey(canon)
		if heldoutCombo(canon) {
			if !needHeld || heldKeys[fullKey] {
				continue
			}
			heldKeys[fullKey] = true
			heldCount[kind]++
			ds.HeldoutSpecs = append(ds.HeldoutSpecs, canon)
		} else {
			if !needTrain || trainKeys[fullKey] {
				continue
			}
			trainKeys[fullKey] = true
			trainCount[kind]++
			ds.TrainSpecs = append(ds.TrainSpecs, canon)
		}
	}
	for _, k := range kinds {
		if trainCount[k] < cfg.TrainQuota[k] || heldCount[k] < cfg.HeldoutQuota[k] {
			return nil, fmt.Errorf("generation underfilled kind %s: train %d/%d heldout %d/%d after %d attempts",
				k, trainCount[k], cfg.TrainQuota[k], heldCount[k], cfg.HeldoutQuota[k], attempts)
		}
	}

	// Coverage patch: every pool atom must appear in some TRAIN spec,
	// or held-out items using it would face an unknown token.
	patchCoverage(ds, s, stats, trainKeys)

	// Assemble training pairs: one spec-form pair + two phrasing
	// pairs (distinct seen frames) per train spec.
	for _, spec := range ds.TrainSpecs {
		target := strings.ToLower(RenderDecl(spec))
		ds.TrainPairs = append(ds.TrainPairs, Pair{Input: SpecText(spec), Output: target, Domain: "decl"})
		frames := FramesFor(spec, false)
		if len(frames) >= 2 {
			i := rng.Intn(len(frames))
			j := rng.Intn(len(frames) - 1)
			if j >= i {
				j++
			}
			for _, fi := range []int{i, j} {
				ds.TrainPairs = append(ds.TrainPairs, Pair{Input: Fill(frames[fi], spec), Output: target, Domain: "decl"})
			}
		}
	}
	rng.Shuffle(len(ds.TrainPairs), func(i, j int) {
		ds.TrainPairs[i], ds.TrainPairs[j] = ds.TrainPairs[j], ds.TrainPairs[i]
	})

	ds.EvalItems = buildEvalItems(ds, rng, cfg)
	ds.Stats = stats.Snapshot()

	if violations := CoverageViolations(ds); len(violations) > 0 {
		return nil, fmt.Errorf("coverage violations: %s", strings.Join(violations, "; "))
	}
	return ds, nil
}

// specKey is the full-spec identity (dedup within a partition).
func specKey(s Spec) string {
	raw, _ := json.Marshal(s)
	return string(raw)
}

// comboKey is the combination identity: kind, decl name, the sorted
// multiset of member type expressions (with embed/signature shape),
// and the receiver shape. Member NAMES are not part of it — a held-out
// combination is unseen in every naming.
func comboKey(s Spec) string {
	var members []string
	for _, f := range s.Fields {
		if f.Embedded {
			members = append(members, "e:"+f.Type.Expr)
		} else {
			members = append(members, "f:"+f.Type.Expr)
		}
	}
	for _, e := range s.Embeds {
		members = append(members, "e:"+e.Expr)
	}
	for _, m := range s.Methods {
		members = append(members, "m:"+sigTypes(m.Params, m.Results))
	}
	if s.Kind == KindFunc {
		members = append(members, "fn:"+sigTypes(s.Params, s.Results))
		if s.Receiver != nil {
			members = append(members, "recv:"+s.Receiver.Type+":"+fmt.Sprint(s.Receiver.Pointer))
		}
	}
	sort.Strings(members)
	return string(s.Kind) + "|" + s.Name + "|" + strings.Join(members, "|")
}

func sigTypes(params []Param, results []TypeRef) string {
	var ps, rs []string
	for _, p := range params {
		ps = append(ps, p.Type.Expr)
	}
	for _, r := range results {
		rs = append(rs, r.Expr)
	}
	return "(" + strings.Join(ps, ",") + ")->(" + strings.Join(rs, ",") + ")"
}

// heldoutCombo assigns a combination to the held-out partition by
// seeded hash (bucket 0 of 5, ~20% of the combination space).
func heldoutCombo(s Spec) bool {
	h := fnv.New64a()
	h.Write([]byte(comboKey(s) + "|declsplit-v1"))
	return h.Sum64()%5 == 0
}

// buildEvalItems assembles the eval matrix from the split.
func buildEvalItems(ds *Dataset, rng *rand.Rand, cfg GenConfig) []EvalItem {
	var items []EvalItem
	n := 0
	add := func(form, combo, frame, frameID, input string, spec Spec) {
		n++
		items = append(items, EvalItem{
			ID: fmt.Sprintf("decl-eval-%04d", n), InputForm: form, Combo: combo,
			Frame: frame, FrameID: frameID, Input: input, Spec: spec,
			Decl: RenderDecl(spec),
		})
	}
	// Unseen combos: spec form + seen-frame phrasing for every
	// held-out spec; a reserved-frame phrasing for a subset.
	for i, spec := range ds.HeldoutSpecs {
		add("spec", "unseen", "", "", SpecText(spec), spec)
		if frames := FramesFor(spec, false); len(frames) > 0 {
			f := frames[rng.Intn(len(frames))]
			add("phrasing", "unseen", "seen", f.ID, Fill(f, spec), spec)
		}
		if i < cfg.UnseenFrameItems {
			if frames := FramesFor(spec, true); len(frames) > 0 {
				f := frames[rng.Intn(len(frames))]
				add("phrasing", "unseen", "unseen", f.ID, Fill(f, spec), spec)
			}
		}
	}
	// Seen combos x unseen frames: train specs phrased by reserved
	// frames (eval-only items; the phrasings are never trained).
	for i := 0; i < cfg.SeenComboItems && i < len(ds.TrainSpecs); i++ {
		spec := ds.TrainSpecs[rng.Intn(len(ds.TrainSpecs))]
		if frames := FramesFor(spec, true); len(frames) > 0 {
			f := frames[rng.Intn(len(frames))]
			add("phrasing", "seen", "unseen", f.ID, Fill(f, spec), spec)
		}
	}
	// Seen x seen sanity: actual train pairs replayed.
	for i := 0; i < cfg.SanityItems && i < len(ds.TrainPairs); i++ {
		p := ds.TrainPairs[rng.Intn(len(ds.TrainPairs))]
		form := "phrasing"
		if strings.HasPrefix(p.Input, "spec ") {
			form = "spec"
		}
		// Recover the spec for scoring: find it by rendered target.
		for _, spec := range ds.TrainSpecs {
			if strings.ToLower(RenderDecl(spec)) == p.Output {
				add(form, "seen", "seen", "", p.Input, spec)
				break
			}
		}
	}
	return items
}

// patchCoverage adds simple train specs until every pool atom — and
// every fused ";" token the held-out targets need — appears in the
// training partition (subject to the combination split and the
// filter, like any other spec). Without it, a held-out item could
// hinge on a token the model has never seen in training, and the eval
// would measure vocabulary gaps instead of generalization.
func patchCoverage(ds *Dataset, s *sampler, stats *FilterStats, trainKeys map[string]bool) {
	usedNames := map[string]bool{} // decl names, field names, verb names
	usedTypes := map[string]bool{} // local type names referenced anywhere
	usedSemi := map[string]bool{}  // fused "token;" forms in rendered train targets
	mark := func(spec Spec) {
		usedNames[spec.Name] = true
		for _, ref := range allTypeRefs(spec) {
			for _, ln := range LocalNames(ref.Expr) {
				usedTypes[ln] = true
			}
		}
		for _, f := range spec.Fields {
			if !f.Embedded {
				usedNames[f.Name] = true
			}
		}
		for _, m := range spec.Methods {
			usedNames[m.Name] = true
		}
		for _, tok := range SplitTokens(RenderDecl(spec)) {
			if strings.HasSuffix(tok, ";") {
				usedSemi[tok] = true
			}
		}
	}
	for _, spec := range ds.TrainSpecs {
		mark(spec)
	}
	add := func(spec Spec) bool {
		canon, err := Canonicalize(spec)
		if err != nil || heldoutCombo(canon) || trainKeys[specKey(canon)] || !Validate(canon, stats) {
			return false
		}
		trainKeys[specKey(canon)] = true
		ds.TrainSpecs = append(ds.TrainSpecs, canon)
		mark(canon)
		return true
	}

	// Local types as decl names and as referenced types.
	for _, name := range LocalTypes {
		if usedNames[name] && usedTypes[name] {
			continue
		}
		for try := 0; try < 40 && !(usedNames[name] && usedTypes[name]); try++ {
			field := FieldNames[s.rng.Intn(len(FieldNames))]
			spec := Spec{Kind: KindStruct, Name: name, Fields: []Field{
				{Name: field, Type: MustTypeRef(name)},
			}}
			add(spec)
		}
	}
	// Field names.
	for _, fname := range FieldNames {
		if usedNames[fname] {
			continue
		}
		for try := 0; try < 40 && !usedNames[fname]; try++ {
			spec := Spec{Kind: KindStruct, Name: s.pick(LocalTypes), Fields: []Field{
				{Name: fname, Type: MustTypeRef("string")},
				{Name: s.pick(FieldNames), Type: MustTypeRef("int")},
			}}
			if spec.Fields[1].Name != fname {
				add(spec)
			}
		}
	}
	// Verb names as functions, and constructors for every local type.
	for _, verb := range VerbNames {
		if usedNames[verb] {
			continue
		}
		for try := 0; try < 40 && !usedNames[verb]; try++ {
			add(Spec{Kind: KindFunc, Name: verb, Params: s.funcParams(2), Results: s.resultList()})
		}
	}
	for _, t := range LocalTypes {
		ctor := "New" + t
		if usedNames[ctor] {
			continue
		}
		for try := 0; try < 40 && !usedNames[ctor]; try++ {
			star := "*"
			if s.rng.Intn(4) == 0 {
				star = ""
			}
			add(Spec{Kind: KindFunc, Name: ctor, Params: s.funcParams(1),
				Results: []TypeRef{MustTypeRef(star + t)}})
		}
	}
	// Fused ";" tokens the held-out targets will need: a held-out
	// field/method whose type ends in word token w renders the token
	// "w;" — some train target must contain it too.
	neededSemi := map[string]bool{}
	for _, spec := range ds.HeldoutSpecs {
		for _, tok := range SplitTokens(RenderDecl(spec)) {
			if strings.HasSuffix(tok, ";") && len(tok) > 1 {
				neededSemi[tok] = true
			}
		}
	}
	for tok := range neededSemi {
		if usedSemi[tok] {
			continue
		}
		base := strings.TrimSuffix(tok, ";")
		// Only simple type tokens fuse with ";" (composites end in
		// punctuation, which leaves ";" as its own token).
		if _, err := MakeTypeRef(base); err != nil {
			continue
		}
		ref, err := MakeTypeRef(base)
		if err != nil || ref.Class == ClassComposite {
			continue
		}
		for try := 0; try < 40 && !usedSemi[tok]; try++ {
			spec := Spec{Kind: KindStruct, Name: s.pick(LocalTypes), Fields: []Field{
				{Name: s.pick(FieldNames), Type: ref},
				{Name: s.pick(FieldNames), Type: MustTypeRef("string")},
			}}
			if spec.Fields[0].Name != spec.Fields[1].Name {
				add(spec)
			}
		}
	}
}

// CoverageViolations lists held-out tokens (in eval inputs AND in the
// targets the model must produce) that never appear in any training
// pair — tokens the model cannot have learned. Pool atoms are covered
// by patchCoverage; frame words must be shared between seen and
// reserved frames by construction.
func CoverageViolations(ds *Dataset) []string {
	trainWords := map[string]bool{}
	for _, p := range ds.TrainPairs {
		for _, t := range SplitTokens(p.Input) {
			trainWords[t] = true
		}
		for _, t := range SplitTokens(p.Output) {
			trainWords[t] = true
		}
	}
	seen := map[string]bool{}
	var out []string
	check := func(text, where string) {
		for _, t := range SplitTokens(text) {
			if !trainWords[t] && !seen[t] {
				seen[t] = true
				out = append(out, fmt.Sprintf("token %q in %s", t, where))
			}
		}
	}
	for _, item := range ds.EvalItems {
		check(item.Input, "input of "+item.ID)
		check(item.Decl, "target of "+item.ID)
	}
	sort.Strings(out)
	return out
}
