package chat

import (
	"encoding/json"
	"fmt"
	"hash/fnv"
	"math"
	"math/rand"
	"os"
	"sort"
	"strings"
	"sync"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// The intent layer (2026-10-20, John's bar: "if I can't talk to it,
// what's the point"). A small softmax classifier trained on
// beginner phrasings (data/training/intent_pairs.jsonl) maps what
// someone types to one of the chat's actions, so nobody has to
// memorize commands. It is deliberately NOT the seq2seq model:
// choosing among ~35 actions with a name/link attached is a small
// output space a tiny model can actually learn — judged by a
// handwritten held-out set (intent_heldout.jsonl), never by the
// training templates.
//
// The classifier only ever ADDS understanding: below the confidence
// threshold the line falls through to the exact/sentence handlers
// and the router, exactly as before.

const (
	intentBuckets   = 8192
	intentMinConf   = 0.50
	intentMinMargin = 0.12
)

// IntentPair is one training example: a phrasing and its action.
type IntentPair struct {
	Input  string `json:"input"`
	Intent string `json:"intent"`
}

// IntentModel is the trained classifier: one weight vector per
// intent over hashed n-gram features. JSON on disk
// (data/intent_model.json) so it stays a readable artifact.
type IntentModel struct {
	Intents []string    `json:"intents"`
	Buckets int         `json:"buckets"`
	Weights [][]float32 `json:"weights"`
	Bias    []float32   `json:"bias"`
}

// intentTokens normalizes a line into word tokens for features.
func intentTokens(line string) []string {
	line = strings.ToLower(strings.TrimSpace(line))
	var b strings.Builder
	for _, r := range line {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '/' || r == '.' || r == '_' || r == '-' {
			b.WriteRune(r)
		} else {
			b.WriteRune(' ')
		}
	}
	return strings.Fields(b.String())
}

// intentFeatures extracts the sparse feature set: word unigrams and
// bigrams (intent lives in word choice), char trigrams per word
// (typo tolerance — "ture" is close to "tour"), and structural
// flags (a URL is present, the line ends in a question, ...).
func intentFeatures(line string) map[int]float32 {
	toks := intentTokens(line)
	feats := map[string]float32{}
	add := func(f string) { feats[f]++ }
	for i, t := range toks {
		add("w:" + t)
		if i+1 < len(toks) {
			add("b:" + t + " " + toks[i+1])
		}
		p := "^" + t + "$"
		for j := 0; j+3 <= len(p); j++ {
			add("c:" + p[j:j+3])
		}
	}
	if githubURL.MatchString(line) {
		feats["F:url"] = 1
	}
	if strings.HasPrefix(strings.TrimSpace(line), "/") {
		feats["F:slash"] = 1
	}
	if strings.HasSuffix(strings.TrimSpace(line), "?") {
		feats["F:question"] = 1
	}
	if strings.Contains(line, "make ") {
		feats["F:make"] = 1
	}
	out := make(map[int]float32, len(feats))
	for f, v := range feats {
		h := fnv.New32a()
		_, _ = h.Write([]byte(f))
		out[int(h.Sum32()%intentBuckets)] += float32(math.Sqrt(float64(v)))
	}
	return out
}

// TrainIntentModel fits the classifier with softmax SGD. Fast by
// construction: a few thousand short examples, sparse updates.
func TrainIntentModel(pairs []IntentPair) *IntentModel {
	idx := map[string]int{}
	var intents []string
	for _, p := range pairs {
		if _, ok := idx[p.Intent]; !ok {
			idx[p.Intent] = len(intents)
			intents = append(intents, p.Intent)
		}
	}
	sort.Strings(intents)
	for i, name := range intents {
		idx[name] = i
	}
	m := &IntentModel{Intents: intents, Buckets: intentBuckets}
	m.Weights = make([][]float32, len(intents))
	for i := range m.Weights {
		m.Weights[i] = make([]float32, intentBuckets)
	}
	m.Bias = make([]float32, len(intents))

	type ex struct {
		feats map[int]float32
		y     int
	}
	exs := make([]ex, len(pairs))
	for i, p := range pairs {
		exs[i] = ex{intentFeatures(p.Input), idx[p.Intent]}
	}
	rng := rand.New(rand.NewSource(42))
	const epochs = 30
	for ep := 0; ep < epochs; ep++ {
		lr := float32(0.6 * (1 - float64(ep)/epochs) + 0.05)
		rng.Shuffle(len(exs), func(a, b int) { exs[a], exs[b] = exs[b], exs[a] })
		for _, e := range exs {
			// forward
			scores := make([]float32, len(intents))
			for k := range intents {
				s := m.Bias[k]
				for f, v := range e.feats {
					s += m.Weights[k][f] * v
				}
				scores[k] = s
			}
			max := scores[0]
			for _, s := range scores {
				if s > max {
					max = s
				}
			}
			var sum float32
			probs := make([]float32, len(intents))
			for k, s := range scores {
				probs[k] = float32(math.Exp(float64(s - max)))
				sum += probs[k]
			}
			// backward: gradient = prob - onehot
			for k := range intents {
				g := probs[k] / sum
				if k == e.y {
					g -= 1
				}
				if g == 0 {
					continue
				}
				for f, v := range e.feats {
					m.Weights[k][f] -= lr * g * v
				}
				m.Bias[k] -= lr * g * 0.1
			}
		}
	}
	return m
}

// Predict returns the best intent, its softmax confidence, and the
// margin over the runner-up.
func (m *IntentModel) Predict(line string) (string, float64, float64) {
	feats := intentFeatures(line)
	scores := make([]float64, len(m.Intents))
	for k := range m.Intents {
		s := float64(m.Bias[k])
		for f, v := range feats {
			s += float64(m.Weights[k][f]) * float64(v)
		}
		scores[k] = s
	}
	max := scores[0]
	for _, s := range scores {
		if s > max {
			max = s
		}
	}
	var sum float64
	probs := make([]float64, len(m.Intents))
	for k, s := range scores {
		probs[k] = math.Exp(s - max)
		sum += probs[k]
	}
	best, second := -1, -1.0
	bestP := -1.0
	for k, p := range probs {
		p /= sum
		probs[k] = p
		if p > bestP {
			second = bestP
			best, bestP = k, p
		} else if p > second {
			second = p
		}
	}
	if second < 0 {
		second = 0
	}
	return m.Intents[best], bestP, bestP - second
}

// ClassifyIntent is Predict gated by the confidence/margin bar:
// below it the caller must fall through to the older handlers.
func (m *IntentModel) ClassifyIntent(line string) (string, float64, bool) {
	intent, conf, margin := m.Predict(line)
	if conf < intentMinConf || margin < intentMinMargin {
		return "", conf, false
	}
	return intent, conf, true
}

// Save writes the model as JSON.
func (m *IntentModel) Save(path string) error {
	data, err := json.Marshal(m)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

// LoadIntentModel reads a trained model from JSON.
func LoadIntentModel(path string) (*IntentModel, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var m IntentModel
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	if len(m.Intents) == 0 || len(m.Weights) != len(m.Intents) {
		return nil, fmt.Errorf("intent model at %s is malformed", path)
	}
	return &m, nil
}

// LoadIntentPairs reads a JSONL corpus.
func LoadIntentPairs(path string) ([]IntentPair, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var out []IntentPair
	for _, l := range strings.Split(string(data), "\n") {
		l = strings.TrimSpace(l)
		if l == "" {
			continue
		}
		var p IntentPair
		if err := json.Unmarshal([]byte(l), &p); err == nil && p.Intent != "" {
			out = append(out, p)
		}
	}
	return out, nil
}

// --- runtime loading ------------------------------------------------------

var (
	intentModelOnce sync.Once
	intentModel     *IntentModel
)

// defaultIntentModel lazily loads data/intent_model.json from the
// chat's project root (or the working directory). Absent file = the
// layer is off and everything behaves as before.
func defaultIntentModel() *IntentModel {
	intentModelOnce.Do(func() {
		for _, base := range []string{unifiedProjectRoot, startFolder()} {
			if base == "" {
				continue
			}
			if m, err := LoadIntentModel(base + "/data/intent_model.json"); err == nil {
				intentModel = m
				return
			}
		}
	})
	return intentModel
}

// --- slots ----------------------------------------------------------------

// slotName guesses the function/type name a line refers to: a token
// the analyzed project actually has wins; else the last capitalized
// token; else the token after of/for/through.
func slotName(line string) string {
	toks := intentTokens(line)
	if lastAnalyzeProject != nil {
		for _, t := range toks {
			if findFuncByQuery(lastAnalyzeProject, t) != nil {
				return t
			}
		}
	}
	orig := strings.Fields(line)
	for i := len(orig) - 1; i >= 0; i-- {
		w := strings.Trim(orig[i], ".,?!'\"")
		if len(w) > 1 && w[0] >= 'A' && w[0] <= 'Z' {
			return w
		}
	}
	for i, t := range toks {
		if (t == "of" || t == "for" || t == "through") && i+1 < len(toks) {
			return toks[i+1]
		}
	}
	return ""
}

// slotAfterMake returns the make target named in the line, if any.
func slotAfterMake(line string) string {
	toks := intentTokens(line)
	for i, t := range toks {
		if t == "make" && i+1 < len(toks) {
			return toks[i+1]
		}
	}
	return ""
}

// --- dispatch -------------------------------------------------------------

// canonicalFor renders a beginner-five intent as the canonical line
// the beginner dispatcher already understands, carrying the slot
// (project ref / function name) out of the original phrasing.
func canonicalFor(intent, line string) string {
	switch intent {
	case "tour":
		if ref := extractAnalyzePath(line); ref != "" {
			return "/tour " + ref
		}
		return "/tour"
	case "glossary":
		return "/glossary"
	case "learn_path":
		return "/learn-path"
	case "watch":
		return "/watch"
	case "example":
		if name := slotName(line); name != "" {
			return "/example " + name
		}
	}
	return ""
}

// tryIntentDispatch is the intent layer's front door, called after
// the exact/sentence handlers and before routing. It only claims a
// line when the trained classifier is confident; everything else
// falls through untouched. Brain-owned intents (concept questions,
// code writing, social, make commands) are deliberately NOT
// dispatched — the router and brains keep those.
// slotFuncNames pulls up to max distinct project functions named in the
// line, in order — the Comparer's two subjects.
func slotFuncNames(line string, proj *analyze.Project, max int) []*analyze.Func {
	var out []*analyze.Func
	seen := map[string]bool{}
	for _, tok := range strings.Fields(line) {
		name := strings.Trim(tok, ".,!?;:'\u201d()")
		if name == "" {
			continue
		}
		if fn := findFuncByQuery(proj, name); fn != nil && !seen[fn.ID] {
			seen[fn.ID] = true
			out = append(out, fn)
			if len(out) >= max {
				break
			}
		}
	}
	return out
}

func tryIntentDispatch(line string, profile *UserProfile) bool {
	m := defaultIntentModel()
	if m == nil {
		return false
	}
	// Never reinterpret an error paste; the dedicated translator
	// owns lines that look like compiler output.
	if strings.Contains(line, ".go:") {
		return false
	}
	intent, _, ok := m.ClassifyIntent(line)
	if !ok {
		return false
	}
	friendObserveIntent(intent)
	switch intent {
	case "tour", "glossary", "learn_path", "example", "watch":
		if canon := canonicalFor(intent, line); canon != "" {
			return handleBeginnerCommand(canon, profile)
		}
		if intent == "example" && ensureProject() {
			if tf := topFuncName(lastAnalyzeProject); tf != "" {
				return handleBeginnerCommand("/example "+tf, profile)
			}
		}
		return false
	case "brains":
		low := strings.ToLower(line)
		if strings.Contains(low, "doing") || strings.Contains(low, "status") || strings.Contains(low, "working") || strings.Contains(low, "score") {
			return handleBeginnerCommand("/brainstatus", profile)
		}
		return handleBeginnerCommand("/brains", profile)
	case "why_brain":
		return handleBeginnerCommand("/whybrain", profile)
	case "suggest":
		friendReport()
		recordBrain("friend", true, "asked directly")
		return true
	case "tell_more":
		return handleBeginnerCommand("/tell-more", profile)
	case "what_now":
		return handleBeginnerCommand("/what-now", profile)
	case "summarize":
		return handleBeginnerCommand("/summarize", profile)
	case "help":
		printHelp()
		return true
	case "analyze":
		ref := extractAnalyzePath(line)
		if ref == "" {
			return false
		}
		if out, ok := handleGoAnalyze("analyze " + ref); ok {
			fmt.Println(out)
			return true
		}
		return false
	case "where":
		// Issue URLs keep their dedicated loop flow.
		if githubURL.MatchString(line) {
			return false
		}
		if out, ok := handleGoAnalyze(line); ok {
			fmt.Println(out)
			return true
		}
		return false
	case "flow":
		if out, ok := handleFlowCommand(line); ok {
			fmt.Println(out)
			return true
		}
		return false
	}
	// Everything below answers from the analyzed project.
	switch intent {
	case "whatis", "show_func", "walkthrough":
		name := slotName(line)
		if name == "" {
			return false
		}
		if !ensureProject() {
			return true
		}
		q := map[string]string{
			"whatis":      "what does " + name + " do",
			"show_func":   "show me " + name,
			"walkthrough": "walk me through " + name,
		}[intent]
		if out, ok := lastAnalyzeProject.Answer(q); ok {
			fmt.Println(out)
			recordBrain("analyst", true, "intent "+intent)
			friendNoteFunc(name)
			return true
		}
		return false
	case "review", "comment", "test_scaffold", "uexplain":
		name := slotName(line)
		if name == "" {
			return false
		}
		if !ensureProject() {
			return true
		}
		lastActionKind, lastSubject = intent, name
		fn := findFuncByQuery(lastAnalyzeProject, name)
		if fn != nil {
			friendNoteFunc(name)
		}
		if fn == nil {
			fmt.Printf("Function %q not found.", name)
			if near := nearFuncNames(lastAnalyzeProject, name); len(near) > 0 {
				fmt.Printf(" Close matches: %s.", strings.Join(near, ", "))
			}
			fmt.Println()
			return true
		}
		switch intent {
		case "review":
			fmt.Println(reviewFunc(name, lastAnalyzeProject))
			recordBrain("critic", true, "intent review")
		case "comment":
			out, _ := generateDocCommentCmd(name, lastAnalyzeProject, profile)
			fmt.Println(out)
			recordBrain("composer", true, "intent comment")
		case "test_scaffold":
			fmt.Println(generateTestSkeleton(name, lastAnalyzeProject))
			recordBrain("composer", true, "intent test_scaffold")
		case "uexplain":
			runUexplainPrint(fn, startFolder())
		}
		return true
	case "why_func":
		name := slotName(line)
		if name == "" {
			return false
		}
		if !ensureProject() {
			return true
		}
		fn := findFuncByQuery(lastAnalyzeProject, name)
		if fn == nil {
			fmt.Printf("Function %q not found in this project.\n", name)
			recordBrain("why", false, "not found: "+name)
			return true
		}
		fmt.Println(lastAnalyzeProject.WhyText(fn))
		recordBrain("why", true, "intent why_func")
		friendNoteFunc(name)
		lastActionKind, lastSubject = "why", name
		return true
	case "compare":
		if !ensureProject() {
			return true
		}
		fns := slotFuncNames(line, lastAnalyzeProject, 2)
		if len(fns) < 2 {
			fmt.Println("Name two functions and the Comparer will line them up — e.g. \"compare Add and Get\".")
			recordBrain("comparer", false, "needed two function names")
			return true
		}
		fmt.Println(lastAnalyzeProject.CompareText(fns[0], fns[1]))
		recordBrain("comparer", true, "intent compare")
		lastActionKind, lastSubject = "compare", fns[0].Name
		return true
	case "arch":
		if !ensureProject() {
			return true
		}
		fmt.Println(describeArchitecture(lastAnalyzeProject))
		recordBrain("analyst", true, "intent arch")
		return true
	case "why_import":
		pkg := slotPkg(line)
		if pkg == "" {
			return false
		}
		if !ensureProject() {
			return true
		}
		fmt.Println(explainImport(pkg, lastAnalyzeProject))
		recordBrain("analyst", true, "intent why_import")
		return true
	case "fix":
		if !ensureProject() {
			return true
		}
		fmt.Println(suggestErrorFix(line, lastAnalyzeProject))
		return true
	}
	return false
}

// slotPkg guesses the imported package a line asks about: the token
// after "import"/"uses", else any token the project actually imports.
func slotPkg(line string) string {
	toks := intentTokens(line)
	for i, t := range toks {
		if (t == "import" || t == "uses" || t == "use") && i+1 < len(toks) {
			return toks[i+1]
		}
	}
	return ""
}
