package chat

import (
	"encoding/json"
	"fmt"
	"hash/fnv"
	"math"
	"math/rand"
	"os"
	"regexp"
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
	best, bestConf, _, secondConf := m.predictTop2(line)
	return best, bestConf, bestConf - secondConf
}

// predictTop2 returns the two most likely intents with their
// softmax probabilities — the clarification fallback needs to know
// WHO the runner-up is, not just the margin.
func (m *IntentModel) predictTop2(line string) (string, float64, string, float64) {
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
	best, second := -1, -1
	for k := range probs {
		probs[k] /= sum
		if best < 0 || probs[k] > probs[best] {
			second = best
			best = k
		} else if second < 0 || probs[k] > probs[second] {
			second = k
		}
	}
	if second < 0 {
		return m.Intents[best], probs[best], "", 0
	}
	return m.Intents[best], probs[best], m.Intents[second], probs[second]
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

// dataNameRes pull the variable name out of a dataflow question
// when the project-token scan in slotVarName cannot (qualified
// names, role-word suffixes — DataflowText cleans the capture).
var dataNameRes = []*regexp.Regexp{
	regexp.MustCompile(`(?i)where does (?:the )?(.+?) (?:get|come|change)`),
	regexp.MustCompile(`(?i)where is (?:the )?(.+?) (?:changed|set|updated)`),
	regexp.MustCompile(`(?i)what (?:changes|updates|modifies|mutates|sets|writes to) (?:the )?(.+?)\s*\??$`),
	regexp.MustCompile(`(?i)who (?:changes|writes to) (?:the )?(.+?)\s*\??$`),
}

// slotVarName pulls the variable/field name out of a dataflow
// question: a token the analyzed project actually tracks (bare or
// dotted) wins; else the name captured by the question shapes.
func slotVarName(line string) string {
	if lastAnalyzeProject != nil {
		for _, tok := range intentTokens(line) {
			if len(tok) < 2 && !strings.Contains(tok, ".") {
				continue
			}
			if lastAnalyzeProject.HasData(tok) {
				return tok
			}
		}
	}
	for _, re := range dataNameRes {
		if m := re.FindStringSubmatch(line); m != nil {
			return strings.TrimSpace(m[1])
		}
	}
	return ""
}

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

// tryIntentDispatch runs the intent layer proper
// (tryIntentDispatchInner) and then the repo-URL floor: if nothing
// claimed a line that names a whole GitHub repo, the floor answers
// it with that repo's analysis instead of letting the line fall on
// to the router, where the social brain could greet it. Every chat
// loop enters through here, so the floor binds in all of them.
func tryIntentDispatch(line string, profile *UserProfile) bool {
	if tryIntentDispatchInner(line, profile) {
		return true
	}
	return tryRepoURLFloor(line)
}

func tryIntentDispatchInner(line string, profile *UserProfile) bool {
	// A pending "did you mean…?" gets first claim on the reply —
	// here, after the beginner commands and the edits parser have
	// had their turn, the same place the question was asked from.
	if done, handled := resolveClarification(line, profile); done {
		return handled
	}
	// Exact symbol references are deterministic too: a pending
	// pick-list's numeric reply, or a dotted name pasted back from a
	// candidate list, must never be left for the classifier to
	// reinterpret (it once read the bare dotted name as a find
	// search and answered "No matches" for the system's own
	// suggestion). See symbolpick.go.
	if done, handled := resolveFuncPick(line); done {
		return handled
	}
	// A question about a repo URL is deterministic too: the line
	// names the exact project to read, so it is claimed before
	// classification. The classifier otherwise reads "what does
	// <url> do" as a summarize of the folder the chat started in
	// (tryNaturalAnalyze sits behind it and never sees the line).
	// See clone.go's repoQuestionURL for what counts as a repo
	// question — issue/PR/deep links are never claimed here.
	if tryRepoURLDispatch(line) {
		return true
	}
	if tryDottedSymbol(line, profile) {
		return true
	}
	m := defaultIntentModel()
	if m == nil {
		return false
	}
	// Never reinterpret an error paste; the dedicated translator
	// owns lines that look like compiler output.
	if strings.Contains(line, ".go:") {
		return false
	}
	intent, conf, ok := m.ClassifyIntent(line)
	if !ok {
		if offerClarification(m, line) {
			return true
		}
		// The layer passed on this line: record the near miss so
		// the outcome log holds the whole picture (a fell-through
		// line is never a training pair — nothing confirmed it).
		top, topConf, _, _ := m.predictTop2(line)
		recordIntentOutcome(line, top, topConf, IntentOutcomeFellThrough)
		return false
	}
	friendObserveIntent(intent)
	handled := dispatchIntent(intent, line, profile)
	if handled {
		recordIntentOutcome(line, intent, conf, IntentOutcomeDispatched)
	} else {
		// Confident, but not the intent layer's to claim
		// (brain-owned intents land here): it fell through.
		recordIntentOutcome(line, intent, conf, IntentOutcomeFellThrough)
	}
	return handled
}

// dispatchIntent runs one confidently-classified intent — and, via
// the clarification fallback, one the user just confirmed. Split
// out of tryIntentDispatch so a confirmed guess routes through the
// exact same cases as a confident one.
func dispatchIntent(intent, line string, profile *UserProfile) bool {
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
	case "start_guide":
		return handleBeginnerCommand("/start", profile)
	case "tell_more":
		return handleBeginnerCommand("/tell-more", profile)
	case "what_now":
		return handleBeginnerCommand("/what-now", profile)
	case "summarize":
		return handleBeginnerCommand("/summarize", profile)
	case "dataflow_q":
		if !ensureProject() {
			return true
		}
		name := slotVarName(line)
		if name == "" {
			return false
		}
		if out, ok := lastAnalyzeProject.DataflowText(name); ok {
			fmt.Println(out)
			recordBrain("analyst", true, "intent dataflow_q")
			return true
		}
		return false
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
	case "build_api":
		routes := slotRoutes(line)
		if len(routes) == 0 {
			fmt.Println("Tell me the routes and I'll compose the API — e.g. \"build an api with /users and /health\".")
			recordBrain("composer", false, "intent build_api: no routes named")
			return true
		}
		runBuildAPI(routes)
		recordBrain("composer", true, "intent build_api")
		return true
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
		if out, cands, ok := lastAnalyzeProject.AnswerCands(q); ok {
			setFuncPick(cands)
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
	case "find":
		concept := slotFindConcept(line)
		if concept == "" {
			return false
		}
		if !ensureProject() {
			return true
		}
		runFindSearch(concept)
		recordBrain("analyst", true, "intent find")
		return true
	case "learn_pieces":
		if !ensureProject() {
			return true
		}
		runLearnPieces()
		recordBrain("composer", true, "intent learn_pieces")
		return true
	case "export_understanding":
		if !ensureProject() {
			return true
		}
		runExportUnderstanding()
		recordBrain("analyst", true, "intent export_understanding")
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

// slotRoutes pulls the route paths ("/users", "/health") out of a
// build-an-API sentence, in the order the user named them.
func slotRoutes(line string) []string {
	var out []string
	for _, f := range strings.Fields(line) {
		tok := strings.Trim(f, ",.!?;:'\"()")
		if len(tok) < 2 || !strings.HasPrefix(tok, "/") {
			continue
		}
		if tok == "/build-api" || tok == "/build" {
			continue
		}
		out = append(out, tok)
	}
	return out
}

// findStopwords are the words of a search sentence that carry the
// asking, not the topic — slotFindConcept strips them so what
// remains is what the search matches on.
var findStopwords = map[string]bool{
	"find": true, "search": true, "searching": true, "look": true,
	"looking": true, "lookup": true, "code": true, "the": true,
	"a": true, "an": true, "is": true, "there": true, "anything": true,
	"any": true, "about": true, "for": true, "does": true, "this": true,
	"have": true, "has": true, "where": true, "handled": true,
	"handles": true, "handle": true, "deals": true, "dealing": true,
	"with": true, "in": true, "on": true, "of": true, "it": true,
	"that": true, "me": true, "show": true, "get": true, "got": true,
	"stuff": true, "logic": true, "related": true, "somewhere": true,
	"here": true, "project": true, "please": true, "can": true,
	"you": true, "do": true, "function": true, "functions": true,
	"thing": true, "things": true,
}

// slotFindConcept strips the search phrasing off a find sentence,
// leaving the topical words ("is there code that handles retries"
// -> "retries").
func slotFindConcept(line string) string {
	var keep []string
	for _, t := range intentTokens(line) {
		if !findStopwords[t] {
			keep = append(keep, t)
		}
	}
	return strings.Join(keep, " ")
}

// --- clarification fallback (intent layer v2) ------------------------------
//
// A sentence the classifier almost understood used to fall through
// silently, and the user learned nothing except that the front door
// "didn't get it". Now a near miss asks ONE short question — "did
// you mean — …?" — and routes on the reply (yes/no, or 1/2 when the
// model is torn between two). The band comes from the classifier's
// measured behavior: over the training and held-out corpora the
// model is essentially never gated out while holding a plausible
// favorite (it is confident, right or wrong — 0 gated-out lines in
// either corpus), so a gated line whose top intent is STILL a
// strong favorite is exactly the out-of-corpus paraphrase this
// fallback exists for. Brain-owned intents (concept_q, write_code,
// social, run_make) and the ones other stages own are never
// suggested: those lines belong to their owners either way.

const (
	// clarifyMinConf is the floor under a near miss: below it the
	// probability mass is spread thin over many intents and the
	// top guess is noise, not a favorite.
	clarifyMinConf = 0.35
	// clarifySecondFloor is the least a runner-up must hold to be
	// worth naming as the second option.
	clarifySecondFloor = 0.20
)

// dispatchableIntent reports whether dispatchIntent has a real
// case for the intent — the only intents a clarification may offer.
func dispatchableIntent(intent string) bool {
	switch intent {
	case "tour", "glossary", "learn_path", "example", "watch",
		"brains", "why_brain", "suggest", "start_guide", "tell_more",
		"what_now", "summarize", "dataflow_q", "help", "analyze",
		"where", "flow", "build_api", "whatis", "show_func",
		"walkthrough", "review", "comment", "test_scaffold",
		"uexplain", "why_func", "compare", "arch", "why_import",
		"fix", "find", "learn_pieces", "export_understanding":
		return true
	}
	return false
}

// clarifyLabel phrases an intent the way the "did you mean"
// question names it: the action in plain words, never the intent's
// code name — the user was never supposed to learn those.
func clarifyLabel(intent string) string {
	switch intent {
	case "tour":
		return "take the tour of this project"
	case "glossary":
		return "show the project's key words and types"
	case "learn_path":
		return "show the order to learn this project in"
	case "example":
		return "show a real example of a function in use"
	case "watch":
		return "run the project and watch what happens"
	case "brains":
		return "list the brains"
	case "why_brain":
		return "explain why that brain answered"
	case "suggest":
		return "hear a suggestion from the Friend"
	case "start_guide":
		return "start the guided walk"
	case "tell_more":
		return "tell you more about that"
	case "what_now":
		return "suggest what to do next"
	case "summarize":
		return "summarize what this project does"
	case "dataflow_q":
		return "trace where a value's data comes from and where it changes"
	case "help":
		return "show what you can ask"
	case "analyze":
		return "analyze a project"
	case "where":
		return "find where a change belongs"
	case "flow":
		return "run one idea end to end"
	case "build_api":
		return "build an API from pieces"
	case "whatis":
		return "explain what a function does"
	case "show_func":
		return "show a function's source"
	case "walkthrough":
		return "walk through a function step by step"
	case "review":
		return "review a function for problems"
	case "comment":
		return "write a doc comment for a function"
	case "test_scaffold":
		return "scaffold a test for a function"
	case "uexplain":
		return "explain a function, checked against the facts"
	case "why_func":
		return "explain why a function is written that way"
	case "compare":
		return "compare two functions"
	case "arch":
		return "describe how this project is put together"
	case "why_import":
		return "explain why a package is imported"
	case "fix":
		return "help fix an error"
	case "find":
		return "search the code for something"
	case "learn_pieces":
		return "learn the reusable pieces in the doc comments"
	case "export_understanding":
		return "export the understanding pairs"
	}
	return intent
}

// pendingClarification is the one outstanding "did you mean…?":
// the line that prompted it and the candidate intents, best first.
// Package state, like the guide's — exactly one at a time, and any
// later line that reaches the intent layer either answers it or
// clears it, so it can never hijack a later, unrelated sentence.
type pendingClarification struct {
	line  string
	cands []string
	conf  float64 // the top candidate's confidence when offered
}

var pendingClarify *pendingClarification

// clarifyCandidates picks the suggestion(s) for a line the gate
// just rejected, or nil when the line is no near miss worth asking
// about. It is only called with the gate already failed.
func clarifyCandidates(m *IntentModel, line string) []string {
	top, conf, second, secondConf := m.predictTop2(line)
	if conf < clarifyMinConf || !dispatchableIntent(top) {
		return nil
	}
	if conf-secondConf >= intentMinMargin {
		// The gate failed on confidence alone: one clear favorite,
		// just under the bar.
		return []string{top}
	}
	// The gate failed on margin: the model is torn between two.
	// Name both when the runner-up is also routable; otherwise the
	// favorite is still the only sensible question.
	if dispatchableIntent(second) && secondConf >= clarifySecondFloor {
		return []string{top, second}
	}
	return []string{top}
}

// offerClarification asks the one "did you mean…?" question for a
// gated-out line and remembers it as pending. It returns false
// when the line is no near miss — the line then falls through
// exactly as it did before the fallback existed.
func offerClarification(m *IntentModel, line string) bool {
	cands := clarifyCandidates(m, line)
	if len(cands) == 0 {
		return false
	}
	_, conf, _, _ := m.predictTop2(line)
	pendingClarify = &pendingClarification{line: line, cands: cands, conf: conf}
	if len(cands) == 1 {
		fmt.Printf("Did you mean — %s? (yes/no)\n", clarifyLabel(cands[0]))
	} else {
		fmt.Printf("Did you mean (1) %s, or (2) %s? (1/2/neither)\n", clarifyLabel(cands[0]), clarifyLabel(cands[1]))
	}
	return true
}

var (
	clarifyYes = map[string]bool{
		"yes": true, "y": true, "yeah": true, "yep": true,
		"sure": true, "ok": true, "okay": true, "yes please": true,
	}
	clarifyNo = map[string]bool{
		"no": true, "n": true, "nope": true, "neither": true,
		"none": true, "no thanks": true, "cancel": true, "never mind": true,
	}
)

// resolveClarification gives a pending "did you mean…?" first
// claim on the reply line. done=false means no answer was given
// (any pending question is forgotten) and the line must be
// processed normally.
func resolveClarification(line string, profile *UserProfile) (done, handled bool) {
	p := pendingClarify
	if p == nil {
		return false, false
	}
	ans := strings.ToLower(strings.Trim(strings.TrimSpace(line), ".!?"))
	pick := -1
	switch {
	case ans == "1":
		pick = 0
	case ans == "2" && len(p.cands) > 1:
		pick = 1
	case clarifyYes[ans]:
		pick = 0
	case clarifyNo[ans]:
		// Declined: suggest nothing more; the line falls through
		// to exactly the behavior it would have had today. The
		// outcome is recorded as a NEGATIVE — the line itself is
		// never mined as a training pair (its true intent is
		// unknown, only the rejected guess is known).
		recordIntentOutcome(p.line, p.cands[0], p.conf, IntentOutcomeClarifiedNo)
		pendingClarify = nil
		return true, false
	default:
		// Not an answer at all: forget the question and let the
		// line be processed fresh.
		pendingClarify = nil
		return false, false
	}
	pendingClarify = nil
	// The user just confirmed this label with their own answer —
	// the strongest training signal the log can hold.
	recordIntentOutcome(p.line, p.cands[pick], p.conf, IntentOutcomeClarifiedYes)
	friendObserveIntent(p.cands[pick])
	return true, dispatchIntent(p.cands[pick], p.line, profile)
}
