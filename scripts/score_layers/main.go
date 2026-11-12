// score_layers scores a layers-pilot model under the registered
// judging protocol (DESIGN §6 as amended by §12): model-only, raw
// prompts, greedy, fresh session per item unless a mode says
// otherwise.
//
//	go run ./scripts/score_layers -model <layers.gob> [-eval <robustness_v1.jsonl>]
//	    [-weakspot] [-heldout <social_bigdata_heldout_20261103.json>]
//	    [-holdout <holdout_test.jsonl> [-holdout-seen <holdout_seen.jsonl>]]
//	    [-off char,class,frame,...] [-shuffle-top] [-no-repair] [-frame top|slots] [-thread]
//
// Modes:
//
//	-eval      frozen robustness Section S exactly (the
//	           score_social_seat protocol: non-empty reply with
//	           >= need want words, word-boundary, case-insensitive)
//	-weakspot  the 10 targeted cases from social_targeted_eval.py
//	           (need/forbid lists, joke shape, code-marker veto),
//	           model-only adaptation disclosed in DESIGN §6
//	-heldout   the 41-item held-out set under the NCM model-only
//	           protocol (sequences joined by a single space), plus
//	           the name-bearing singles diagnostic and — with
//	           -thread — the threaded multiturn run of the 12
//	           sequences (memory facts accumulate, turn threaded)
//	-holdout   the combination holdout (DESIGN §12): corpus pairs
//	           in held-out (top x topic x tone) cells, judged by
//	           the held-out protocol's rule (non-empty reply
//	           containing >= 1 of the reference reply's content
//	           words, word-boundary, case-insensitive); the
//	           -holdout-seen file (pairs in kept cells) is scored
//	           the same way for the non-collapse bar
//
// -off forces trained layers off at inference; -shuffle-top
// applies the registered permutation to the predicted top code;
// -no-repair disables the frame's re-correction loop; -frame
// drops one frame half at inference. All are diagnostics on a
// trained model, never ablation verdicts (DESIGN §6/§12).
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"regexp"
	"sort"
	"strings"

	"github.com/golangast/gollemer/internal/ai/neural/nnu/seq2seq"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
	"github.com/golangast/gollemer/internal/ai/neural/tokenizer"
	"github.com/golangast/gollemer/internal/ai/training/chat"
)

func loadModel(path string) (*seq2seq.LayersModel, error) {
	placeholder, err := tokenizer.NewTokenizer(vocab.NewVocabulary())
	if err != nil {
		return nil, err
	}
	m, err := seq2seq.LoadLayers(path, placeholder)
	if err != nil {
		return nil, err
	}
	tok, err := tokenizer.NewTokenizer(m.Base.OutputVocab)
	if err != nil {
		return nil, err
	}
	m.Base.Tokenizer = tok
	return m, nil
}

func main() {
	modelPath := flag.String("model", "", "layers model .gob to score")
	evalPath := flag.String("eval", "", "robustness_v1.jsonl (Section S)")
	weakspot := flag.Bool("weakspot", false, "run the 10 targeted weak-spot cases")
	heldoutPath := flag.String("heldout", "", "social_bigdata_heldout_20261103.json")
	corpusPath := flag.String("corpus", "data/training/chat_pairs.jsonl", "corpus for the name-bearing diagnostic's reply-token frequencies")
	offFlag := flag.String("off", "", "comma-separated layers forced off at inference (diagnostic)")
	shuffle := flag.Bool("shuffle-top", false, "permute the predicted top code (diagnostic)")
	noRepair := flag.Bool("no-repair", false, "disable the frame re-correction loop (the loop's own ablation)")
	frameMode := flag.String("frame", "", "drop one frame half at inference: top|slots (diagnostic)")
	holdoutPath := flag.String("holdout", "", "combination-holdout test file (holdout_test.jsonl)")
	holdoutSeenPath := flag.String("holdout-seen", "", "combination-holdout seen-cell sample (holdout_seen.jsonl)")
	thread := flag.Bool("thread", false, "with -heldout: also run the 12 sequences threaded turn-by-turn")
	flag.Parse()
	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "score_layers: -model is required")
		os.Exit(2)
	}
	if *evalPath == "" && !*weakspot && *heldoutPath == "" && *holdoutPath == "" {
		fmt.Fprintln(os.Stderr, "score_layers: one of -eval, -weakspot, -heldout, -holdout is required")
		os.Exit(2)
	}
	model, err := loadModel(*modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "score_layers: loading model: %v\n", err)
		os.Exit(1)
	}
	disable := map[string]bool{}
	for _, name := range strings.Split(*offFlag, ",") {
		if name = strings.TrimSpace(name); name != "" {
			disable[name] = true
		}
	}
	opts := seq2seq.LayersPredictOptions{MaxLen: 60, Disable: disable, ShuffleTop: *shuffle, NoRepair: *noRepair}
	switch *frameMode {
	case "":
	case "top":
		opts.FrameSlotsOff = true // keep the top, drop the slots
	case "slots":
		opts.FrameTopOff = true // keep the slots, drop the top
	default:
		fmt.Fprintf(os.Stderr, "score_layers: -frame must be top|slots, got %q\n", *frameMode)
		os.Exit(2)
	}
	fmt.Printf("model: %s\ntrained layers: %v\ndisabled at inference: %v shuffle-top: %v no-repair: %v frame-mode: %q\n---\n",
		*modelPath, model.TrainedLayers(), keysOf(disable), *shuffle, *noRepair, *frameMode)

	if *evalPath != "" {
		if err := scoreEval(model, *evalPath, opts); err != nil {
			fmt.Fprintf(os.Stderr, "score_layers: %v\n", err)
			os.Exit(1)
		}
	}
	if *weakspot {
		scoreWeakspot(model, opts)
	}
	if *heldoutPath != "" {
		if err := scoreHeldout(model, *heldoutPath, *corpusPath, opts, *thread); err != nil {
			fmt.Fprintf(os.Stderr, "score_layers: %v\n", err)
			os.Exit(1)
		}
	}
	if *holdoutPath != "" {
		if err := scoreHoldout(model, *holdoutPath, *holdoutSeenPath, opts); err != nil {
			fmt.Fprintf(os.Stderr, "score_layers: %v\n", err)
			os.Exit(1)
		}
	}
}

func keysOf(m map[string]bool) []string {
	var out []string
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func predict(m *seq2seq.LayersModel, line string, sess *chat.LayersSession, opts seq2seq.LayersPredictOptions) (seq2seq.LayersPrediction, error) {
	return chat.PredictLayersText(m, line, sess, opts)
}

// ---- Section S ----

type evalItem struct {
	ID      string   `json:"id"`
	Section string   `json:"section"`
	Family  string   `json:"family"`
	Variant string   `json:"variant"`
	Input   string   `json:"input"`
	Want    []string `json:"want"`
	Need    int      `json:"need"`
}

func scoreEval(m *seq2seq.LayersModel, path string, opts seq2seq.LayersPredictOptions) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	var items []evalItem
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1024*1024), 1024*1024)
	for sc.Scan() {
		var it evalItem
		if err := json.Unmarshal(sc.Bytes(), &it); err != nil {
			return fmt.Errorf("bad eval line: %w", err)
		}
		if it.Section == "S" {
			items = append(items, it)
		}
	}
	totals := map[string][2]int{}
	bump := func(key string, pass bool) {
		t := totals[key]
		t[1]++
		if pass {
			t[0]++
		}
		totals[key] = t
	}
	repaired, residual := 0, 0
	for _, it := range items {
		pred, err := predict(m, it.Input, &chat.LayersSession{}, opts)
		if err != nil {
			return fmt.Errorf("predict %s: %w", it.ID, err)
		}
		if len(pred.Repairs) > 0 {
			repaired++
		}
		if pred.Residual {
			residual++
		}
		reply := pred.Text
		need := it.Need
		if need <= 0 {
			need = 1
		}
		hits := 0
		for _, w := range it.Want {
			if regexp.MustCompile(`(?i)\b` + regexp.QuoteMeta(w) + `\b`).MatchString(reply) {
				hits++
			}
		}
		pass := strings.TrimSpace(reply) != "" && hits >= need
		fmt.Printf("%s %s [%s/%s] %q -> %q (want-hit %d/%d)\n",
			map[bool]string{true: "PASS", false: "FAIL"}[pass], it.ID, it.Family, it.Variant, it.Input, reply, hits, len(it.Want))
		bump("S total", pass)
		bump("variant "+it.Variant, pass)
		if it.Variant == "typo" || it.Variant == "funcdrop" {
			bump("S mutated (typo+funcdrop)", pass)
		}
	}
	fmt.Println("---")
	for _, key := range []string{"S total", "S mutated (typo+funcdrop)", "variant clean", "variant paraphrase", "variant typo", "variant funcdrop"} {
		t := totals[key]
		fmt.Printf("%s: %d/%d\n", key, t[0], t[1])
	}
	fmt.Printf("frame repairs fired on %d/%d items; %d frames still inconsistent after the bound\n", repaired, len(items), residual)
	return nil
}

// ---- Weak-spot (cases copied from scripts/social_targeted_eval.py) ----

var happyWords = []string{"wonderful", "great", "fantastic", "amazing", "congratulations", "happy", "glad", "love", "thrilled"}
var sadWords = []string{"sorry", "sad", "tough", "here for you", "talk about it", "listen"}
var codeMarkers = []string{"func ", "package main", "package ", "import (", ":=", "go mod", "go run", "go build", "gofmt", "make train", "make real-chat", "{", "}", "func(", "// "}

func jokeShaped(a string) bool {
	return strings.Contains(a, "?") && strings.HasSuffix(strings.TrimSpace(a), "!") && len(strings.Fields(a)) >= 6
}

type weakCase struct {
	prompt string
	need   []string
	forbid []string
	kind   string
	shape  bool // joke-shape rule instead of need
}

var weakCases = []weakCase{
	{prompt: "i am happy!", need: happyWords, forbid: []string{"sorry"}, kind: "happy-contrast"},
	{prompt: "i am sad today", need: sadWords, forbid: happyWords, kind: "sad-contrast"},
	{prompt: "good morning!", need: []string{"morning"}, kind: "morning"},
	{prompt: "good evening!", need: []string{"evening"}, kind: "evening"},
	{prompt: "tell me a joke.", kind: "joke", shape: true},
	{prompt: "do you know a joke?", kind: "joke", shape: true},
	{prompt: "what is your name", need: []string{"gollemer"}, kind: "identity-name"},
	{prompt: "what exactly are you?", need: []string{"gollemer"}, kind: "identity-what"},
	{prompt: "i am feeling down", need: sadWords, forbid: happyWords, kind: "paraphrase-sad"},
	{prompt: "know any funny jokes", kind: "paraphrase-joke", shape: true},
}

func scoreWeakspot(m *seq2seq.LayersModel, opts seq2seq.LayersPredictOptions) {
	passed := 0
	var misses []string
	for _, c := range weakCases {
		pred, err := predict(m, c.prompt, &chat.LayersSession{}, opts)
		if err != nil {
			fmt.Printf("score_layers: weakspot predict: %v\n", err)
			os.Exit(1)
		}
		a := pred.Text
		low := strings.ToLower(a)
		ok := true
		detail := ""
		if c.shape {
			ok = jokeShaped(a)
			if !ok {
				detail = "not joke-shaped"
			}
		} else {
			ok = false
			for _, n := range c.need {
				if strings.Contains(low, n) {
					ok = true
					break
				}
			}
			if !ok {
				detail = fmt.Sprintf("missing one of %v", c.need)
			}
		}
		for _, w := range c.forbid {
			if strings.Contains(low, w) {
				ok = false
				detail += fmt.Sprintf(" WRONG-POLARITY:%s", w)
			}
		}
		for _, mk := range codeMarkers {
			if strings.Contains(low, strings.ToLower(mk)) {
				ok = false
				detail += fmt.Sprintf(" CODE-MARKER:%q", mk)
			}
		}
		mark := "FAIL"
		if ok {
			mark = "PASS"
			passed++
		} else {
			misses = append(misses, c.kind)
		}
		fmt.Printf("[%s][%s] %s%s\n       -> %s\n", mark, c.kind, c.prompt, detail, a)
	}
	fmt.Printf("weak-spot: %d/%d  misses: %v\n", passed, len(weakCases), misses)
}

// ---- Held-out 41 ----

type heldout struct {
	Note    string `json:"note"`
	Singles []struct {
		ID      string   `json:"id"`
		Prompt  string   `json:"prompt"`
		Topic   string   `json:"topic"`
		WantAny []string `json:"want_any"`
	} `json:"singles"`
	Sequences []struct {
		ID      string   `json:"id"`
		Topic   string   `json:"topic"`
		Turns   []string `json:"turns"`
		WantAny []string `json:"want_any"`
	} `json:"sequences"`
}

func anyHit(reply string, wantAny []string) bool {
	low := strings.ToLower(reply)
	for _, w := range wantAny {
		if strings.Contains(low, strings.ToLower(w)) {
			return true
		}
	}
	return false
}

// topReplyTokens returns the N most frequent tokens across the
// corpus's reply texts (the name-bearing diagnostic's reference
// set, DESIGN §6).
func topReplyTokens(corpusPath string, n int) (map[string]bool, error) {
	data, err := os.ReadFile(corpusPath)
	if err != nil {
		return nil, err
	}
	freq := map[string]int{}
	for _, line := range strings.Split(string(data), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		var pair struct {
			Output string `json:"output"`
		}
		if err := json.Unmarshal([]byte(line), &pair); err != nil {
			continue
		}
		for _, tok := range strings.FieldsFunc(strings.ToLower(pair.Output), func(r rune) bool {
			return r == ' ' || r == '\t'
		}) {
			freq[tok]++
		}
	}
	type kv struct {
		w string
		c int
	}
	var kvs []kv
	for w, c := range freq {
		kvs = append(kvs, kv{w, c})
	}
	sort.Slice(kvs, func(i, j int) bool {
		if kvs[i].c != kvs[j].c {
			return kvs[i].c > kvs[j].c
		}
		return kvs[i].w < kvs[j].w
	})
	out := map[string]bool{}
	for i := 0; i < n && i < len(kvs); i++ {
		out[kvs[i].w] = true
	}
	return out, nil
}

func scoreHeldout(m *seq2seq.LayersModel, path, corpusPath string, opts seq2seq.LayersPredictOptions, thread bool) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var ho heldout
	if err := json.Unmarshal(data, &ho); err != nil {
		return err
	}
	top, err := topReplyTokens(corpusPath, 500)
	if err != nil {
		return fmt.Errorf("name-bearing diagnostic needs the corpus: %w", err)
	}
	total, passed := 0, 0
	nameTotal, namePassed := 0, 0
	for _, s := range ho.Singles {
		pred, err := predict(m, s.Prompt, &chat.LayersSession{}, opts)
		if err != nil {
			return err
		}
		ok := anyHit(pred.Text, s.WantAny)
		total++
		if ok {
			passed++
		}
		// Name-bearing: the prompt shares a token with want_any
		// that is not among the 500 most frequent reply tokens.
		nameBearing := false
		promptToks := map[string]bool{}
		for _, t := range strings.Fields(strings.ToLower(s.Prompt)) {
			promptToks[strings.Trim(t, ".,!?")] = true
		}
		for _, w := range s.WantAny {
			lw := strings.ToLower(w)
			if promptToks[lw] && !top[lw] {
				nameBearing = true
			}
		}
		tag := ""
		if nameBearing {
			nameTotal++
			if ok {
				namePassed++
			}
			tag = " [name-bearing]"
		}
		fmt.Printf("%s %s %q -> %q%s\n", map[bool]string{true: "PASS", false: "FAIL"}[ok], s.ID, s.Prompt, pred.Text, tag)
	}
	for _, q := range ho.Sequences {
		joined := strings.Join(q.Turns, " ")
		pred, err := predict(m, joined, &chat.LayersSession{}, opts)
		if err != nil {
			return err
		}
		ok := anyHit(pred.Text, q.WantAny)
		total++
		if ok {
			passed++
		}
		fmt.Printf("%s %s (joined) %q -> %q\n", map[bool]string{true: "PASS", false: "FAIL"}[ok], q.ID, joined, pred.Text)
	}
	fmt.Printf("held-out: %d/%d\n", passed, total)
	fmt.Printf("held-out name-bearing singles: %d/%d (of %d singles)\n", namePassed, nameTotal, len(ho.Singles))

	if thread {
		tPassed, nameOK := 0, false
		for _, q := range ho.Sequences {
			sess := &chat.LayersSession{}
			var last seq2seq.LayersPrediction
			for _, turn := range q.Turns {
				pred, err := predict(m, turn, sess, opts)
				if err != nil {
					return err
				}
				sess.Observe(turn, pred.TokenIDs)
				last = pred
			}
			ok := anyHit(last.Text, q.WantAny)
			if ok {
				tPassed++
			}
			if q.Topic == "pets-context" && ok {
				nameOK = true
			}
			fmt.Printf("%s %s (threaded) turns=%v -> final %q\n", map[bool]string{true: "PASS", false: "FAIL"}[ok], q.ID, q.Turns, last.Text)
		}
		fmt.Printf("held-out threaded sequences: %d/%d  name cell (pets-context): %v\n", tPassed, len(ho.Sequences), nameOK)
	}
	return nil
}

// ---- Combination holdout (DESIGN §12) ----

type holdoutItem struct {
	Input  string   `json:"input"`
	Output string   `json:"output"`
	Top    int      `json:"top"`
	Topic  int      `json:"topic"`
	Tone   int      `json:"tone"`
	Want   []string `json:"want"`
}

func loadHoldoutItems(path string) ([]holdoutItem, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var items []holdoutItem
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1024*1024), 1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var it holdoutItem
		if err := json.Unmarshal([]byte(line), &it); err != nil {
			return nil, fmt.Errorf("bad holdout line: %w", err)
		}
		items = append(items, it)
	}
	return items, sc.Err()
}

// scoreHoldoutSet scores one holdout file with the registered
// rule and prints per-cell subtotals plus the set total. Returns
// (passed, total).
func scoreHoldoutSet(m *seq2seq.LayersModel, label, path string, opts seq2seq.LayersPredictOptions) (int, int, error) {
	items, err := loadHoldoutItems(path)
	if err != nil {
		return 0, 0, err
	}
	cells := map[string][2]int{}
	cellKey := func(it holdoutItem) string {
		return fmt.Sprintf("top=%d topic=%d tone=%d", it.Top, it.Topic, it.Tone)
	}
	passed := 0
	for _, it := range items {
		pred, err := predict(m, it.Input, &chat.LayersSession{}, opts)
		if err != nil {
			return 0, 0, fmt.Errorf("predict %q: %w", it.Input, err)
		}
		reply := pred.Text
		ok := false
		if strings.TrimSpace(reply) != "" {
			for _, w := range it.Want {
				if regexp.MustCompile(`(?i)\b` + regexp.QuoteMeta(w) + `\b`).MatchString(reply) {
					ok = true
					break
				}
			}
		}
		if ok {
			passed++
		}
		c := cells[cellKey(it)]
		c[1]++
		if ok {
			c[0]++
		}
		cells[cellKey(it)] = c
		fmt.Printf("%s [%s] %q -> %q\n", map[bool]string{true: "PASS", false: "FAIL"}[ok], cellKey(it), it.Input, reply)
	}
	var keys []string
	for k := range cells {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		c := cells[k]
		fmt.Printf("%s cell %s: %d/%d\n", label, k, c[0], c[1])
	}
	fmt.Printf("%s total: %d/%d\n", label, passed, len(items))
	return passed, len(items), nil
}

func scoreHoldout(m *seq2seq.LayersModel, testPath, seenPath string, opts seq2seq.LayersPredictOptions) error {
	fmt.Println("--- combination holdout: held-out cells")
	hp, ht, err := scoreHoldoutSet(m, "holdout", testPath, opts)
	if err != nil {
		return err
	}
	if seenPath != "" {
		fmt.Println("--- combination holdout: seen cells")
		sp, st, err := scoreHoldoutSet(m, "seen", seenPath, opts)
		if err != nil {
			return err
		}
		if ht > 0 && st > 0 {
			fmt.Printf("holdout summary: held-out cells %.1f%% (%d/%d), seen cells %.1f%% (%d/%d)\n",
				100*float64(hp)/float64(ht), hp, ht, 100*float64(sp)/float64(st), sp, st)
		}
	}
	return nil
}
