package chat

// The speccopy proposer: the trained model wired into the
// write family as a GATED PROPOSER, and only that. It sits in
// the claim chain immediately after the concept book and
// before the write-sentence boundary, and it fires solely
// where a declaration-shaped ask (add/put/create + a
// struct/interface/function/method + a named file or folder)
// MISSED the concept book's grammar — every sentence the book
// serves is served exactly as before, with the model never
// consulted.
//
// The model's measured record is why the shape is this tight:
// on phrasings and combinations it never trained on, its
// whole-spec exact score was 44.4% / 34.7% against a ~90% bar
// (converged copy rerun, 2026-10-30) — it did not earn a
// writer's seat. So it proposes, through every safeguard the
// deterministic writers pass through:
//
//   - its raw output must parse as a spec (decl.ParseSpecText)
//     and restore through the decl registry's case maps
//     (decl.RestoreSpec) — a hallucinated name or type ends
//     the claim silently, and the boundary answers the line
//     exactly as it does today;
//   - the restored spec renders deterministically
//     (decl.RenderDecl — the eval's own pipeline), and the
//     rendered text enters the shared spine as data
//     (analyze.EditProposedDecl): same plan display, same
//     [y/n] gate, same transaction, same build verification
//     as a concept-book write, with one plan note saying
//     plainly that a model proposed it;
//   - the model file is local-only (never in git). Absent —
//     or unreadable, or not a pointer-generator at all — the
//     feature is silently off and the chat behaves exactly as
//     it did before this claim existed.

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"github.com/golangast/gollemer/internal/ai/analyze"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/seq2seq"
	mainvocab "github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
	"github.com/golangast/gollemer/internal/ai/neural/tokenizer"
	"github.com/golangast/gollemer/internal/ai/training/decl"
)

var (
	proposerOnce = &sync.Once{} // a pointer so tests can swap in a fresh one
	// proposerFunc is the loaded model's prediction seam:
	// phrasing in, spec text out. Nil means the model is
	// unavailable and the proposer never fires.
	proposerFunc func(phrasing string) (string, error)
)

// proposerPredictor lazily loads the speccopy model once and
// returns its prediction function, or nil when the model file
// isn't there (or isn't usable) — the feature-off state.
func proposerPredictor() func(string) (string, error) {
	proposerOnce.Do(loadSpecProposer)
	return proposerFunc
}

// loadSpecProposer finds and loads the speccopy checkpoint
// under the same roots the concept book reads its data from.
// Every failure is silent by design: a chat without the model
// is a chat that behaves exactly as before.
func loadSpecProposer() {
	for _, base := range []string{unifiedProjectRoot, startFolder()} {
		if base == "" {
			continue
		}
		path := RealModelPath(base, SpecCopyDomain)
		if _, err := os.Stat(path); err != nil {
			continue
		}
		// The brain loader's pattern (loadRealModel): a
		// placeholder tokenizer for the load, then the real
		// one rebuilt from the model's own vocabulary.
		placeholder, err := tokenizer.NewTokenizer(mainvocab.NewVocabulary())
		if err != nil {
			return
		}
		model, err := seq2seq.Load(path, placeholder)
		if err != nil {
			return
		}
		if model.OutputVocab == nil || model.OutputVocab.Size() == 0 {
			return
		}
		tok, err := tokenizer.NewTokenizer(model.OutputVocab)
		if err != nil {
			return
		}
		model.Tokenizer = tok
		if model.Decoder == nil || model.Decoder.PtrGen == nil {
			return // a checkpoint without the copy head can't propose
		}
		proposerFunc = func(phrasing string) (string, error) {
			return model.PredictPtrGen(phrasing, 64)
		}
		return
	}
}

// proposedAsk is one declaration-shaped ask the concept book
// missed: the target as typed, and the sentence with its
// target clause removed, lowercased into the model's world —
// its training phrasings never carry a destination.
type proposedAsk struct {
	target   string
	phrasing string
}

// proposedKindWords are the declaration kinds the model knows:
// its three spec kinds, with "method" for a receiver-bearing
// function.
var proposedKindWords = map[string]bool{
	"struct": true, "interface": true, "function": true, "func": true, "method": true,
}

// parseProposedAsk reads one line as a declaration-shaped ask
// for the proposer. The bar is deliberately as tight as the
// boundary's: a write verb the edit family owns (add|put, plus
// create), a declaration kind word, and a named target — and
// never the delegate's "that calls" clause, which names a
// behaviour to wrap rather than a signature to propose.
func parseProposedAsk(line string) (proposedAsk, bool) {
	s := strings.TrimSpace(line)
	s = strings.TrimSuffix(s, ".")
	s = strings.TrimSuffix(s, "?")
	s = strings.TrimSpace(s)
	fields := strings.Fields(s)
	if len(fields) < 4 {
		return proposedAsk{}, false
	}
	low := make([]string, len(fields))
	for i, f := range fields {
		low[i] = strings.ToLower(strings.Trim(f, ",;:\"'()"))
	}
	switch low[0] {
	case "add", "put", "create":
	default:
		return proposedAsk{}, false
	}
	kind := false
	for _, w := range low[1:] {
		if proposedKindWords[w] {
			kind = true
			break
		}
	}
	if !kind {
		return proposedAsk{}, false
	}
	for i := 1; i+1 < len(low); i++ {
		if low[i] == "that" && low[i+1] == "calls" {
			return proposedAsk{}, false
		}
	}
	target, hasTarget := writeBoundaryTarget(fields, low)
	if !hasTarget || target == "" {
		return proposedAsk{}, false
	}
	phraseFields, ok := stripTargetClause(fields, low, target)
	if !ok {
		return proposedAsk{}, false
	}
	phrasing := strings.ToLower(strings.Join(phraseFields, " "))
	stillKind := false
	for _, w := range strings.Fields(phrasing) {
		if proposedKindWords[w] {
			stillKind = true
			break
		}
	}
	if phrasing == "" || !stillKind {
		return proposedAsk{}, false
	}
	return proposedAsk{target: target, phrasing: phrasing}, true
}

// stripTargetClause removes the target clause from the line's
// fields: the tail form ("… to [the] [file] <target>") by an
// exact tail match, else the mid-line form the delegate's
// sentences use ("… in <file>.go …"), where the preposition
// and the .go token come out and the rest of the sentence
// stays.
func stripTargetClause(fields, low []string, target string) ([]string, bool) {
	for i := len(fields) - 2; i >= 1; i-- {
		if low[i] != "to" && low[i] != "in" && low[i] != "into" {
			continue
		}
		j := i + 1
		for j < len(fields) && (low[j] == "the" || conceptDesignator(low[j])) {
			j++
		}
		if j < len(fields) && strings.Join(fields[j:], " ") == target {
			return fields[:i], true
		}
	}
	for j := 1; j < len(fields); j++ {
		if !strings.HasSuffix(low[j], ".go") || strings.Trim(fields[j], ",;:\"'()") != target {
			continue
		}
		k := j - 1
		for k >= 1 && (low[k] == "the" || conceptDesignator(low[k])) {
			k--
		}
		if low[k] == "to" || low[k] == "in" || low[k] == "into" {
			out := append([]string{}, fields[:k]...)
			return append(out, fields[j+1:]...), true
		}
	}
	return nil, false
}

// tryProposedDecl is the chat-loop entry: it claims
// declaration-shaped grammar misses the model can propose for,
// and runs the gated flow on the chat's line reader.
func tryProposedDecl(line string, r *LineReader) bool {
	return runProposedDecl(line, func(prompt string) (string, error) {
		return r.ReadLine(prompt)
	})
}

// runProposedDecl carries one grammar-missed declaration ask
// from the model's proposal to the shared edit spine. It
// returns false — the line falls through to the boundary,
// answered exactly as today — whenever the ask isn't the
// proposer's shape, the model is unavailable, or its answer
// can't be parsed and restored into a real declaration.
func runProposedDecl(line string, ask editAsk) bool {
	// The concept book's own sentences are never the model's:
	// chain order already serves them first, and this check
	// keeps the guarantee true at the seam itself.
	if _, ok := parseConcept(line); ok {
		return false
	}
	pa, ok := parseProposedAsk(line)
	if !ok {
		return false
	}
	predict := proposerPredictor()
	if predict == nil {
		return false
	}
	raw, err := predict(pa.phrasing)
	if err != nil {
		return false
	}
	spec, err := decl.ParseSpecText(raw)
	if err != nil {
		return false
	}
	restored, ok := decl.RestoreSpec(spec)
	if !ok {
		return false
	}
	text := decl.RenderDecl(restored)
	if text == "" {
		return false
	}

	// Target resolution and project acquisition: the concept
	// flow's own rules, in its own words — the sentence was
	// this claim's once the model produced a real proposal.
	tag := cTag("gollemer [composer]>")
	rootRaw := creationRoot()
	if rootRaw == "" {
		fmt.Printf("%s I don't have a project folder to edit in — start me inside your project and ask again.\n", tag)
		recordBrain("composer", false, "proposed refused: no root")
		return true
	}
	rootAbs, err := filepath.EvalSymlinks(rootRaw)
	if err != nil {
		abs, aerr := filepath.Abs(rootRaw)
		if aerr != nil {
			fmt.Printf("%s I couldn't find the folder I'd edit in (%s).\n", tag, rootRaw)
			recordBrain("composer", false, "proposed refused: root unreadable")
			return true
		}
		rootAbs = abs
	}
	if st, serr := os.Stat(rootAbs); serr != nil || !st.IsDir() {
		fmt.Printf("%s The folder I'd edit in (%s) isn't there — start me inside your project and ask again.\n", tag, rootRaw)
		recordBrain("composer", false, "proposed refused: root missing")
		return true
	}
	rel, refusal := resolveServerFile(rootAbs, pa.target)
	if refusal != "" {
		fmt.Printf("%s Not adding that: %s.\n", tag, strings.TrimSuffix(refusal, "."))
		recordBrain("composer", false, "proposed refused: target")
		return true
	}
	if lastAnalyzeProject == nil || lastAnalyzeRoot != rootRaw {
		if !analyzeQuiet(rootAbs) {
			recordBrain("composer", false, "proposed refused: project unreadable")
			return true
		}
	} else if !ensureFreshProject() {
		recordBrain("composer", false, "proposed refused: stale project unreadable")
		return true
	}

	receiver := ""
	if restored.Receiver != nil {
		receiver = restored.Receiver.Type
	}
	return executeEditPlan(analyze.EditRequest{
		Kind:             analyze.EditProposedDecl,
		File:             rel,
		ProposedText:     text,
		ProposedName:     restored.Name,
		ProposedKind:     string(restored.Kind),
		ProposedReceiver: receiver,
		ProposedImports:  strings.Join(proposedImports(restored), " "),
	}, ask)
}

// proposedImports lists the import paths a restored spec's
// type expressions reference, through the decl registry's own
// map — the facts the planner needs to dress the target file.
func proposedImports(spec decl.Spec) []string {
	set := map[string]bool{}
	add := func(expr string) {
		for _, path := range decl.ImportPaths(expr) {
			set[path] = true
		}
	}
	for _, f := range spec.Fields {
		add(f.Type.Expr)
	}
	for _, e := range spec.Embeds {
		add(e.Expr)
	}
	for _, m := range spec.Methods {
		for _, prm := range m.Params {
			add(prm.Type.Expr)
		}
		for _, r := range m.Results {
			add(r.Expr)
		}
	}
	for _, prm := range spec.Params {
		add(prm.Type.Expr)
	}
	for _, r := range spec.Results {
		add(r.Expr)
	}
	if spec.Receiver != nil {
		add(spec.Receiver.Type)
	}
	out := make([]string, 0, len(set))
	for path := range set {
		out = append(out, path)
	}
	sort.Strings(out)
	return out
}
