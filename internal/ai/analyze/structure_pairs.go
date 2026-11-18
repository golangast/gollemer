// Package analyze — structure-to-language training pairs (v2).
//
// V1 FAILED its held-out test (2026-10-20): the model parroted memorized
// function names on unseen code and invented behaviors. Diagnosis:
//   1. The output had to COPY a name the decoder had never generated —
//      whole-word tokens make every new name effectively invisible.
//   2. 854 near-identical templated pairs taught the template, not the
//      reading; ignoring the input barely hurt the training loss.
//
// V2 attacks the mechanism, not the capacity:
//   - Placeholders: the output never contains the real function or
//     receiver name. It emits "@" (function) and "@@" (receiver); the
//     CALLER substitutes the true names after generation. The model
//     cannot say the wrong name — it never writes it.
//   - Subword inputs: names enter as split camel words ("make request"),
//     so unseen names decompose into pieces the model knows.
//   - Name-swap augmentation: the same behavior pattern is emitted with
//     hundreds of different names (real + invented). Memorization stops
//     working; the only signal that predicts the output is the input.
//   - Behavior-steering synthetics: random name/receiver/behavior
//     combinations whose outputs follow the behavior codes exactly, so
//     the gradient rewards reading the behavior list.
//
// The verdict probe: "zorblax on client : may-fail" must yield an
// explanation about @ (→ Zorblax) claiming may-fail and nothing else.
package analyze

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"strings"
)

// StructurePair is a single training example: structural facts → explanation.
type StructurePair struct {
	Input  string `json:"input"`
	Output string `json:"output"`
	Kind   string `json:"kind"` // "func_explain", "type_explain", "swap", "synthetic"
}

// Behavior codes used in fact strings. Order is canonical: locking,
// iteration, failure.
const (
	behReadLocks = "read-locks"
	behLocks     = "locks"
	behIterates  = "iterates"
	behMayFail   = "may-fail"
)

// FuncBehaviors extracts the behavior codes for a function from its
// signature and source. This is the exact set the chat-side gate checks.
func FuncBehaviors(fn *Func) []string {
	src := strings.ToLower(fn.Src)
	var behaviors []string
	if strings.Contains(src, "rlock()") {
		behaviors = append(behaviors, behReadLocks)
	} else if strings.Contains(src, ".lock()") {
		behaviors = append(behaviors, behLocks)
	}
	if strings.Contains(src, "range") {
		behaviors = append(behaviors, behIterates)
	}
	if strings.Contains(fn.Sig, "error") {
		behaviors = append(behaviors, behMayFail)
	}
	return behaviors
}

// SplitCamelWords splits an identifier into lowercase words:
// "MakeRequest" → "make request", "HTTPServer" → "http server".
// Subword inputs let unseen names decompose into known pieces.
func SplitCamelWords(name string) string {
	return strings.Join(camelWords(name), " ")
}

func camelWords(name string) []string {
	var words []string
	var cur []rune
	runes := []rune(name)
	flush := func() {
		if len(cur) > 0 {
			words = append(words, strings.ToLower(string(cur)))
			cur = cur[:0]
		}
	}
	for i, r := range runes {
		switch {
		case r == '_' || r == '-' || r == ' ':
			flush()
		case r >= 'A' && r <= 'Z':
			// New word at a case transition, except inside an acronym run
			// (HTTPServer → http, server: split before the last capital
			// when a lowercase follows).
			if len(cur) > 0 {
				prev := runes[i-1]
				nextLower := i+1 < len(runes) && runes[i+1] >= 'a' && runes[i+1] <= 'z'
				if (prev >= 'a' && prev <= 'z') || (prev >= 'A' && prev <= 'Z' && nextLower) {
					flush()
				}
			}
			cur = append(cur, r)
		default:
			cur = append(cur, r)
		}
	}
	flush()
	return words
}

// FactsString builds the v2 model input from raw parts:
// "make request on client : may-fail". Single source of truth — the
// generator and the chat (/uexplain) both build inputs through here so
// train and inference formats can never drift apart.
func FactsString(name, receiver string, behaviors []string) string {
	var b strings.Builder
	b.WriteString(SplitCamelWords(name))
	if receiver != "" {
		b.WriteString(" on " + SplitCamelWords(strings.Trim(receiver, "*")))
	}
	if len(behaviors) > 0 {
		b.WriteString(" : " + strings.Join(behaviors, ","))
	}
	return b.String()
}

// FuncFactsString is FactsString for a real analyzed function.
func FuncFactsString(fn *Func) string {
	return FactsString(fn.Name, fn.Receiver, FuncBehaviors(fn))
}

// behaviorClause renders one behavior code as an output clause.
func behaviorClause(code string) string {
	switch code {
	case behReadLocks:
		return "it locks for reading"
	case behLocks:
		return "it locks for writing"
	case behIterates:
		return "it iterates over the data"
	case behMayFail:
		return "it may fail"
	}
	return ""
}

var verbStems = map[string]bool{
	"make": true, "get": true, "set": true, "create": true, "build": true,
	"handle": true, "process": true, "read": true, "write": true,
	"load": true, "save": true, "find": true, "check": true,
	"validate": true, "parse": true, "format": true, "compute": true,
	"generate": true, "register": true, "store": true, "fetch": true,
	"send": true, "update": true, "delete": true, "remove": true,
	"add": true, "list": true, "show": true, "print": true, "open": true,
	"close": true, "start": true, "stop": true, "run": true,
	"resolve": true, "convert": true, "encode": true, "decode": true,
	"serve": true, "apply": true, "seed": true, "scan": true,
	"walk": true, "describe": true, "explain": true, "review": true,
	"suggest": true, "learn": true, "compose": true, "insert": true,
	"append": true, "draw": true, "complete": true, "finalize": true,
	"freeze": true, "lock": true, "unlock": true, "sort": true,
	"search": true, "filter": true, "merge": true, "split": true,
	"trim": true, "render": true, "route": true, "clone": true,
	"export": true, "import": true, "train": true, "predict": true,
	"classify": true, "normalize": true, "extract": true, "derive": true,
	"count": true, "test": true, "wrap": true,
}

// conjugateWord conjugates a single lowercase verb: make→makes,
// process→processes, try→tries.
func conjugateWord(w string) string {
	switch {
	case strings.HasSuffix(w, "s") || strings.HasSuffix(w, "x") ||
		strings.HasSuffix(w, "ch") || strings.HasSuffix(w, "sh"):
		return w + "es"
	case strings.HasSuffix(w, "y") && len(w) > 1 && !strings.ContainsRune("aeiou", rune(w[len(w)-2])):
		return w[:len(w)-1] + "ies"
	default:
		return w + "s"
	}
}

// verbPhrase derives the output verb phrase from a function name.
// Mirrors the chat template's docVerb rules so model and template agree.
func pairVerbPhrase(name string) string {
	words := camelWords(name)
	if len(words) == 0 {
		return "runs"
	}
	rest := strings.Join(words[1:], " ")
	switch words[0] {
	case "get":
		if rest != "" {
			return "returns the " + rest
		}
	case "set":
		if rest != "" {
			return "sets the " + rest
		}
	case "new":
		if rest != "" {
			return "creates and returns a new " + rest
		}
		return "creates a new one"
	case "must":
		if rest != "" {
			return "returns the " + rest + ", and panics if it is missing"
		}
	case "is":
		if rest != "" {
			return "reports whether it is " + rest
		}
	case "has":
		if rest != "" {
			return "reports whether it has " + rest
		}
	case "can":
		if rest != "" {
			return "reports whether it can " + rest
		}
	}
	if verbStems[words[0]] {
		if rest != "" {
			return conjugateWord(words[0]) + " the " + rest
		}
		return conjugateWord(words[0])
	}
	// Plural noun ("Names") → returns the names.
	if strings.HasSuffix(words[len(words)-1], "s") && !strings.HasSuffix(words[len(words)-1], "ss") {
		return "returns the " + strings.Join(words, " ")
	}
	if rest != "" {
		return conjugateWord(words[0]) + " the " + rest
	}
	return conjugateWord(words[0])
}

// ExplanationFromFacts builds the v2 target output from raw parts, with
// "@" for the function name and "@@" for the receiver. The caller
// substitutes the real names after generation.
func ExplanationFromFacts(name, receiver string, behaviors []string) string {
	var b strings.Builder
	b.WriteString("@ " + pairVerbPhrase(name))
	if receiver != "" {
		b.WriteString(" on the @@")
	}
	b.WriteString(".")
	for _, code := range behaviors {
		if c := behaviorClause(code); c != "" {
			b.WriteString(" " + strings.ToUpper(c[:1]) + c[1:] + ".")
		}
	}
	return b.String()
}

// GenerateStructurePairs walks a project and creates v2 training pairs:
// real pairs, name-swap augmentations, and behavior-steering synthetics.
func GenerateStructurePairs(p *Project) []StructurePair {
	var pairs []StructurePair
	var names, receivers []string
	seenNames := map[string]bool{}
	seenRecv := map[string]bool{}

	type rawFunc struct {
		name, receiver string
		behaviors      []string
	}
	var raw []rawFunc

	for _, pkg := range p.Packages {
		for _, fn := range pkg.Funcs {
			if fn.IsTest || fn.Name == "" || fn.Sig == "" {
				continue
			}
			recv := strings.Trim(fn.Receiver, "*")
			raw = append(raw, rawFunc{fn.Name, recv, FuncBehaviors(fn)})
			if !seenNames[fn.Name] {
				seenNames[fn.Name] = true
				names = append(names, fn.Name)
			}
			if recv != "" && !seenRecv[recv] {
				seenRecv[recv] = true
				receivers = append(receivers, recv)
			}
			pairs = append(pairs, StructurePair{
				Input:  FuncFactsString(fn),
				Output: ExplanationFromFacts(fn.Name, recv, FuncBehaviors(fn)),
				Kind:   "func_explain",
			})
		}
		for _, typ := range pkg.Types {
			if typ.Name == "" {
				continue
			}
			n := len(typ.Methods)
			if n == 0 {
				continue
			}
			if n > 9 {
				n = 9 // keep the count word small and learnable
			}
			pairs = append(pairs, StructurePair{
				Input:  SplitCamelWords(typ.Name) + fmt.Sprintf(" type : %d methods", n),
				Output: fmt.Sprintf("@ is a struct with %d methods.", n),
				Kind:   "type_explain",
			})
		}
	}

	// Name-swap augmentation: same behaviors, different names/receivers.
	// Memorization cannot predict these outputs — only reading can.
	rng := rand.New(rand.NewSource(42))
	pool := append(append([]string{}, names...), swapNamePool...)
	recvPool := append(append([]string{}, receivers...), swapReceiverPool...)
	for _, rf := range raw {
		for k := 0; k < 2; k++ {
			nn := pool[rng.Intn(len(pool))]
			if nn == rf.name {
				continue
			}
			nr := rf.receiver
			if nr != "" && rng.Intn(2) == 0 {
				nr = recvPool[rng.Intn(len(recvPool))]
			}
			pairs = append(pairs, StructurePair{
				Input:  FactsString(nn, nr, rf.behaviors),
				Output: ExplanationFromFacts(nn, nr, rf.behaviors),
				Kind:   "swap",
			})
		}
	}

	// Behavior-steering synthetics: random combinations whose outputs
	// follow the behavior codes exactly.
	behaviorSets := [][]string{
		{}, {behLocks}, {behReadLocks}, {behIterates}, {behMayFail},
		{behLocks, behIterates}, {behReadLocks, behIterates},
		{behLocks, behMayFail}, {behReadLocks, behMayFail},
		{behIterates, behMayFail}, {behLocks, behIterates, behMayFail},
		{behReadLocks, behIterates, behMayFail},
	}
	for i := 0; i < 400; i++ {
		nn := pool[rng.Intn(len(pool))]
		nr := ""
		if rng.Intn(4) > 0 {
			nr = recvPool[rng.Intn(len(recvPool))]
		}
		bs := behaviorSets[rng.Intn(len(behaviorSets))]
		pairs = append(pairs, StructurePair{
			Input:  FactsString(nn, nr, bs),
			Output: ExplanationFromFacts(nn, nr, bs),
			Kind:   "synthetic",
		})
	}

	// Dedupe identical inputs (keep the first).
	seen := map[string]bool{}
	out := pairs[:0]
	for _, pr := range pairs {
		if seen[pr.Input] {
			continue
		}
		seen[pr.Input] = true
		out = append(out, pr)
	}
	return out
}

// Invented names for augmentation — pronounceable nonsense plus
// verb-y compounds, so swapped names cover shapes real names take.
var swapNamePool = []string{
	"Zorblax", "Quindle", "Fennwick", "Torvane", "Melbrin", "Ostrav",
	"Keldoon", "Prantix", "Velmore", "Brindle", "Cormack", "Dunmore",
	"Elowen", "Falcrest", "Gorvane", "Kestrel", "MakeWidget",
	"FetchGrommet", "ParseWicket", "StoreBaffle", "ListTundras",
	"GetHobnob", "CheckLumen", "BuildTessel", "Names", "Values",
	"Entries", "HandlePortal", "ReadQuarry", "WriteLantern",
}

var swapReceiverPool = []string{
	"Client", "Store", "Handler", "Engine", "Cache", "Broker",
	"Widget", "Grommet", "Server", "Manager", "Portal", "Quarry",
}

// ExportStructurePairs writes pairs to a JSONL file for training.
// The domain is written explicitly so dataset selection is exact —
// v1's classified-pair contamination cannot recur.
func ExportStructurePairs(pairs []StructurePair, path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	for _, p := range pairs {
		data, err := json.Marshal(map[string]string{
			"input":  p.Input,
			"output": p.Output,
			"domain": "understanding",
		})
		if err != nil {
			continue
		}
		f.Write(append(data, '\n'))
	}
	return nil
}
