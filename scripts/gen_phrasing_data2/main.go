// gen_phrasing_data2 generates the phrasing layer's ATTEMPT-2
// training data (2026-10-31): the same messy -> canonical task
// as gen_phrasing_data, with the generalization levers run 1's
// verdict pointed at:
//
//   - DENSITY: ~5x the pairs, with per-family floors (concept
//     rows go from 16 pairs to >= 250 each).
//   - FACTOR CROSSING: on top of the random core x wrapper
//     sampling, a twins pass emits minimal pairs — the same ask
//     dressed in every core with one wrapper, in one core with
//     every wrapper, and rebuilt with fresh slot values under
//     the same core+wrapper — so held-out items are
//     recombinations of seen parts, not new parts.
//   - RANDOM-NAME POOLS: the classic train pools are unioned
//     with large generated pools (syllable-built nonsense
//     names, generated files/fields/texts/keys) so slot values
//     are mostly words with no meaning to memorize: the model
//     must learn "carry any token".
//
// THE JUDGE IS NOT REGENERATED. data/training/
// phrasing_heldout.jsonl stays exactly the file run 1 faced;
// this generator only READS it, for the leak checks: train and
// held-out template families disjoint (train wrappers/glues/
// core dialects only, as in v1), slot pools disjoint (the
// held-out pools below are the judge's, kept for the check),
// normalized input overlap exactly 0. Any leak fails the run.
//
// Outputs: data/training/phrasing_pairs2.jsonl (arm A's pairs)
// and data/training/phrasing_fragments2.jsonl (arm B's
// classifier examples: single inputs with their family, plus
// each compound's two piece cores with theirs — the fragments
// a glue split would produce).
//
// Run: go run ./scripts/gen_phrasing_data2
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// ---------- shared shapes ----------

type pair struct {
	Input    string   `json:"input"`
	Output   string   `json:"output"`
	Domain   string   `json:"domain,omitempty"`
	Family   string   `json:"family"`
	Template string   `json:"template"`
	Slots    []string `json:"slots,omitempty"`
}

type heldItem struct {
	Input    string `json:"input"`
	Output   string `json:"output"`
	Family   string `json:"family"`
	Cell     string `json:"cell"`
	Template string `json:"template"`
}

// slots carries the slot values of one ask: the open-vocabulary
// words a recognition model must COPY, verbatim, from the
// input. The trainer masks them to practice exactly that.
type ask struct {
	family    string
	canonical string
	slots     []string
	// core builders produce the ask in plain words (close to
	// canonical, word order free); wrappers dress a core.
	cores []func() string
}

// ---------- slot pools (train / held-out disjoint) ----------

var namesTrain = []string{
	"Jim", "Config", "Server", "Store", "Cache", "Basket", "Pantry", "Timer",
	"Recipe", "Ledger", "Bucket", "Garden", "Ticket", "Wallet", "Harbor",
	"Bridge", "Candle", "Docket", "Ember", "Falcon", "Reader", "Writer",
	"Baker", "Keeper", "Mover", "Porter", "Sifter", "Boiler", "Cooler",
	"Heater", "Mixer", "Jake", "Brew", "Steep", "Serve", "Fetch", "Parse",
	"Print", "Run", "Start", "Load", "Save", "Check", "Count", "Greet",
	"Pour", "Sort", "Trim",
}

var namesHeldout = []string{
	"Mosaic", "Nomad", "Osprey", "Pumice", "Quill", "Rowan", "Sorrel",
	"Topaz", "Umbra", "Vespa", "Quarry", "Saddle", "Tundra", "Velvet",
	"Walnut", "Notifier", "Quilter", "Roaster", "Tender", "Mend", "Whisk",
	"Roast", "Sift", "Tend", "Orbit", "Plumb", "Nudge", "Rinse",
}

var filesTrain = []string{
	"main.go", "jim/main.go", "store.go", "app/main.go", "tea.go",
	"shop/main.go", "util.go", "brew.go",
}

var filesHeldout = []string{
	"mosaic.go", "nest/main.go", "vault.go", "api/main.go", "kilo.go",
}

var fieldNamesTrain = []string{
	"Timeout", "Retries", "SugarKg", "Label", "Weight", "Count", "Price",
	"Name", "Size", "Color", "Speed", "Limit",
}

var fieldNamesHeldout = []string{
	"Volume", "Height", "Flavor", "Batch", "Crumb", "Glaze",
}

var fieldTypes = []string{
	"int", "string", "bool", "float64", "[]string", "[]int",
	"map[string]int", "time.Duration",
}

var caseValsTrain = []string{"rice", "mint", "basil", "retry", "pause", "turbo", "eco"}
var caseValsHeldout = []string{"clove", "sage", "rapid", "quiet"}

var keysTrain = []string{"tea", "coffee", "thanks", "welcome", "soup"}
var keysHeldout = []string{"cocoa", "cider", "salad", "hello"}

var tablesTrain = []string{"Recipes", "Greetings", "Prices", "Menu"}
var tablesHeldout = []string{"Catalog", "Baskets"}

var textsTrain = []string{
	"hello world", "the results", "all done", "starting now", "tea is ready",
	"good morning", "order up", "it works", "here we go", "task complete",
	"fresh bread", "soup is on", "brewing now", "ready or not",
}

var textsHeldout = []string{
	"dinner is served", "the kettle sings", "watch this space",
	"ready when you are", "supper is ready", "the oven timer",
	"mind the gap", "fresh out the oven",
}

var calleesTrain = []string{"Load", "Fetch", "Brew", "Parse", "Save", "Store.Load", "Cache.Fetch"}
var calleesHeldout = []string{"Whisk", "Roast", "Nest.Fetch", "Plumb"}

// ---------- concept book (mirrors data/concepts.json) ----------

type concept struct {
	id, kind, canonical string
	synonyms            []string // extra messy-input words for the concept
}

var statementConcepts = []concept{
	{"variable", "statement", "variable", []string{"var"}},
	{"constant", "statement", "constant", []string{"const"}},
	{"for-loop", "statement", "for loop", []string{"loop"}},
	{"range-loop", "statement", "range loop", []string{"for range", "range"}},
	{"if-else", "statement", "if statement", []string{"if else", "if"}},
	{"switch", "statement", "switch statement", []string{"switch"}},
	{"array", "statement", "array", nil},
	{"slice", "statement", "slice", nil},
	{"map", "statement", "map", []string{"dictionary"}},
	{"closure", "statement", "closure", []string{"anonymous function", "function value"}},
	{"pointers", "statement", "pointer", []string{"pointers"}},
	{"strings-runes", "statement", "string", []string{"strings and runes", "rune"}},
	{"errors", "statement", "error", []string{"new error", "errors"}},
	{"goroutine", "statement", "goroutine", []string{"go routine"}},
	{"channel", "statement", "channel", nil},
	{"buffered-channel", "statement", "buffered channel", nil},
	{"channel-sync", "statement", "channel synchronization", []string{"channel sync", "wait for a goroutine"}},
	{"select", "statement", "select statement", []string{"select", "non-blocking channel"}},
	{"timeout", "statement", "timeout", nil},
	{"close-channel", "statement", "close channel", []string{"closing a channel"}},
	{"channel-range", "statement", "channel range", []string{"range over a channel", "loop over a channel"}},
	{"timer", "statement", "timer", nil},
	{"ticker", "statement", "ticker", nil},
	{"waitgroup", "statement", "wait group", []string{"waitgroup"}},
	{"atomic-counter", "statement", "atomic counter", []string{"atomic"}},
	{"mutex", "statement", "mutex", nil},
	{"sorting", "statement", "sorting", []string{"sort"}},
	{"sorting-by", "statement", "custom sort", []string{"sort by function", "sort by"}},
	{"defer", "statement", "defer", []string{"deferred call"}},
	{"recover", "statement", "recover", []string{"panic recovery"}},
	{"string-functions", "statement", "string functions", nil},
	{"string-format", "statement", "string formatting", []string{"format a string", "sprintf"}},
	{"text-template", "statement", "text template", []string{"text templates"}},
	{"regexp", "statement", "regular expression", []string{"regex", "regexp"}},
	{"json", "statement", "json", []string{"marshal to json", "json encoding"}},
	{"xml", "statement", "xml", []string{"marshal to xml"}},
	{"time-now", "statement", "current time", []string{"time", "now"}},
	{"epoch", "statement", "epoch", []string{"unix time"}},
	{"time-format", "statement", "time formatting", []string{"format a time"}},
	{"random", "statement", "random number", []string{"random numbers", "random"}},
	{"number-parsing", "statement", "number parsing", []string{"parse numbers"}},
	{"url-parsing", "statement", "url parsing", []string{"parse a url"}},
	{"sha256", "statement", "sha256 hash", []string{"sha256", "hash"}},
	{"base64", "statement", "base64 encoding", []string{"base64"}},
	{"read-file", "statement", "file reading", []string{"read file", "reading a file"}},
	{"write-file", "statement", "file writing", []string{"write file", "writing a file"}},
	{"file-paths", "statement", "file path", []string{"file paths", "join paths"}},
	{"directories", "statement", "list a directory", []string{"read a directory", "list files"}},
	{"temp-file", "statement", "temporary file", []string{"temp file"}},
	{"log-line", "statement", "log line", []string{"log message", "logging"}},
	{"http-client", "statement", "http client", []string{"http get"}},
	{"context", "statement", "context", nil},
	{"spawn-process", "statement", "spawn a process", []string{"run a program", "external command"}},
	// print-line is deliberately NOT here: it is its own family
	// (the saying slot), generated below.
}

var declConcepts = []concept{
	{"struct", "declaration", "struct", []string{"struct type"}},
	{"interface", "declaration", "interface", nil},
	{"function", "declaration", "function", []string{"func"}},
	{"enum", "declaration", "enum", []string{"enumeration"}},
	{"embedding", "declaration", "embedded struct", []string{"struct embedding", "embedding"}},
	{"recursion", "declaration", "recursive function", []string{"recursion"}},
}

func article(name string) string {
	if name != "" && strings.ContainsRune("aeiou", rune(name[0])) {
		return "an"
	}
	return "a"
}

// pick returns a deterministic-random element.
func pick(r *rand.Rand, xs []string) string { return xs[r.Intn(len(xs))] }

func pickName(r *rand.Rand, pool []string, avoid ...string) string {
	for tries := 0; tries < 20; tries++ {
		n := pick(r, pool)
		ok := true
		for _, a := range avoid {
			if n == a {
				ok = false
			}
		}
		if ok {
			return n
		}
	}
	return pool[0]
}

// ---------- canonical builders ----------
//
// Every string built here must be accepted verbatim by the
// shipped parser named in the comment. Slot values keep the
// case the user typed; the model world lowercases, and the
// verbatim decoder restores copied words from the input.

func canonStmt(c concept, inFunc, file string) string {
	if inFunc != "" {
		return fmt.Sprintf("add %s %s in %s to %s", article(c.canonical), c.canonical, inFunc, file)
	}
	return fmt.Sprintf("add %s %s to %s", article(c.canonical), c.canonical, file)
}

func canonDecl(c concept, name, file, like string) string {
	s := fmt.Sprintf("add %s %s named %s", article(c.canonical), c.canonical, name)
	if like != "" {
		s += " like " + like
	}
	return s + " to " + file
}

func canonPrint(text, inFunc, file string) string {
	if inFunc != "" {
		return fmt.Sprintf("add a print line saying \"%s\" in %s to %s", text, inFunc, file)
	}
	return fmt.Sprintf("add a print line saying \"%s\" to %s", text, file)
}

func canonInstantiate(typ, fn, varName string) string {
	s := fmt.Sprintf("add instantiation of struct %s to handler %s", typ, fn)
	if varName != "" {
		s += " named " + varName
	}
	return s
}

func canonDelegate(file, callee, wrapper string) string {
	if wrapper != "" {
		return fmt.Sprintf("add a function %s in %s that calls %s", wrapper, file, callee)
	}
	return fmt.Sprintf("add a function in %s that calls %s", file, callee)
}

func canonCloneCopy(fn, newName string) string { return fmt.Sprintf("copy %s as %s", fn, newName) }

func canonCloneLike(fn, newName string) string {
	return fmt.Sprintf("add a function like %s named %s", fn, newName)
}

func canonServer(file string) string { return fmt.Sprintf("add a webserver to %s", file) }

func canonFieldA(strct, field, typ string) string {
	return fmt.Sprintf("add a %s field of type %s to %s", field, typ, strct)
}

func canonFieldB(strct, field, typ string) string {
	return fmt.Sprintf("add a %s field %s to %s", typ, field, strct)
}

func canonCaseA(value, fn string) string { return fmt.Sprintf("add a %s case to %s", value, fn) }
func canonCaseB(value, fn string) string { return fmt.Sprintf("add a case for %s to %s", value, fn) }

func canonRegisterA(key, table string) string { return fmt.Sprintf("register %s in %s", key, table) }
func canonRegisterB(key, table string) string {
	return fmt.Sprintf("add %s to the %s table", key, table)
}

// ---------- messy cores ----------
//
// A core is the ask in plain words — the concept named by any
// of its book names or synonyms, slots in a free order. Train
// and held-out cores are DIFFERENT code (disjoint template
// families); wrappers then dress a core into a full sentence.

type pools struct {
	names, files, fieldNames, caseVals, keys, tables, texts, callees []string
}

// ---------- generated random pools (attempt 2) ----------
//
// Syllable-built nonsense values, unioned into the train pools
// below. They carry no meaning to memorize — a name the model
// can only CARRY. Built from a fixed seed so the pools are a
// constant of the generator, and checked against every
// held-out pool: a generated value that collides with the
// judge's world is dropped, never emitted.

var genOnsets = []string{
	"b", "br", "c", "cr", "d", "dr", "f", "fl", "g", "gr", "h", "k", "kl",
	"l", "m", "n", "p", "pl", "pr", "qu", "r", "s", "sk", "st", "t", "tr",
	"v", "w", "z", "th", "sh", "mbr", "ndr",
}
var genVowels = []string{"a", "e", "i", "o", "u", "ai", "ea", "ou"}
var genCodas = []string{
	"", "b", "ck", "d", "f", "g", "k", "l", "lm", "m", "n", "nd", "nt",
	"p", "r", "rb", "rk", "rm", "rn", "rp", "s", "st", "t", "th", "v", "x", "z",
}

func genWord(r *rand.Rand) string {
	w := pick(r, genOnsets) + pick(r, genVowels) + pick(r, genCodas)
	if r.Intn(2) == 0 {
		w += pick(r, genVowels) + pick(r, genCodas)
	}
	return w
}

func genPool(r *rand.Rand, n int, cap bool, deny map[string]bool) []string {
	seen := map[string]bool{}
	var out []string
	for tries := 0; len(out) < n && tries < n*60; tries++ {
		w := genWord(r)
		if cap {
			w = strings.ToUpper(w[:1]) + w[1:]
		}
		if seen[w] || deny[w] || deny[strings.ToLower(w)] {
			continue
		}
		seen[w] = true
		out = append(out, w)
	}
	return out
}

func denySet(pools ...[]string) map[string]bool {
	out := map[string]bool{}
	for _, p := range pools {
		for _, s := range p {
			out[s] = true
			out[strings.ToLower(s)] = true
		}
	}
	return out
}

var (
	genNames      []string
	genFieldNames []string
	genTables     []string
	genKeys       []string
	genCaseVals   []string
	genFiles      []string
	genTexts      []string
)

func init() {
	r := rand.New(rand.NewSource(777))
	nameDeny := denySet(namesHeldout, fieldNamesHeldout, tablesHeldout, calleesHeldout, namesTrain, fieldNamesTrain, tablesTrain)
	genNames = genPool(r, 380, true, nameDeny)
	genFieldNames = genPool(r, 90, true, nameDeny)
	genTables = genPool(r, 60, true, nameDeny)
	wordDeny := denySet(keysHeldout, caseValsHeldout, keysTrain, caseValsTrain)
	genKeys = genPool(r, 70, false, wordDeny)
	genCaseVals = genPool(r, 70, false, wordDeny)
	fileDeny := denySet(filesHeldout)
	for len(genFiles) < 26 {
		w := genWord(r)
		f := w + ".go"
		if r.Intn(2) == 0 {
			f = genWord(r) + "/" + w + ".go"
		}
		if !fileDeny[f] && !fileDeny[strings.ToLower(f)] {
			fileDeny[f] = true
			genFiles = append(genFiles, f)
		}
	}
	// Print texts: recombinations of ordinary words — the
	// phrase as a whole is the slot, and held-out phrases stay
	// out (denySet covers them).
	textDeny := denySet(textsHeldout, textsTrain)
	leads := []string{"the", "a", "my", "this", "one"}
	mids := []string{"red", "old", "new", "big", "warm", "cold", "first", "last", "next", "best", "loud", "soft"}
	nouns := []string{"bell", "door", "lamp", "road", "song", "list", "name", "page", "line", "star", "wave", "stone", "garden", "market", "window", "bridge"}
	tails := []string{"is ready", "is here", "works", "again", "today", "now", "is done", "is open", "sings", "waits"}
	for len(genTexts) < 90 {
		t := pick(r, leads) + " " + pick(r, mids) + " " + pick(r, nouns) + " " + pick(r, tails)
		if !textDeny[t] {
			textDeny[t] = true
			genTexts = append(genTexts, t)
		}
	}
}

func unionPools(classic, gen []string) []string {
	out := append(append([]string{}, classic...), gen...)
	return out
}

func trainPools() pools {
	return pools{
		unionPools(namesTrain, genNames),
		unionPools(filesTrain, genFiles),
		unionPools(fieldNamesTrain, genFieldNames),
		unionPools(caseValsTrain, genCaseVals),
		unionPools(keysTrain, genKeys),
		unionPools(tablesTrain, genTables),
		unionPools(textsTrain, genTexts),
		calleesTrain,
	}
}

func heldoutPools() pools {
	return pools{namesHeldout, filesHeldout, fieldNamesHeldout, caseValsHeldout, keysHeldout, tablesHeldout, textsHeldout, calleesHeldout}
}

// conceptWord picks how the messy sentence names the concept:
// the canonical name, one of its book synonyms, or (train only)
// the bare display name again — variety the model must map to
// the one canonical.
func conceptWord(r *rand.Rand, c concept) string {
	words := append([]string{c.canonical}, c.synonyms...)
	return pick(r, words)
}

// buildStmtAsk makes one concept-statement ask with the given
// core dialect ("train" or "heldout").
func buildStmtAsk(r *rand.Rand, p pools, c concept, dialect string) ask {
	file := pick(r, p.files)
	inFunc := ""
	if r.Intn(2) == 0 {
		inFunc = pickName(r, p.names)
	}
	if inFunc == "" {
		inFunc = "" // file-only form: the planner's main rule
	}
	canon := canonStmt(c, inFunc, file)
	word := conceptWord(r, c)
	target := "to " + file
	if inFunc != "" {
		target = fmt.Sprintf("in %s to %s", inFunc, file)
	}
	var cores []func() string
	if dialect == "train" {
		cores = []func() string{
			func() string { return fmt.Sprintf("add %s %s %s", article(word), word, target) },
			func() string { return fmt.Sprintf("put %s %s %s", article(word), word, target) },
			func() string { return fmt.Sprintf("stick %s %s %s", article(word), word, target) },
			func() string { return fmt.Sprintf("add %s %s", target, word) }, // scrambled: target first
			func() string { return fmt.Sprintf("i need %s %s %s", article(word), word, target) },
			func() string {
				return fmt.Sprintf("add %s %s %s", article(word), word, strings.Replace(target, "to ", "into ", 1))
			},
		}
	} else {
		cores = []func() string{
			func() string { return fmt.Sprintf("place %s %s %s", article(word), word, target) },
			func() string { return fmt.Sprintf("insert %s %s %s", article(word), word, target) },
			func() string { return fmt.Sprintf("add %s %s %s", article(word), word, target) },
			func() string { return fmt.Sprintf("get %s %s %s", article(word), word, target) },
		}
	}
	slots := []string{file}
	if inFunc != "" {
		slots = []string{inFunc, file}
	}
	return ask{family: "concept-" + c.id, canonical: canon, slots: slots, cores: cores}
}

func buildDeclAsk(r *rand.Rand, p pools, c concept, dialect string) ask {
	name := pickName(r, p.names)
	file := pick(r, p.files)
	like := ""
	if c.id == "interface" && r.Intn(3) == 0 {
		like = pickName(r, p.names, name)
	}
	canon := canonDecl(c, name, file, like)
	word := conceptWord(r, c)
	var cores []func() string
	if dialect == "train" {
		cores = []func() string{
			func() string { return fmt.Sprintf("add %s %s named %s to %s", article(word), word, name, file) },
			func() string { return fmt.Sprintf("add %s %s called %s to %s", article(word), word, name, file) },
			func() string { return fmt.Sprintf("make %s %s named %s in %s", article(word), word, name, file) },
			func() string { return fmt.Sprintf("add new %s named %s to %s", word, name, file) },
			func() string { return fmt.Sprintf("i want %s %s named %s in %s", article(word), word, name, file) },
		}
	} else {
		cores = []func() string{
			func() string { return fmt.Sprintf("add %s %s named %s to %s", article(word), word, name, file) },
			func() string {
				return fmt.Sprintf("create %s %s with the name %s in %s", article(word), word, name, file)
			},
			func() string { return fmt.Sprintf("place %s %s named %s in %s", article(word), word, name, file) },
		}
	}
	if like != "" {
		// The like slot has one phrasing family of its own per
		// dialect; regenerate cores around it.
		base := cores
		cores = nil
		for _, b := range base {
			b := b
			cores = append(cores, func() string { return b() + " like " + like })
		}
	}
	slots := []string{name, file}
	if like != "" {
		slots = []string{name, like, file}
	}
	return ask{family: "concept-" + c.id, canonical: canon, slots: slots, cores: cores}
}

func buildPrintAsk(r *rand.Rand, p pools, dialect string) ask {
	text := pick(r, p.texts)
	file := pick(r, p.files)
	inFunc := ""
	if r.Intn(2) == 0 {
		inFunc = pickName(r, p.names)
	}
	canon := canonPrint(text, inFunc, file)
	where := "to " + file
	if inFunc != "" {
		where = fmt.Sprintf("in %s to %s", inFunc, file)
	}
	var cores []func() string
	if dialect == "train" {
		cores = []func() string{
			func() string { return fmt.Sprintf("add a print line saying \"%s\" %s", text, where) },
			func() string { return fmt.Sprintf("print \"%s\" %s", text, where) },
			func() string { return fmt.Sprintf("add a println that says \"%s\" %s", text, where) },
			func() string { return fmt.Sprintf("make it print \"%s\" %s", text, where) },
			func() string { return fmt.Sprintf("put a print statement saying \"%s\" %s", text, where) },
		}
	} else {
		cores = []func() string{
			func() string { return fmt.Sprintf("add a print line saying \"%s\" %s", text, where) },
			func() string { return fmt.Sprintf("have it print \"%s\" %s", text, where) },
			func() string { return fmt.Sprintf("insert a print of \"%s\" %s", text, where) },
			func() string { return fmt.Sprintf("log \"%s\" %s", text, where) },
		}
	}
	slots := []string{text, file}
	if inFunc != "" {
		slots = []string{text, inFunc, file}
	}
	return ask{family: "print", canonical: canon, slots: slots, cores: cores}
}

func buildInstantiateAsk(r *rand.Rand, p pools, dialect string) ask {
	typ := pickName(r, p.names)
	fn := pickName(r, p.names, typ)
	varName := ""
	if r.Intn(4) == 0 {
		varName = strings.ToLower(typ) + "y"
	}
	canon := canonInstantiate(typ, fn, varName)
	var cores []func() string
	if dialect == "train" {
		cores = []func() string{
			func() string { return fmt.Sprintf("add instantiation of struct %s to handler %s", typ, fn) },
			func() string { return fmt.Sprintf("add an instance of %s in %s", typ, fn) },
			func() string { return fmt.Sprintf("instantiate %s in %s", typ, fn) },
			func() string { return fmt.Sprintf("add a %s in %s", typ, fn) },
			func() string { return fmt.Sprintf("make a %s inside %s", typ, fn) },
			func() string { return fmt.Sprintf("put a new %s in %s", typ, fn) },
		}
	} else {
		cores = []func() string{
			func() string { return fmt.Sprintf("add instantiation of struct %s to handler %s", typ, fn) },
			func() string { return fmt.Sprintf("create an instance of %s inside %s", typ, fn) },
			func() string { return fmt.Sprintf("place a %s in %s", typ, fn) },
			func() string { return fmt.Sprintf("add one %s to %s", typ, fn) },
		}
	}
	if varName != "" {
		base := cores
		cores = nil
		for _, b := range base {
			b := b
			cores = append(cores, func() string { return b() + " named " + varName })
		}
	}
	slots := []string{typ, fn}
	if varName != "" {
		slots = append(slots, varName)
	}
	return ask{family: "instantiate", canonical: canon, slots: slots, cores: cores}
}

func buildDelegateAsk(r *rand.Rand, p pools, dialect string) ask {
	file := pick(r, p.files)
	callee := pick(r, p.callees)
	wrapper := ""
	if r.Intn(2) == 0 {
		wrapper = pickName(r, p.names)
	}
	canon := canonDelegate(file, callee, wrapper)
	var cores []func() string
	if dialect == "train" {
		cores = []func() string{
			func() string { return fmt.Sprintf("add a function in %s that calls %s", file, callee) },
			func() string { return fmt.Sprintf("wrap %s in %s", callee, file) },
			func() string { return fmt.Sprintf("add a wrapper in %s calling %s", file, callee) },
			func() string { return fmt.Sprintf("make a function in %s that calls %s", file, callee) },
		}
	} else {
		cores = []func() string{
			func() string { return fmt.Sprintf("add a function in %s that calls %s", file, callee) },
			func() string { return fmt.Sprintf("add a helper in %s which calls %s", file, callee) },
			func() string { return fmt.Sprintf("place a function in %s calling %s", file, callee) },
		}
	}
	if wrapper != "" {
		// Every core must carry the wrapper name too — an
		// anonymous core under a named canonical would teach
		// the model to invent names.
		cores = []func() string{
			func() string { return fmt.Sprintf("add a function %s in %s that calls %s", wrapper, file, callee) },
			func() string { return fmt.Sprintf("add a wrapper named %s in %s that calls %s", wrapper, file, callee) },
		}
	}
	slots := []string{file, callee}
	if wrapper != "" {
		slots = []string{wrapper, file, callee}
	}
	return ask{family: "delegate", canonical: canon, slots: slots, cores: cores}
}

func buildCloneAsk(r *rand.Rand, p pools, dialect string) ask {
	fn := pickName(r, p.names)
	newName := pickName(r, p.names, fn)
	like := r.Intn(2) == 0
	canon := canonCloneCopy(fn, newName)
	family := "clone"
	if like {
		canon = canonCloneLike(fn, newName)
	}
	var cores []func() string
	if dialect == "train" {
		cores = []func() string{
			func() string { return fmt.Sprintf("copy %s as %s", fn, newName) },
			func() string { return fmt.Sprintf("clone %s as %s", fn, newName) },
			func() string { return fmt.Sprintf("duplicate %s and call it %s", fn, newName) },
			func() string { return fmt.Sprintf("add a function like %s named %s", fn, newName) },
		}
	} else {
		cores = []func() string{
			func() string { return fmt.Sprintf("copy %s as %s", fn, newName) },
			func() string { return fmt.Sprintf("make a copy of %s called %s", fn, newName) },
			func() string { return fmt.Sprintf("clone %s into %s", fn, newName) },
		}
	}
	return ask{family: family, canonical: canon, slots: []string{fn, newName}, cores: cores}
}

func buildServerAsk(r *rand.Rand, p pools, dialect string) ask {
	file := pick(r, p.files)
	canon := canonServer(file)
	var cores []func() string
	if dialect == "train" {
		cores = []func() string{
			func() string { return fmt.Sprintf("add a webserver to %s", file) },
			func() string { return fmt.Sprintf("add a web server to %s", file) },
			func() string { return fmt.Sprintf("put an http server in %s", file) },
			func() string { return fmt.Sprintf("add the webserver to %s", file) },
		}
	} else {
		cores = []func() string{
			func() string { return fmt.Sprintf("add a webserver to %s", file) },
			func() string { return fmt.Sprintf("place a web server in %s", file) },
			func() string { return fmt.Sprintf("insert an http server into %s", file) },
		}
	}
	return ask{family: "add-server", canonical: canon, slots: []string{file}, cores: cores}
}

func buildFieldAsk(r *rand.Rand, p pools, dialect string) ask {
	strct := pickName(r, p.names)
	field := pick(r, p.fieldNames)
	typ := pick(r, fieldTypes)
	formB := r.Intn(2) == 0
	canon := canonFieldA(strct, field, typ)
	if formB {
		canon = canonFieldB(strct, field, typ)
	}
	var cores []func() string
	if dialect == "train" {
		// Every core must carry EVERY slot: a core without the
		// type would teach the model to invent one.
		cores = []func() string{
			func() string { return fmt.Sprintf("add a %s field of type %s to %s", field, typ, strct) },
			func() string { return fmt.Sprintf("add a %s field %s to %s", typ, field, strct) },
			func() string { return fmt.Sprintf("give %s a %s field of type %s", strct, field, typ) },
		}
	} else {
		cores = []func() string{
			func() string { return fmt.Sprintf("add a %s field of type %s to %s", field, typ, strct) },
			func() string { return fmt.Sprintf("add a %s field %s to %s", typ, field, strct) },
			func() string { return fmt.Sprintf("place a %s field in %s", field, strct) },
		}
	}
	return ask{family: "add-field", canonical: canon, slots: []string{field, typ, strct}, cores: cores}
}

func buildCaseAsk(r *rand.Rand, p pools, dialect string) ask {
	value := pick(r, p.caseVals)
	fn := pickName(r, p.names)
	formB := r.Intn(2) == 0
	canon := canonCaseA(value, fn)
	if formB {
		canon = canonCaseB(value, fn)
	}
	var cores []func() string
	if dialect == "train" {
		cores = []func() string{
			func() string { return fmt.Sprintf("add a %s case to %s", value, fn) },
			func() string { return fmt.Sprintf("add a case for %s to %s", value, fn) },
			func() string { return fmt.Sprintf("in %s also handle %s", fn, value) },
			func() string { return fmt.Sprintf("make %s handle %s too", fn, value) },
		}
	} else {
		cores = []func() string{
			func() string { return fmt.Sprintf("add a %s case to %s", value, fn) },
			func() string { return fmt.Sprintf("add a case for %s to %s", value, fn) },
			func() string { return fmt.Sprintf("let %s handle %s as well", fn, value) },
		}
	}
	return ask{family: "add-case", canonical: canon, slots: []string{value, fn}, cores: cores}
}

func buildRegisterAsk(r *rand.Rand, p pools, dialect string) ask {
	key := pick(r, p.keys)
	table := pick(r, p.tables)
	formB := r.Intn(2) == 0
	canon := canonRegisterA(key, table)
	if formB {
		canon = canonRegisterB(key, table)
	}
	var cores []func() string
	if dialect == "train" {
		cores = []func() string{
			func() string { return fmt.Sprintf("register %s in %s", key, table) },
			func() string { return fmt.Sprintf("add %s to the %s table", key, table) },
			func() string { return fmt.Sprintf("put %s in %s", key, table) },
		}
	} else {
		cores = []func() string{
			func() string { return fmt.Sprintf("register %s in %s", key, table) },
			func() string { return fmt.Sprintf("add %s to the %s table", key, table) },
			func() string { return fmt.Sprintf("place %s into %s", key, table) },
		}
	}
	return ask{family: "register", canonical: canon, slots: []string{key, table}, cores: cores}
}

// ---------- wrappers ----------
//
// A wrapper dresses a core into a full sentence. Train and
// held-out wrapper families are disjoint sets (the leak check
// compares family ids). A wrapper's id is recorded as the
// pair's template: "core<k>+<wrapper>".

type wrapper struct {
	id string
	fn func(core string) string
}

var trainWrappers = []wrapper{
	{"bare", func(c string) string { return c }},
	{"bare-period", func(c string) string { return c + "." }},
	{"please-pre", func(c string) string { return "please " + c }},
	{"please-post", func(c string) string { return c + " please" }},
	{"can-you", func(c string) string { return "can you " + c }},
	{"could-you", func(c string) string { return "could you " + c + " for me" }},
	{"i-need", func(c string) string { return "i need you to " + c }},
	{"i-want", func(c string) string { return "i want to " + c }},
	{"just", func(c string) string { return "just " + c }},
	{"quick", func(c string) string { return "quick, " + c }},
	{"typo-please", func(c string) string { return "plese " + c }},
	{"hey", func(c string) string { return "hey, " + c }},
}

var heldoutWrappers = []wrapper{
	{"would-you-mind", func(c string) string { return "would you mind " + gerund(c) }},
	{"lets", func(c string) string { return "let's " + c }},
	{"id-like", func(c string) string { return "i'd like to " + c }},
	{"go-ahead", func(c string) string { return "go ahead and " + c }},
	{"hey-gollemer", func(c string) string { return "hey gollemer, " + c }},
	{"can-we-get", func(c string) string { return "can we " + c }},
	{"kindly", func(c string) string { return "kindly " + c }},
	{"when-you-can", func(c string) string { return c + " when you can" }},
}

// gerund turns a core's leading verb into its -ing form for
// "would you mind ___" — only the simple leading verbs the
// held-out cores use get a rule; anything else passes through.
func gerund(core string) string {
	fields := strings.Fields(core)
	if len(fields) == 0 {
		return core
	}
	switch fields[0] {
	case "add":
		fields[0] = "adding"
	case "place":
		fields[0] = "placing"
	case "insert":
		fields[0] = "inserting"
	case "get":
		fields[0] = "getting"
	case "create":
		fields[0] = "creating"
	case "register":
		fields[0] = "registering"
	case "copy":
		fields[0] = "copying"
	case "have":
		fields[0] = "having"
	}
	return strings.Join(fields, " ")
}

// dress picks one core and one wrapper and renders the messy
// input; the template id names both halves of the family.
func dress(r *rand.Rand, a ask, wrappers []wrapper) (input, template string) {
	ci := r.Intn(len(a.cores))
	core := a.cores[ci]()
	w := wrappers[r.Intn(len(wrappers))]
	return w.fn(core), fmt.Sprintf("core%d+%s", ci, w.id)
}

// dressIdx renders one ask with a CHOSEN core and wrapper —
// the twins pass's tool (v2): factor crossing by construction.
func dressIdx(a ask, ci, wi int, wrappers []wrapper) (input, template string) {
	w := wrappers[wi%len(wrappers)]
	return w.fn(a.cores[ci%len(a.cores)]()), fmt.Sprintf("core%d+%s", ci%len(a.cores), w.id)
}

// ---------- the none class ----------
//
// Lines the layer must never claim: generation asks (they are
// the brains' and the boundary's), concept questions, social
// lines, toolchain asks, and write asks no shipped shape
// serves. The model answers these with the single word "none".

func noneInputs(r *rand.Rand, p pools, dialect string) []string {
	var out []string
	// Generation asks: verb x job x thing — the gocode family's
	// home turf, and the boundary's pinned shapes. The two
	// dialects use disjoint verb/job sets so no fixed sentence
	// can appear in both splits.
	verbs := []string{"write", "give me", "code", "create", "make"}
	jobs := []string{"reverses", "sorts", "parses", "counts the words in", "doubles", "flattens"}
	things := []string{"a string", "a slice of ints", "the input", "a list of names", "a sentence"}
	extras := []string{
		"write a program that prints the prime numbers",
		"give me a function that checks for palindromes",
		"code a function to compute a factorial",
		"write a function that reverses a string",
		"add a function that sorts numbers",
	}
	questions := []string{
		"what is a goroutine", "what is a channel", "how do channels work",
		"why does go have no classes", "explain defer to me",
		"what does go vet do", "how do i run the tests",
		"what is the difference between a slice and an array",
		"what is an interface", "how does a map work",
		"when should i use a pointer", "what does fmt.Println do",
	}
	social := []string{
		"hello", "hi there", "how are you today", "good morning",
		"thanks for the help", "tell me a joke", "who made you",
		"goodbye for now", "that is great", "nice one",
	}
	toolchain := []string{
		"run the tests", "build the project", "what does make chat do",
		"test my code", "check if the dependencies are updated",
		"format the code", "show me the commands",
	}
	if dialect == "heldout" {
		verbs = []string{"write me", "build me", "draft"}
		jobs = []string{"merges", "splits", "reverses the words in", "capitalizes", "trims"}
		things = []string{"two slices", "a paragraph", "the greeting", "a csv row"}
		extras = []string{
			"could you write a function that flattens a slice",
			"write a function that merges two maps",
			"give me code that reads a csv file",
		}
		questions = []string{
			"what exactly is an interface", "how do waitgroups work",
			"why is there no ternary operator", "explain the select statement",
			"what does the underscore mean in go", "how are errors handled",
		}
		social = []string{
			"hiya, how is it going", "morning all", "thanks a bunch",
			"see you tomorrow", "that sounds lovely",
		}
		toolchain = []string{
			"run the benchmarks", "tidy the modules",
			"what go version is installed", "list the dependencies",
		}
	}
	for _, v := range verbs {
		for _, j := range jobs {
			out = append(out, fmt.Sprintf("%s a function that %s %s", v, j, pick(r, things)))
		}
	}
	out = append(out, extras...)
	out = append(out, questions...)
	out = append(out, social...)
	out = append(out, toolchain...)
	// Unservable write asks: an add verb and a real object, but
	// no shipped shape assembles them.
	unservable := []string{
		"add a getter to %s", "add error handling to %s",
		"add a retry loop to %s", "make %s faster",
		"add a setter for %s", "add logging to %s",
		"add a constructor to %s", "add validation to %s",
	}
	for _, name := range p.names {
		for _, u := range unservable {
			if r.Intn(3) == 0 {
				out = append(out, fmt.Sprintf(u, name))
			}
		}
	}
	if dialect == "heldout" {
		out = append(out,
			"could you write a function that flattens a slice",
			"what exactly is an interface",
			"hiya, how is it going",
			"please add some comments to the code",
			"add logging everywhere",
		)
	}
	return out
}

// ---------- compounds ----------
//
// Two asks glued into one breath. The canonical target joins
// the two canonicals with " ## ". Pairings are named
// "<fam1>+<fam2>"; train pairings and held-out unseen
// pairings are disjoint SETS of family pairs.

var trainGlues = []struct {
	id string
	fn func(a, b string) string
}{
	{"and", func(a, b string) string { return a + " and " + b }},
	{"and-then", func(a, b string) string { return a + " and then " + b }},
	{"comma-and", func(a, b string) string { return a + ", and " + b }},
	{"also", func(a, b string) string { return a + " and also " + b }},
	{"two-sentences", func(a, b string) string { return a + ". " + b }},
	{"have-it", func(a, b string) string { return a + " and have it " + b }},
}

var heldoutGlues = []struct {
	id string
	fn func(a, b string) string
}{
	{"plus", func(a, b string) string { return a + " plus " + b }},
	{"after-that", func(a, b string) string { return a + " and after that " + b }},
	{"and-next", func(a, b string) string { return a + " and next " + b }},
	{"then-have-it", func(a, b string) string { return a + " then have it " + b }},
}

// compoundAsk pairs two asks. The piece cores are drawn from
// each ask's own core list; "have it" glues only fit a print
// second piece, which the pairing table guarantees. It also
// returns the two piece cores as a glue split would leave
// them — arm B's fragment examples (v2).
func compoundAsk(r *rand.Rand, a, b ask, glues []struct {
	id string
	fn func(a, b string) string
}) (input, output, template, coreA, coreB string) {
	ca := a.cores[r.Intn(len(a.cores))]()
	cb := b.cores[r.Intn(len(b.cores))]()
	g := glues[r.Intn(len(glues))]
	if g.id == "have-it" || g.id == "then-have-it" {
		// "have it print ..." — the second core must be a print
		// core in pronoun form; rebuild it from the print ask's
		// slots (text, inFunc, file ride in the canonical).
		cb = haveItPrintCore(b)
	}
	return g.fn(ca, cb), a.canonical + " ## " + b.canonical, "glue:" + g.id, ca, cb
}

// haveItPrintCore renders a print ask's second-piece core in
// the "print ..." tail form that follows a "have it" glue. The
// print ask's canonical carries the slots; recover them from
// its slot list (text first).
func haveItPrintCore(b ask) string {
	text := b.slots[0]
	return fmt.Sprintf("print \"%s\"", text)
}

// ---------- emittability ----------
//
// The verbatim decoder keeps the Predict discipline within a
// piece (no immediate repeat, no repeated bigram) and resets
// it at the "##" separator. A gold target that breaks the rule
// inside a piece can never be produced: the generator refuses
// to emit it (the decl pipeline's standing rule for targets).

func emittable(target string) bool {
	for _, piece := range strings.Split(target, " ## ") {
		toks := splitTokens(piece)
		seen := map[[2]string]bool{}
		for i := 1; i < len(toks); i++ {
			if toks[i] == toks[i-1] {
				return false
			}
			pr := [2]string{toks[i-1], toks[i]}
			if seen[pr] {
				return false
			}
			seen[pr] = true
		}
	}
	return true
}

// splitTokens mirrors the trainer's tokenizeLower closely
// enough for the emittability rule: whitespace words with the
// structural punctuation as single tokens.
func splitTokens(s string) []string {
	var toks []string
	for _, f := range strings.Fields(strings.ToLower(s)) {
		var cur strings.Builder
		for _, r := range f {
			if strings.ContainsRune(`{}[],:"?!`, r) {
				if cur.Len() > 0 {
					toks = append(toks, cur.String())
					cur.Reset()
				}
				toks = append(toks, string(r))
			} else {
				cur.WriteRune(r)
			}
		}
		if cur.Len() > 0 {
			toks = append(toks, cur.String())
		}
	}
	return toks
}

func normInput(s string) string {
	return strings.Join(strings.Fields(strings.ToLower(s)), " ")
}

// ---------- coordinated compound builders ----------

// buildPrintAskShared makes a print ask whose statement lands
// in fn inside main.go — the compound convention (a handler
// piece's file is main.go unless a piece names another).
func buildPrintAskShared(r *rand.Rand, p pools, dialect, fn string) ask {
	text := pick(r, p.texts)
	canon := canonPrint(text, fn, "main.go")
	where := fmt.Sprintf("in %s to main.go", fn)
	var cores []func() string
	if dialect == "train" {
		cores = []func() string{
			func() string { return fmt.Sprintf("add a print line saying \"%s\" %s", text, where) },
			func() string { return fmt.Sprintf("print \"%s\" %s", text, where) },
			func() string { return fmt.Sprintf("make it print \"%s\" %s", text, where) },
		}
	} else {
		cores = []func() string{
			func() string { return fmt.Sprintf("add a print line saying \"%s\" %s", text, where) },
			func() string { return fmt.Sprintf("have it print \"%s\" %s", text, where) },
		}
	}
	return ask{family: "print", canonical: canon, slots: []string{text, fn, "main.go"}, cores: cores}
}

// buildInstantiateAskTyped makes an instantiate ask for a
// type chosen by the caller (the decl+instantiate pairing:
// declare the struct, then instantiate the very same name).
func buildInstantiateAskTyped(r *rand.Rand, p pools, dialect, typ string) ask {
	fn := pickName(r, p.names, typ)
	canon := canonInstantiate(typ, fn, "")
	var cores []func() string
	if dialect == "train" {
		cores = []func() string{
			func() string { return fmt.Sprintf("add instantiation of struct %s to handler %s", typ, fn) },
			func() string { return fmt.Sprintf("add a %s in %s", typ, fn) },
			func() string { return fmt.Sprintf("make a %s inside %s", typ, fn) },
		}
	} else {
		cores = []func() string{
			func() string { return fmt.Sprintf("add instantiation of struct %s to handler %s", typ, fn) },
			func() string { return fmt.Sprintf("place a %s in %s", typ, fn) },
		}
	}
	return ask{family: "instantiate", canonical: canon, slots: []string{typ, fn}, cores: cores}
}

type pairing struct {
	id    string
	build func(r *rand.Rand, p pools, dialect string) (ask, ask)
}

func conceptByID(id string) concept {
	for _, c := range append(append([]concept{}, statementConcepts...), declConcepts...) {
		if c.id == id {
			return c
		}
	}
	return concept{}
}

var trainPairings = []pairing{
	{"instantiate+print", func(r *rand.Rand, p pools, d string) (ask, ask) {
		a := buildInstantiateAsk(r, p, d)
		return a, buildPrintAskShared(r, p, d, a.slots[1])
	}},
	{"print+instantiate", func(r *rand.Rand, p pools, d string) (ask, ask) {
		return buildPrintAsk(r, p, d), buildInstantiateAsk(r, p, d)
	}},
	{"decl+print", func(r *rand.Rand, p pools, d string) (ask, ask) {
		a := buildDeclAsk(r, p, declConcepts[r.Intn(len(declConcepts))], d)
		return a, buildPrintAskShared(r, p, d, "main")
	}},
	{"decl+instantiate", func(r *rand.Rand, p pools, d string) (ask, ask) {
		a := buildDeclAsk(r, p, conceptByID("struct"), d)
		return a, buildInstantiateAskTyped(r, p, d, a.slots[0])
	}},
	{"stmt+stmt", func(r *rand.Rand, p pools, d string) (ask, ask) {
		c1 := statementConcepts[r.Intn(len(statementConcepts))]
		c2 := statementConcepts[r.Intn(len(statementConcepts))]
		for c2.id == c1.id {
			c2 = statementConcepts[r.Intn(len(statementConcepts))]
		}
		return buildStmtAsk(r, p, c1, d), buildStmtAsk(r, p, c2, d)
	}},
	{"print+print", func(r *rand.Rand, p pools, d string) (ask, ask) {
		return buildPrintAsk(r, p, d), buildPrintAsk(r, p, d)
	}},
	{"field+print", func(r *rand.Rand, p pools, d string) (ask, ask) {
		return buildFieldAsk(r, p, d), buildPrintAskShared(r, p, d, "main")
	}},
}

var heldoutPairings = []pairing{
	{"server+print", func(r *rand.Rand, p pools, d string) (ask, ask) {
		return buildServerAsk(r, p, d), buildPrintAskShared(r, p, d, "main")
	}},
	{"clone+print", func(r *rand.Rand, p pools, d string) (ask, ask) {
		return buildCloneAsk(r, p, d), buildPrintAskShared(r, p, d, "main")
	}},
	{"delegate+print", func(r *rand.Rand, p pools, d string) (ask, ask) {
		return buildDelegateAsk(r, p, d), buildPrintAskShared(r, p, d, "main")
	}},
	{"case+register", func(r *rand.Rand, p pools, d string) (ask, ask) {
		return buildCaseAsk(r, p, d), buildRegisterAsk(r, p, d)
	}},
	{"decl+field", func(r *rand.Rand, p pools, d string) (ask, ask) {
		return buildDeclAsk(r, p, declConcepts[r.Intn(len(declConcepts))], d), buildFieldAsk(r, p, d)
	}},
}

// ---------- assembly ----------

type genReport struct {
	Seed               int64          `json:"seed"`
	TrainPairs         int            `json:"train_pairs"`
	TrainByFamily      map[string]int `json:"train_by_family"`
	Fragments          int            `json:"fragments"`
	SkippedUnemittable int            `json:"skipped_unemittable_targets"`
	ConflictingInputs  int            `json:"conflicting_inputs_dropped"`
	DuplicateInputs    int            `json:"duplicate_inputs_dropped"`
	TemplateLeak       int            `json:"template_family_leak"`
	SlotLeak           []string       `json:"slot_value_leak_beyond_fixtures"`
	InputOverlap       int            `json:"train_heldout_input_overlap"`
	JudgeItems         int            `json:"judge_items_read_for_checks"`
}

type frag struct {
	Input    string `json:"input"`
	Family   string `json:"family"`
	Template string `json:"template"`
}

func main() {
	outDir := flag.String("out", "data/training", "output directory")
	seed := flag.Int64("seed", 20261031, "generator seed")
	judgePath := flag.String("judge", "data/training/phrasing_heldout.jsonl", "the fixed held-out judge, read for leak checks only")
	flag.Parse()
	r := rand.New(rand.NewSource(*seed))
	rep := &genReport{
		Seed:          *seed,
		TrainByFamily: map[string]int{},
	}

	trainSeen := map[string]string{} // normalized input -> output
	var train []pair
	var fragments []frag
	fragSeen := map[string]bool{}
	addFrag := func(input, family, template string) {
		n := normInput(input) + "|" + family
		if fragSeen[n] {
			return
		}
		fragSeen[n] = true
		fragments = append(fragments, frag{Input: input, Family: family, Template: template})
	}
	addTrain := func(p pair) {
		if !emittable(p.Output) {
			rep.SkippedUnemittable++
			return
		}
		n := normInput(p.Input)
		if prev, dup := trainSeen[n]; dup {
			if prev != p.Output {
				rep.ConflictingInputs++
				for i, q := range train {
					if normInput(q.Input) == n {
						train = append(train[:i], train[i+1:]...)
						break
					}
				}
				delete(trainSeen, n)
			} else {
				rep.DuplicateInputs++
			}
			return
		}
		trainSeen[n] = p.Output
		train = append(train, p)
		rep.TrainByFamily[p.Family]++
	}

	tp := trainPools()
	emit := func(build func() ask, count int) {
		for i := 0; i < count; i++ {
			a := build()
			in, tmpl := dress(r, a, trainWrappers)
			addTrain(pair{Input: in, Output: a.canonical, Domain: "phrasing",
				Family: a.family, Template: "train/" + tmpl, Slots: a.slots})
			addFrag(in, a.family, "train/"+tmpl)
		}
	}
	// emitTwins is the factor-crossing pass: for each group,
	// one ask is dressed in EVERY core under one wrapper and in
	// one core under EVERY wrapper, and a second ask (fresh
	// slot values, same builder) is dressed under the group's
	// first core+wrapper — minimal pairs on each factor.
	emitTwins := func(build func() ask, groups int) {
		for g := 0; g < groups; g++ {
			a := build()
			wi := r.Intn(len(trainWrappers))
			for ci := range a.cores {
				in, tmpl := dressIdx(a, ci, wi, trainWrappers)
				addTrain(pair{Input: in, Output: a.canonical, Domain: "phrasing",
					Family: a.family, Template: "train/twin/" + tmpl, Slots: a.slots})
			}
			ci := r.Intn(len(a.cores))
			for wj := range trainWrappers {
				in, tmpl := dressIdx(a, ci, wj, trainWrappers)
				addTrain(pair{Input: in, Output: a.canonical, Domain: "phrasing",
					Family: a.family, Template: "train/twin/" + tmpl, Slots: a.slots})
			}
			a2 := build()
			in, tmpl := dressIdx(a2, ci, wi, trainWrappers)
			addTrain(pair{Input: in, Output: a2.canonical, Domain: "phrasing",
				Family: a2.family, Template: "train/twin/" + tmpl, Slots: a2.slots})
		}
	}

	// Singles at attempt-2 density. Concept rows: v1's 16 ->
	// ~250 each (base + twins); the rest scale ~5x.
	for _, c := range statementConcepts {
		c := c
		b := func() ask { return buildStmtAsk(r, tp, c, "train") }
		emit(b, 195)
		emitTwins(b, 3)
	}
	for _, c := range declConcepts {
		c := c
		b := func() ask { return buildDeclAsk(r, tp, c, "train") }
		emit(b, 250)
		emitTwins(b, 3)
	}
	emit(func() ask { return buildPrintAsk(r, tp, "train") }, 1150)
	emitTwins(func() ask { return buildPrintAsk(r, tp, "train") }, 6)
	emit(func() ask { return buildInstantiateAsk(r, tp, "train") }, 1350)
	emitTwins(func() ask { return buildInstantiateAsk(r, tp, "train") }, 8)
	emit(func() ask { return buildDelegateAsk(r, tp, "train") }, 850)
	emitTwins(func() ask { return buildDelegateAsk(r, tp, "train") }, 4)
	emit(func() ask { return buildCloneAsk(r, tp, "train") }, 600)
	emitTwins(func() ask { return buildCloneAsk(r, tp, "train") }, 3)
	emit(func() ask { return buildServerAsk(r, tp, "train") }, 450)
	emit(func() ask { return buildFieldAsk(r, tp, "train") }, 1000)
	emitTwins(func() ask { return buildFieldAsk(r, tp, "train") }, 5)
	emit(func() ask { return buildCaseAsk(r, tp, "train") }, 750)
	emitTwins(func() ask { return buildCaseAsk(r, tp, "train") }, 3)
	emit(func() ask { return buildRegisterAsk(r, tp, "train") }, 650)
	emitTwins(func() ask { return buildRegisterAsk(r, tp, "train") }, 3)

	// None pairs: each generated line bare and under three
	// wrappers (wrapper-invariance for the none class, denser
	// than v1's single extra dressing).
	noneWrap := []wrapper{trainWrappers[0], trainWrappers[2], trainWrappers[4], trainWrappers[5], trainWrappers[8]}
	// The none generator's unservable loop scales with the
	// (now much larger) name pool; subsample its lines so the
	// class stays near v1's share of the corpus (~4%).
	noneLines := noneInputs(r, tp, "train")
	r.Shuffle(len(noneLines), func(i, j int) { noneLines[i], noneLines[j] = noneLines[j], noneLines[i] })
	if len(noneLines) > 260 {
		noneLines = noneLines[:260]
	}
	for _, line := range noneLines {
		addTrain(pair{Input: line, Output: "none", Domain: "phrasing", Family: "none", Template: "train/none-bare"})
		addFrag(line, "none", "train/none-bare")
		for k := 0; k < 3; k++ {
			w := noneWrap[r.Intn(len(noneWrap))]
			in := w.fn(line)
			addTrain(pair{Input: in, Output: "none", Domain: "phrasing", Family: "none", Template: "train/none+" + w.id})
			addFrag(in, "none", "train/none+"+w.id)
		}
	}

	// Compounds at ~3.5x v1, same seen pairings, train glues.
	glueSet := func(b ask) []struct {
		id string
		fn func(a, b string) string
	} {
		if b.family == "print" {
			return trainGlues
		}
		var out []struct {
			id string
			fn func(a, b string) string
		}
		for _, g := range trainGlues {
			if g.id != "have-it" {
				out = append(out, g)
			}
		}
		return out
	}
	for _, pr := range trainPairings {
		for i := 0; i < 500; i++ {
			a, b := pr.build(r, tp, "train")
			in, out, tmpl, ca, cb := compoundAsk(r, a, b, glueSet(b))
			if r.Intn(4) == 0 {
				w := trainWrappers[2+r.Intn(4)] // please/can-you family
				in = w.fn(in)
				tmpl += "+" + w.id
			}
			addTrain(pair{Input: in, Output: out, Domain: "phrasing",
				Family: "compound:" + pr.id, Template: "train/" + tmpl,
				Slots: append(append([]string{}, a.slots...), b.slots...)})
			// Fragments: the piece cores a glue split leaves.
			addFrag(ca, a.family, "train/frag-a")
			addFrag(cb, b.family, "train/frag-b")
			if r.Intn(4) == 0 {
				w := trainWrappers[r.Intn(len(trainWrappers))]
				addFrag(w.fn(ca), a.family, "train/frag-a+"+w.id)
			}
		}
	}
	rep.TrainPairs = len(train)
	rep.Fragments = len(fragments)

	// ---------- leak checks against the fixed judge ----------
	judgeInputs := map[string]bool{}
	judgeFile, err := os.Open(*judgePath)
	if err != nil {
		fail(fmt.Errorf("read judge for leak checks: %w", err))
	}
	sc := bufioScanner(judgeFile)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var it heldItem
		if err := json.Unmarshal([]byte(line), &it); err != nil {
			fail(err)
		}
		judgeInputs[normInput(it.Input)] = true
		rep.JudgeItems++
	}
	judgeFile.Close()

	heldoutWrapperIDs := map[string]bool{}
	for _, w := range heldoutWrappers {
		heldoutWrapperIDs[w.id] = true
	}
	heldoutGlueIDs := map[string]bool{}
	for _, g := range heldoutGlues {
		heldoutGlueIDs[g.id] = true
	}
	for _, p := range train {
		for id := range heldoutWrapperIDs {
			if strings.Contains(p.Template, id) {
				rep.TemplateLeak++
			}
		}
		for id := range heldoutGlueIDs {
			if strings.Contains(p.Template, "glue:"+id) {
				rep.TemplateLeak++
			}
		}
		if judgeInputs[normInput(p.Input)] {
			rep.InputOverlap++
		}
	}
	// Slot pools: every held-out pool value (minus the grammar
	// fixtures both splits share by design — main/main.go and
	// the closed field-type vocabulary) must be absent from
	// the train slot values.
	fixtures := map[string]bool{"main": true, "main.go": true}
	for _, t := range fieldTypes {
		fixtures[t] = true
	}
	heldSlotVals := map[string]bool{}
	for _, s := range append(append(append(append(append(append(append(append(
		[]string{}, namesHeldout...), filesHeldout...), fieldNamesHeldout...),
		caseValsHeldout...), keysHeldout...), tablesHeldout...), textsHeldout...), calleesHeldout...) {
		heldSlotVals[s] = true
	}
	leakSet := map[string]bool{}
	for _, p := range train {
		for _, s := range p.Slots {
			if heldSlotVals[s] && !fixtures[s] {
				leakSet[s] = true
			}
		}
	}
	for s := range leakSet {
		rep.SlotLeak = append(rep.SlotLeak, s)
	}
	sort.Strings(rep.SlotLeak)
	if rep.TemplateLeak > 0 || len(rep.SlotLeak) > 0 || rep.InputOverlap > 0 {
		fail(fmt.Errorf("leak check failed: template=%d slot=%v input-overlap=%d",
			rep.TemplateLeak, rep.SlotLeak, rep.InputOverlap))
	}

	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		fail(err)
	}
	if err := writeJSONL(filepath.Join(*outDir, "phrasing_pairs2.jsonl"), train); err != nil {
		fail(err)
	}
	if err := writeJSONL(filepath.Join(*outDir, "phrasing_fragments2.jsonl"), fragments); err != nil {
		fail(err)
	}
	raw, err := json.MarshalIndent(rep, "", "  ")
	if err != nil {
		fail(err)
	}
	if err := os.WriteFile(filepath.Join(*outDir, "phrasing_report2.json"), raw, 0o644); err != nil {
		fail(err)
	}
	fmt.Printf("train pairs: %d (families: %d), fragments: %d\n", rep.TrainPairs, len(rep.TrainByFamily), rep.Fragments)
	for _, k := range sortedKeys(rep.TrainByFamily) {
		fmt.Printf("  train %-28s %d\n", k, rep.TrainByFamily[k])
	}
	fmt.Printf("skipped unemittable=%d conflicting=%d duplicates=%d\n", rep.SkippedUnemittable, rep.ConflictingInputs, rep.DuplicateInputs)
	fmt.Printf("leaks vs judge (%d items): template=%d slot=%v input-overlap=%d\n",
		rep.JudgeItems, rep.TemplateLeak, rep.SlotLeak, rep.InputOverlap)
}

func bufioScanner(f *os.File) *bufio.Scanner {
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1024*1024), 1024*1024)
	return sc
}

func sortedKeys(m map[string]int) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func writeJSONL[T any](path string, rows []T) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	for _, r := range rows {
		if err := enc.Encode(r); err != nil {
			return err
		}
	}
	return nil
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "gen_phrasing_data2:", err)
	os.Exit(1)
}
