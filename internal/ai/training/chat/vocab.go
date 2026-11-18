package chat

// Personal vocabulary (conversational-feel package, item 2,
// 2026-11-05): the user's own names for his files and
// folders, taught explicitly and kept as his data.
//
// John works in his own words: a file he thinks of as "the
// api file" had to be spelled "jim/main.go" in every single
// sentence, forever, because the deterministic grammars read
// paths, not his names. Now he can teach the name once —
//
//	call jim/main.go the api file
//	call this the api file        ("this" = what just landed)
//	call the folder jim the web folder
//
// — and from then on the name is accepted anywhere a
// sentence names that file or folder: "add a handler named
// Jake to the api file" lands in jim/main.go. The reading is
// never silent: when a name resolves, the chat prints it —
// Reading "the api file" as jim/main.go — your name for it —
// before the claim's plan or refusal, the same discipline
// as the session-context readings (sessionctx.go).
//
// The store is the user's, outside every project tree,
// under the teach seat's discipline (taughtkinds.go): a
// sibling file ~/.gollemer_vocab.json — deliberately NOT a
// section of the memory store or the taught store, so
// forgetting his preferences never erases his names,
// forgetting a piece never touches them, and forgetting a
// name never touches either — behind a small Load/Save
// seam, 0600, written atomically. Aliases are keyed by
// project root: a name means a path, and a path means
// nothing across projects, so an alias taught in one
// project is inert in every other.
//
// Precedence, pinned: a real path wins over a name, always.
// A phrase that already exists on disk in the project can't
// be taught as a name, and if a file or folder with the
// phrase's own name appears later, the literal reading wins
// and the alias stops firing — a name never shadows the
// real thing. A phrase the book, the page family, or his
// taught kinds already own can't be taught either (the
// families keep their words), nor can the reference words
// the chat itself reads ("the other file", "it", "the
// webserver", …). Redefinition is refused with the current
// target named — forget first, like the teach seat.
//
// Resolution is a REWRITE, the session-context pattern: the
// loop's spoken-target pass (rewriteSpokenTargets) replaces
// a line's target tail — a preposition plus the spoken
// target — with the real path and prints the reading, so
// every deterministic grammar below reads the line exactly
// as if the path had been typed. Only the tail is ever
// rewritten, only behind a target preposition
// (to|in|into|from|at), and only for lines led by a write
// family's verb — a spoken name in prose is not a target.
// "the other file" rides the same pass (followup.go's
// sibling reference): it resolves to the session's second
// file when exactly two are in play, and to nothing
// otherwise. Follow-up lines are skipped by the pass — the
// follow-up claim resolves its own target slot, so a
// context-free follow-up line stays byte-identical.

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"
)

// ----- the store -----------------------------------------------------------

// vocabAlias is one taught name: the phrase (normalized,
// no leading "the" — the article is the sentence's, not the
// name's), the target it means in one project, and the
// project it means it in.
type vocabAlias struct {
	Phrase   string    `json:"phrase"`  // normalized phrase ("api file")
	Display  string    `json:"display"` // the phrase as he typed it
	Target   string    `json:"target"`  // slash-relative path, no trailing slash
	Folder   bool      `json:"folder"`  // the target is a folder, not a file
	Root     string    `json:"root"`    // the project the name belongs to
	TaughtAt time.Time `json:"taught_at"`
}

type vocabData struct {
	Version int          `json:"version"`
	Aliases []vocabAlias `json:"aliases,omitempty"`
}

func emptyVocabData() vocabData { return vocabData{Version: 1} }

func normalizeVocabData(d vocabData) vocabData {
	if d.Version == 0 {
		d.Version = 1
	}
	if d.Aliases == nil {
		d.Aliases = []vocabAlias{}
	}
	return d
}

// vocabStore is the storage seam, the taught store's shape:
// load one value or save one value.
type vocabStore interface {
	Load() (vocabData, error)
	Save(vocabData) error
}

var vocabStorePathOverride string

func openVocabStore() vocabStore { return jsonVocabStore{path: vocabStorePath()} }

func vocabStorePath() string {
	if vocabStorePathOverride != "" {
		return vocabStorePathOverride
	}
	home, err := os.UserHomeDir()
	if err != nil || home == "" {
		return ".gollemer_vocab.json"
	}
	return filepath.Join(home, ".gollemer_vocab.json")
}

type jsonVocabStore struct{ path string }

func (s jsonVocabStore) Load() (vocabData, error) {
	data, err := os.ReadFile(s.path)
	if errors.Is(err, fs.ErrNotExist) {
		return emptyVocabData(), nil
	}
	if err != nil {
		return emptyVocabData(), err
	}
	if len(strings.TrimSpace(string(data))) == 0 {
		return emptyVocabData(), nil
	}
	var out vocabData
	if err := json.Unmarshal(data, &out); err != nil {
		return emptyVocabData(), err
	}
	return normalizeVocabData(out), nil
}

func (s jsonVocabStore) Save(d vocabData) error {
	d = normalizeVocabData(d)
	data, err := json.MarshalIndent(d, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	dir := filepath.Dir(s.path)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, ".gollemer-vocab-*.tmp")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if err := tmp.Chmod(0o600); err != nil {
		tmp.Close()
		return err
	}
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpName, s.path)
}

func loadVocab() (vocabData, error) { return openVocabStore().Load() }

func saveVocab(d vocabData) error { return openVocabStore().Save(d) }

// aliasesForRoot is the project's own names, in the store's
// order. Root comparison resolves symlinks on both sides
// (sameAnalyzedRoot), so a project reached by two spellings
// still finds its names.
func (d vocabData) aliasesForRoot(rootAbs string) []vocabAlias {
	var out []vocabAlias
	for _, a := range d.Aliases {
		if sameAnalyzedRoot(a.Root, rootAbs) {
			out = append(out, a)
		}
	}
	return out
}

// findAlias returns the project's alias with this normalized
// phrase, or nil.
func (d vocabData) findAlias(rootAbs, phrase string) *vocabAlias {
	for i := range d.Aliases {
		if d.Aliases[i].Phrase == phrase && sameAnalyzedRoot(d.Aliases[i].Root, rootAbs) {
			return &d.Aliases[i]
		}
	}
	return nil
}

// vocabRootAbs settles the project the vocabulary works in
// the claims' own way: the creation root, absolute and
// symlink-resolved, silently ("" when unsettleable — the
// caller simply doesn't engage, and the claims below settle
// roots with their own refusals exactly as today).
func vocabRootAbs() (string, bool) {
	raw := creationRoot()
	if raw == "" {
		return "", false
	}
	return claimRootAbs(raw)
}

// ----- spoken-target resolution ---------------------------------------------

// matchVocabAlias reads target-clause words — the fields of
// a spoken target, original case with a lowercase shadow —
// as "[the] <phrase>" and returns the project's alias for
// the phrase, if one is taught. The WHOLE clause must be
// the name; a name is never matched inside a longer tail.
func matchVocabAlias(fields, low []string, rootAbs string) (vocabAlias, bool) {
	w := low
	if len(w) > 0 && w[0] == "the" {
		w = w[1:]
	}
	if len(w) == 0 {
		return vocabAlias{}, false
	}
	phrase := normalizeKindPhrase(strings.Join(w, " "))
	data, err := loadVocab()
	if err != nil {
		return vocabAlias{}, false
	}
	if a := data.findAlias(rootAbs, phrase); a != nil {
		return *a, true
	}
	return vocabAlias{}, false
}

// resolveSpokenTarget settles ONE spoken target — the words
// a sentence used where a path belongs — against the user's
// names and the session's other file. It returns the real
// path and the reading line that says so, or "" when the
// words are neither: an ordinary path, an unknown name, a
// reference with no referent. The caller prints the reading
// before anything acts on the resolved line.
func resolveSpokenTarget(target string, rootAbs string) (resolved, note string) {
	fields := strings.Fields(strings.TrimSpace(target))
	if len(fields) == 0 {
		return "", ""
	}
	low := make([]string, len(fields))
	for i, f := range fields {
		low[i] = strings.ToLower(f)
	}
	// "the other file": the session's second file, when
	// exactly two are in play (sessionctx.go).
	if strings.Join(low, " ") == "the other file" {
		if workContextLive(rootAbs) {
			if other, ok := otherFileCandidate(); ok {
				return other, fmt.Sprintf("Reading \"the other file\" as %s — the other file from this session's work.", other)
			}
		}
		return "", ""
	}
	a, ok := matchVocabAlias(fields, low, rootAbs)
	if !ok {
		return "", ""
	}
	// A real path wins over a name: if a file or folder with
	// the phrase's own name has appeared on disk since the
	// name was taught, the literal reading is the true one
	// and the alias stops firing.
	if _, err := os.Stat(filepath.Join(rootAbs, filepath.FromSlash(a.Phrase))); err == nil {
		return "", ""
	}
	return a.Target, fmt.Sprintf("Reading %q as %s — your name for it.", strings.Join(fields, " "), a.Target)
}

// spokenTargetPreps are the prepositions a target clause
// opens with across the deterministic grammars — the only
// positions the rewrite pass touches.
func spokenTargetPrep(w string) bool {
	switch w {
	case "to", "in", "into", "from", "at":
		return true
	}
	return false
}

// spokenRewriteVerbs are the lead verbs of the deterministic
// write claims whose sentences carry target clauses. The
// rewrite pass serves those claims alone: a spoken name in
// a line led by anything else (a question, small talk) is
// prose, not a target, and is never rewritten or announced.
var spokenRewriteVerbs = map[string]bool{
	"add": true, "put": true, "create": true, "make": true,
	"build": true, "serve": true, "connect": true, "wire": true,
	"register": true, "attach": true, "link": true, "plug": true,
	"hook": true, "run": true, "start": true, "launch": true,
}

// rewriteSpokenTargets is the unified loop's spoken-target
// pass (vocab.go): seated after the vocabulary's own teach
// and manage lines and the clarifying answers, ahead of
// every claim that reads a target. When a line's target
// tail — a preposition and the spoken words behind it — is
// a taught name or "the other file", the tail is replaced
// with the real path and the reading is printed, so the
// claims below read the line exactly as if the path had
// been typed. Anything else returns the line untouched and
// prints nothing: no project, no names, no matching tail —
// behavior is byte-identical. Follow-up lines are skipped:
// the follow-up claim resolves its own target slot, so a
// follow-up in a session with no context to read it against
// keeps today's behavior to the byte.
func rewriteSpokenTargets(line string) string {
	rootAbs, ok := vocabRootAbs()
	if !ok {
		return line
	}
	s := strings.TrimSpace(line)
	s = strings.TrimRight(s, ".?!")
	s = strings.TrimSpace(s)
	fields := strings.Fields(s)
	if len(fields) < 3 {
		return line
	}
	low := make([]string, len(fields))
	for i, f := range fields {
		low[i] = strings.ToLower(f)
	}
	if !spokenRewriteVerbs[low[0]] {
		return line
	}
	if isFollowUpShape(line) {
		return line
	}
	// Field start offsets, for the splice.
	offsets := make([]int, len(fields))
	pos := 0
	for i, f := range fields {
		idx := strings.Index(s[pos:], f)
		if idx < 0 {
			return line
		}
		offsets[i] = pos + idx
		pos = offsets[i] + len(f)
	}
	for i := len(fields) - 2; i >= 1; i-- {
		if !spokenTargetPrep(low[i]) {
			continue
		}
		resolved, note := resolveSpokenTarget(strings.Join(fields[i+1:], " "), rootAbs)
		if resolved == "" {
			continue
		}
		fmt.Printf("%s %s\n", cTag("gollemer [composer]>"), note)
		return s[:offsets[i]] + fields[i] + " " + resolved
	}
	return line
}

// ----- grammar ---------------------------------------------------------------

// vocabTeachAsk is one teach sentence: "this" or a spoken
// target, an optional file|folder designator, and the name
// phrase as typed.
type vocabTeachAsk struct {
	this       bool
	designator string // "file" | "folder" | ""
	target     string // as typed ("" when this)
	phrase     string // as typed, fields joined
}

// parseVocabTeach reads the teach shapes:
//
//	call this the <name>
//	call this file|folder the <name>
//	call <target> the <name>
//	call the file|folder <target> the <name>
//
// Anything else declines — in particular "call me …", which
// is the memory store's shape and sits above this claim in
// the loop besides.
func parseVocabTeach(line string) (vocabTeachAsk, bool) {
	fields, low := splitSentence(line)
	if len(fields) < 4 || low[0] != "call" {
		return vocabTeachAsk{}, false
	}
	i := 1
	var ta vocabTeachAsk
	switch {
	case low[i] == "this":
		ta.this = true
		i++
		if i < len(fields) && (low[i] == "file" || low[i] == "folder") {
			ta.designator = low[i]
			i++
		}
	case low[i] == "the" && i+1 < len(fields) && (low[i+1] == "file" || low[i+1] == "folder"):
		ta.designator = low[i+1]
		i += 2
		if i >= len(fields) {
			return vocabTeachAsk{}, false
		}
		ta.target = fields[i]
		i++
	default:
		ta.target = fields[i]
		if ta.target == "me" || low[i] == "the" {
			return vocabTeachAsk{}, false
		}
		i++
	}
	if i >= len(fields) || low[i] != "the" {
		return vocabTeachAsk{}, false
	}
	i++
	if i >= len(fields) {
		return vocabTeachAsk{}, false
	}
	ta.phrase = strings.Join(fields[i:], " ")
	return ta, true
}

// vocabReservedPhrases are the normalized names that can
// never be taught: the words the chat itself reads as
// references or claim slots, which a user's name must sit
// beside, never over — the same discipline as the teach
// seat's reserved kind names.
var vocabReservedPhrases = map[string]string{
	"it":             "a pronoun the chat already reads",
	"this":           "a word the teach sentences themselves parse by",
	"that":           "a pronoun the chat already reads",
	"file":           "a word the sentences read as a target designator",
	"folder":         "a word the sentences read as a target designator",
	"directory":      "a word the sentences read as a target designator",
	"the file":       "a target phrase the chat already reads",
	"that file":      "a reference to the file you're working on",
	"this file":      "a reference to the file you're working on",
	"the same file":  "a reference to the file you're working on",
	"same file":      "a reference to the file you're working on",
	"the folder":     "a target phrase the chat already reads",
	"that folder":    "a reference to the folder you're working in",
	"this folder":    "a reference to the folder you're working in",
	"the other file": "a reference to the session's other file",
	"other file":     "a reference to the session's other file",
	"handler":        "a word the wiring sentences parse by",
	"the handler":    "a word the wiring sentences parse by",
	"webserver":      "a word the server sentences parse by",
	"the webserver":  "a word the server sentences parse by",
	"server":         "a word the server sentences parse by",
	"the server":     "a word the server sentences parse by",
	"main":           "the name of the program's entry point, which the sentences read as a function reference",
}

// tryVocab is the chat-loop entry: the vocabulary's teach
// and manage sentences, claimed deterministically beside
// the clarifying answers — ahead of the spoken-target pass,
// whose rewrite a teach line's own tail would otherwise
// meet (a redefinition's line ends in the very name it
// redefines).
func tryVocab(line string, r *LineReader) bool {
	return runVocab(line, func(prompt string) (string, error) {
		return r.ReadLine(prompt)
	})
}

func runVocab(line string, ask editAsk) bool {
	if runVocabManage(line) {
		return true
	}
	if ta, ok := parseVocabTeach(line); ok {
		runVocabTeach(ta)
		return true
	}
	return false
}

// ----- teach -----------------------------------------------------------------

func runVocabTeach(ta vocabTeachAsk) {
	tag := cTag("gollemer [composer]>")
	phrase := normalizeKindPhrase(ta.phrase)

	// The name's own shape: a name, not a path, and not a
	// word the chat already reads as something else.
	if strings.Contains(phrase, "/") || strings.HasSuffix(phrase, ".go") {
		fmt.Printf("%s Not naming it that: %q is a path, not a name — a name is words, like \"api file\" or \"the db file\".\n", tag, ta.phrase)
		recordBrain("composer", false, "vocab refused: phrase is a path")
		return
	}
	if why, taken := vocabReservedPhrases[phrase]; taken {
		fmt.Printf("%s Not naming it that: %q is %s — your names sit beside the chat's own words, never over them. Pick a name of your own, like \"api file\".\n", tag, phrase, why)
		recordBrain("composer", false, "vocab refused: reserved phrase")
		return
	}
	if owner, taken := reservedKindNames()[phrase]; taken {
		fmt.Printf("%s Not naming it that: %q is already one of %s's piece names — your names are for files and folders, and a name that doubles as a piece would tangle the sentences that use both. Pick a different name, like \"api file\".\n", tag, phrase, owner)
		recordBrain("composer", false, "vocab refused: phrase shadows a family")
		return
	}
	if kinds, err := loadTaughtKinds(); err == nil {
		if kinds.findTaughtKind(phrase) != nil {
			fmt.Printf("%s Not naming it that: you've already taught me a piece kind called %q — one phrase can't be both a piece and a file's name (\"forget the %s\" would have two things to forget). Pick a different name for the file.\n", tag, phrase, phrase)
			recordBrain("composer", false, "vocab refused: phrase is a taught kind")
			return
		}
	}

	rootAbs, ok := vocabRootAbs()
	if !ok {
		fmt.Printf("%s I don't have a project folder to name things in — start me inside your project and ask again.\n", tag)
		recordBrain("composer", false, "vocab refused: no root")
		return
	}

	// The target: what the name will mean.
	var target string
	var folder bool
	var refusal string
	if ta.this {
		target, folder, refusal = vocabThisTarget(rootAbs, ta.designator)
	} else {
		target, folder, refusal = vocabSpokenTarget(rootAbs, ta.designator, ta.target)
	}
	if refusal != "" {
		fmt.Printf("%s Not naming it that: %s.\n", tag, strings.TrimSuffix(refusal, "."))
		recordBrain("composer", false, "vocab refused: target")
		return
	}

	// A name never shadows the real thing: a file or folder
	// already carrying the phrase's own name in this project
	// keeps it.
	if st, err := os.Stat(filepath.Join(rootAbs, filepath.FromSlash(phrase))); err == nil {
		kindWord := "file"
		if st.IsDir() {
			kindWord = "folder"
		}
		fmt.Printf("%s Not naming it that: there's already a %s named %q in this project — a name can't stand in front of the real thing. Pick a name nothing here is called, like \"api file\".\n", tag, kindWord, phrase)
		recordBrain("composer", false, "vocab refused: phrase exists on disk")
		return
	}

	data, err := loadVocab()
	if err != nil {
		fmt.Printf("%s I couldn't read your names store (%s): %v\n", tag, vocabStorePath(), err)
		recordBrain("composer", false, "vocab refused: store unreadable")
		return
	}
	if old := data.findAlias(rootAbs, phrase); old != nil {
		fmt.Printf("%s Not naming it that: %q is already your name for %s in this project. If you want it to mean something else, forget it first: say \"forget the %s\".\n", tag, phrase, old.Target, phrase)
		recordBrain("composer", false, "vocab refused: name already taught")
		return
	}
	data.Aliases = append(data.Aliases, vocabAlias{
		Phrase:   phrase,
		Display:  strings.Join(strings.Fields(ta.phrase), " "),
		Target:   target,
		Folder:   folder,
		Root:     canonicalMemoryRoot(rootAbs),
		TaughtAt: time.Now(),
	})
	if err := saveVocab(data); err != nil {
		fmt.Printf("%s I couldn't save it to your names store (%s): %v — nothing was kept.\n", tag, vocabStorePath(), err)
		recordBrain("composer", false, "vocab refused: store unwritable")
		return
	}
	disp := target
	if folder {
		disp = target + "/"
	}
	fmt.Printf("%s Learned your name for it:\n", tag)
	fmt.Printf("  \"the %s\" — %s in this project.\n", phrase, disp)
	if folder {
		fmt.Printf("  Use it anywhere a sentence names that folder — \"add a handler named <Name> to the %s\" lands in %s by the folder's own rules.\n", phrase, disp)
	} else {
		fmt.Printf("  Use it anywhere a sentence names that file — \"add a print line to the %s\" lands in %s.\n", phrase, target)
	}
	fmt.Printf("  It's yours alone: it lives in %s, separate from your memory and your taught pieces, so forgetting one never touches the others, and it works in every session in this project. Say \"forget the %s\" to remove it.\n", vocabStorePath(), phrase)
	recordBrain("composer", true, "vocab learned: "+phrase)
}

// vocabThisTarget resolves a teach sentence's "this": the
// file or folder that landed most recently in this session
// (sessionctx.go), the designator's word when the sentence
// carries one. Every other state refuses in words — "this"
// is never guessed.
func vocabThisTarget(rootAbs, designator string) (target string, folder bool, refusal string) {
	noThis := "I don't have a \"this\" in this session yet — nothing has landed. Name the file or folder outright: \"call jim/main.go the api file\""
	if !workContextLive(rootAbs) {
		return "", false, noThis
	}
	switch designator {
	case "file":
		if workCtx.lastFile == "" {
			return "", false, "nothing has landed in a file yet in this session — name the file outright: \"call jim/main.go the api file\""
		}
		return workCtx.lastFile, false, ""
	case "folder":
		if workCtx.folder == "" {
			return "", false, "you're working in the project root itself, so there's no working folder to name — name the folder outright: \"call jim the web folder\""
		}
		return workCtx.folder, true, ""
	}
	switch workCtx.lastKind {
	case "file":
		if workCtx.lastFile != "" {
			return workCtx.lastFile, false, ""
		}
	case "folder":
		if workCtx.folder != "" {
			return workCtx.folder, true, ""
		}
	}
	return "", false, noThis
}

// vocabSpokenTarget resolves a teach sentence's spoken
// target to its project-relative path and its kind: the
// designator's word wins; then a trailing slash; then the
// disk; then a .go suffix; then a path's slash reads as a
// folder's. A bare word that fits none of those is refused
// with the two spelled forms, never guessed.
func vocabSpokenTarget(rootAbs, designator, typed string) (target string, folder bool, refusal string) {
	abs, rel, fence := serverTargetResolve(rootAbs, typed)
	if fence != "" {
		return "", false, fence
	}
	rel = filepath.ToSlash(rel)
	switch designator {
	case "file":
		return rel, false, ""
	case "folder":
		return strings.TrimSuffix(rel, "/"), true, ""
	}
	if strings.HasSuffix(typed, "/") {
		return strings.TrimSuffix(rel, "/"), true, ""
	}
	if st, err := os.Stat(abs); err == nil {
		return rel, st.IsDir(), ""
	}
	if strings.HasSuffix(strings.ToLower(typed), ".go") {
		return rel, false, ""
	}
	if strings.Contains(typed, "/") {
		return strings.TrimSuffix(rel, "/"), true, ""
	}
	return "", false, fmt.Sprintf("I can't tell whether %s is a file or a folder — say \"call the file %s the <name>\" or \"call the folder %s the <name>\" and I'll know", typed, typed, typed)
}

// ----- manage ------------------------------------------------------------------

var (
	vocabListRe      = regexp.MustCompile(`(?i)^\s*(what do you call things|what do i call things|what are my names for things|list my names for things)\s*[?.!]*\s*$`)
	vocabForgetAllRe = regexp.MustCompile(`(?i)^\s*forget (what i call things|all my names for things)\s*[!.]*\s*$`)
	vocabForgetRe    = regexp.MustCompile(`(?i)^\s*forget the\s+(.+?)\s*[!.]*\s*$`)
)

// runVocabManage is the vocabulary's listing and forgetting.
// The one-name forget ("forget the <phrase>") is claimed
// ONLY when the phrase is a name in this project — any
// other "forget the …" line falls through untouched, to the
// teach seat's forget below (a taught kind by that phrase)
// or the memory store's, exactly as today.
func runVocabManage(line string) bool {
	if vocabListRe.MatchString(line) {
		listVocab()
		return true
	}
	if vocabForgetAllRe.MatchString(line) {
		forgetAllVocab()
		return true
	}
	if m := vocabForgetRe.FindStringSubmatch(line); m != nil {
		return forgetVocab(normalizeKindPhrase(m[1]))
	}
	return false
}

func listVocab() {
	tag := cTag("gollemer [composer]>")
	rootAbs, ok := vocabRootAbs()
	if !ok {
		fmt.Printf("%s I don't have a project folder open — your names live with their projects, so start me inside one and ask again.\n", tag)
		recordBrain("composer", false, "vocab list refused: no root")
		return
	}
	data, err := loadVocab()
	if err != nil {
		fmt.Printf("%s I couldn't read your names store (%s): %v\n", tag, vocabStorePath(), err)
		recordBrain("composer", false, "vocab list refused: store unreadable")
		return
	}
	mine := data.aliasesForRoot(rootAbs)
	if len(mine) == 0 {
		if len(data.Aliases) > 0 {
			fmt.Printf("%s You haven't given me any names for things in this project yet — the names I know live in your other projects, and each only means something in its own. Teach me one here: say \"call jim/main.go the api file\", and sentences can use the name from then on.\n", tag)
		} else {
			fmt.Printf("%s You haven't given me any names for things yet. Teach me one: say \"call jim/main.go the api file\", and from then on sentences can name that file by your name for it.\n", tag)
		}
		recordBrain("composer", true, "vocab listed: none")
		return
	}
	sort.Slice(mine, func(i, j int) bool { return mine[i].Phrase < mine[j].Phrase })
	fmt.Printf("%s Your names for things in this project:\n", tag)
	for _, a := range mine {
		disp := a.Target
		kind := "file"
		if a.Folder {
			disp += "/"
			kind = "folder"
		}
		fmt.Printf("  \"the %s\" — %s (a %s), taught %s.\n", a.Phrase, disp, kind, taughtDate(a.TaughtAt))
	}
	fmt.Printf("  Say \"forget the <name>\" to remove one, or \"forget what I call things\" to remove them all.\n")
	recordBrain("composer", true, "vocab listed")
}

// forgetVocab removes one name from this project. It
// reports handled ONLY when a name was actually there: any
// other "forget the …" line is not this store's to answer.
func forgetVocab(phrase string) bool {
	tag := cTag("gollemer [composer]>")
	rootAbs, ok := vocabRootAbs()
	if !ok {
		return false
	}
	data, err := loadVocab()
	if err != nil {
		fmt.Printf("%s I couldn't read your names store (%s): %v\n", tag, vocabStorePath(), err)
		recordBrain("composer", false, "vocab forget refused: store unreadable")
		return true
	}
	for i := range data.Aliases {
		a := data.Aliases[i]
		if a.Phrase != phrase || !sameAnalyzedRoot(a.Root, rootAbs) {
			continue
		}
		data.Aliases = append(data.Aliases[:i], data.Aliases[i+1:]...)
		if err := saveVocab(data); err != nil {
			fmt.Printf("%s I couldn't update your names store (%s): %v — nothing was forgotten.\n", tag, vocabStorePath(), err)
			recordBrain("composer", false, "vocab forget refused: store unwritable")
			return true
		}
		disp := a.Target
		if a.Folder {
			disp += "/"
		}
		fmt.Printf("%s Forgotten: \"the %s\" — your name for %s in this project. The name is gone; %s itself stays, of course.\n", tag, a.Phrase, disp, disp)
		recordBrain("composer", true, "vocab forgotten: "+phrase)
		return true
	}
	return false
}

func forgetAllVocab() {
	tag := cTag("gollemer [composer]>")
	rootAbs, ok := vocabRootAbs()
	if !ok {
		fmt.Printf("%s I don't have a project folder open — your names live with their projects, so there's no project's names here to forget.\n", tag)
		recordBrain("composer", false, "vocab forget refused: no root")
		return
	}
	data, err := loadVocab()
	if err != nil {
		fmt.Printf("%s I couldn't read your names store (%s): %v\n", tag, vocabStorePath(), err)
		recordBrain("composer", false, "vocab forget refused: store unreadable")
		return
	}
	mine := data.aliasesForRoot(rootAbs)
	if len(mine) == 0 {
		fmt.Printf("%s You haven't given me any names for things in this project — nothing to forget. (Names in your other projects stay as they are.)\n", tag)
		recordBrain("composer", true, "vocab forget: none here")
		return
	}
	kept := data.Aliases[:0]
	for _, a := range data.Aliases {
		if !sameAnalyzedRoot(a.Root, rootAbs) {
			kept = append(kept, a)
		}
	}
	data.Aliases = kept
	if err := saveVocab(data); err != nil {
		fmt.Printf("%s I couldn't update your names store (%s): %v — nothing was forgotten.\n", tag, vocabStorePath(), err)
		recordBrain("composer", false, "vocab forget refused: store unwritable")
		return
	}
	parts := make([]string, 0, len(mine))
	for _, a := range mine {
		parts = append(parts, fmt.Sprintf("\"the %s\" (for %s)", a.Phrase, a.Target))
	}
	word := "names"
	if len(mine) == 1 {
		word = "name"
	}
	fmt.Printf("%s Forgotten: your %d %s for things in this project — %s. Names in your other projects stay as they are.\n", tag, len(mine), word, strings.Join(parts, ", "))
	recordBrain("composer", true, "vocab forgotten: all in project")
}
