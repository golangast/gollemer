package chat

// The ambiguous-piece guidance: a bare kind word that names
// more than one book piece gets a disambiguation instead of
// whichever single answer it would otherwise reach — the two
// real pieces named, how they differ in one plain phrase
// each, and the exact working sentences spelled out with the
// user's own name and target carried in. Nothing here writes,
// plans, or gates: it only talks, in the write boundary's
// voice.
//
// Two entries, two seats — the difference is the whole
// design:
//
//   - "handler" is claimed at the very end of the
//     deterministic chain, directly before the write boundary
//     — the seat these lines reach today. The live failure
//     (2026-11-01, John's screenshot): "add handler Jim" got
//     the boundary's "I can't add handler yet — that's not
//     one of the pieces I can assemble today", honest but
//     useless: the book carries TWO handlers (form and
//     session) and the word alone doesn't say which. No
//     earlier claim wants the bare word, so the guidance
//     simply follows the book.
//   - "database" is claimed directly BEFORE the concept
//     book, because the book itself is the problem: its bare
//     name "database" belongs to the inline statement row, so
//     "add a database to jake" silently stuffed sql.Open and
//     a Ping into jake/main.go's main when what John wanted
//     was a database FILE (the database-connect declaration).
//     The guidance preempts the book for exactly the bare
//     shape — no function context, no sibling-row words —
//     and every other database line (the explicit
//     "in <Func>" statement form, the connect declaration,
//     the query/exec/transaction rows) still belongs to the
//     book untouched.
//
// The mechanism is data-shaped: ambiguousKinds names, per
// bare word, the book rows it could mean, and the answers are
// RENDERED from the loaded book — each suggested sentence
// uses one of its row's own names. A row the book in play
// doesn't carry drops out of the answer; fewer than two
// surviving rows means there is no ambiguity worth a claim,
// and the line keeps its current owner.

import (
	"fmt"
	"go/token"
	"strconv"
	"strings"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// ambLineForm is how one candidate's suggested sentence is
// built from the book row.
type ambLineForm int

const (
	// ambLineNamed: a declaration sentence — "add a <piece>
	// named <Name>", plus the target tail when the user
	// named one.
	ambLineNamed ambLineForm = iota
	// ambLineInline: a statement sentence, which needs the
	// function context the bare ask didn't give — "add a
	// <piece> in <Func> to <file>", the context and file
	// supplied by the guidance (see inlineFile).
	ambLineInline
)

// ambiguousRowRef is one concrete piece a bare kind word
// could mean: the book row's id (resolved against the loaded
// book at answer time), the plain-words difference the
// guidance names, and the form its suggested sentence takes.
type ambiguousRowRef struct {
	id    string
	gloss string
	form  ambLineForm
	// say, when set, is the book name the suggested sentence
	// is spelled with — the clearest of the row's names for
	// a beginner ("database connection function", where the
	// row's first name is the terser "database connect").
	// It is only ever a name the loaded row actually
	// carries; a book that drops it falls back to the row's
	// canonical name (see ambPieceRow.sayName).
	say string
	// label, when set, is how the roster names this piece
	// instead of "<article> <canonical name>" — for a row
	// whose canonical name is the bare word itself (the
	// inline database piece is just "database").
	label string
	// noteLabel, when set, marks the row an untargeted
	// answer's closing note is about: the note names where
	// that row lands with no file named (its folder field
	// and the name its sentence used).
	noteLabel string
}

// ambiguousKind is one bare kind word, the book rows it is
// ambiguous between, and the rules its own claim parses by.
type ambiguousKind struct {
	word string
	// beforeBook seats the claim directly ahead of the
	// concept book (the bare word is the book's own claim —
	// preempt it) instead of at the chain's end, before the
	// write boundary (the bare word reaches the boundary
	// unclaimed — follow it).
	beforeBook bool
	// bareName: a single name token may stand in the name
	// slot without "named" ("add handler Jim"). Database's
	// bare token after the kind word is a sibling row's name
	// far more often than a name, so its names ride "named"
	// only.
	bareName bool
	// funcGuard: an "in <bareword>" function context in the
	// tail disqualifies the line — that context is exactly
	// what makes the ask the book's statement form, not the
	// bare piece.
	funcGuard bool
	// nameDefault is the name a named-form sentence uses
	// when the user gave none (database's connection
	// function is "Connect" in every example the product
	// shows). Empty: the sentence shows the <Name> slot.
	nameDefault string
	// funcName is the function an inline-form sentence is
	// spelled into ("main").
	funcName string
	// sharedFolderNote: an untargeted answer ends by naming
	// the folder all candidates land in by default (the
	// rows' shared folder field).
	sharedFolderNote bool
	rows             []ambiguousRowRef
}

var ambiguousKinds = []ambiguousKind{
	{word: "handler", bareName: true, sharedFolderNote: true, rows: []ambiguousRowRef{
		{id: "form-handler", gloss: "reads a submitted form", form: ambLineNamed},
		{id: "session-handler", gloss: "works with the user's session", form: ambLineNamed},
	}},
	{word: "database", beforeBook: true, funcGuard: true, nameDefault: "Connect", funcName: "main", rows: []ambiguousRowRef{
		{id: "database-connect", gloss: "a connection pool you open once and keep", form: ambLineNamed, say: "database connection function", noteLabel: "the connection function"},
		{id: "database", gloss: "sql.Open and a Ping inside one function", form: ambLineInline, label: "the inline database piece"},
	}},
}

// ambiguousKindFor returns the kind seated at beforeBook
// whose bare word matches, if any. The two seats consult
// disjoint halves of the table: a word is only ever
// ambiguous at its own seat.
func ambiguousKindFor(word string, beforeBook bool) (ambiguousKind, bool) {
	for _, k := range ambiguousKinds {
		if k.word == word && k.beforeBook == beforeBook {
			return k, true
		}
	}
	return ambiguousKind{}, false
}

// ambPieceRow is one candidate resolved against the loaded
// book: the row itself plus its table reference.
type ambPieceRow struct {
	entry analyze.ConceptEntry
	ref   ambiguousRowRef
}

// sayName is the book name this candidate's sentence is
// spelled with: the table's chosen name when the loaded row
// carries it, else the row's canonical name.
func (row ambPieceRow) sayName() string {
	if row.ref.say != "" {
		for _, n := range row.entry.Names {
			if n == row.ref.say {
				return n
			}
		}
	}
	return row.entry.Canonical()
}

// ambPieceReq is one claimed line: the kind it hit, the
// candidates as resolved from the book, the name as typed
// ("" when the line named none) and the target as typed
// ("" when the line named none).
type ambPieceReq struct {
	kind   ambiguousKind
	word   string // the kind word as typed, e.g. "handler"
	rows   []ambPieceRow
	name   string
	target string
}

// ambNameStops are the words that can sit in the name slot's
// position without being a name — a line whose would-be name
// is one of these is not these claims' shape at all ("add
// handler for the login page" keeps the boundary's answer).
var ambNameStops = map[string]bool{
	"a": true, "an": true, "the": true,
	"to": true, "in": true, "into": true,
	"named": true, "name": true, "is": true, "that": true,
	"for": true, "with": true, "from": true,
	"on": true, "at": true, "by": true,
	"file": true, "folder": true, "directory": true,
}

// ambNameToken reports whether a lowered token can be the
// declaration's name: a real identifier, and not one of the
// shapes' own words.
func ambNameToken(w string) bool {
	if ambNameStops[w] {
		return false
	}
	return token.IsIdentifier(w)
}

// parseAmbiguousPiece reads one line as a bare-handler add
// against the loaded book (the chain-end claim).
func parseAmbiguousPiece(line string) (ambPieceReq, bool) {
	return parseAmbiguousPieceFrom(line, conceptEntries())
}

// parseAmbiguousPieceFrom is parseAmbiguousPiece against a
// given book — the seam the data-driven tests drive a fixture
// book through.
func parseAmbiguousPieceFrom(line string, entries []analyze.ConceptEntry) (ambPieceReq, bool) {
	return parseAmbiguousSeat(line, entries, false)
}

// parseAmbiguousDatabase reads one line as a bare-database
// add against the loaded book (the pre-book claim).
func parseAmbiguousDatabase(line string) (ambPieceReq, bool) {
	return parseAmbiguousDatabaseFrom(line, conceptEntries())
}

// parseAmbiguousDatabaseFrom is parseAmbiguousDatabase
// against a given book.
func parseAmbiguousDatabaseFrom(line string, entries []analyze.ConceptEntry) (ambPieceReq, bool) {
	return parseAmbiguousSeat(line, entries, true)
}

// parseAmbiguousSeat is the shared parser for both seats:
// add-led lines whose object slot is exactly one of the
// seat's ambiguous kind words, followed by nothing but an
// optional name and an optional target tail.
func parseAmbiguousSeat(line string, entries []analyze.ConceptEntry, beforeBook bool) (ambPieceReq, bool) {
	s := strings.TrimSpace(line)
	s = strings.TrimSuffix(s, ".")
	s = strings.TrimSuffix(s, "?")
	s = strings.TrimSpace(s)
	fields := strings.Fields(s)
	if len(fields) < 2 {
		return ambPieceReq{}, false
	}
	low := make([]string, len(fields))
	for i, f := range fields {
		low[i] = strings.ToLower(strings.Trim(f, ",;:\"'()"))
	}
	// Add only — deliberately: this seat was built as the fix
	// for the "add handler Jim" screenshot lines, and its
	// calibration table pins the other verbs out ("put
	// handler Jim" is a named non-claim there). The 2026-11-05
	// phrasing-gap audit probed widening the seats to put and
	// left the pin standing: the seat answers the add shapes,
	// and a "put a handler…" line stays the honest fallthrough
	// rather than guidance wearing a verb it wasn't built for.
	if low[0] != "add" {
		return ambPieceReq{}, false
	}
	i := 1
	if low[i] == "a" || low[i] == "an" {
		i++
	}
	if i >= len(fields) {
		return ambPieceReq{}, false
	}
	kind, ok := ambiguousKindFor(low[i], beforeBook)
	if !ok {
		return ambPieceReq{}, false
	}
	req := ambPieceReq{kind: kind, word: strings.Trim(fields[i], ",;:\"'()")}
	i++

	// The name slot: "named <Name>" for both kinds, or —
	// handler only — one bare name token standing where the
	// concrete pieces' grammar would put "named". A name is
	// optional; the nameless lines get the same guidance
	// with the name filled in by the kind's default or the
	// <Name> slot.
	switch {
	case i < len(fields) && low[i] == "named":
		if i+1 >= len(fields) || !ambNameToken(low[i+1]) {
			return ambPieceReq{}, false
		}
		req.name = strings.Trim(fields[i+1], ",;:\"'()")
		i += 2
	case i < len(fields) && low[i] != "to" && low[i] != "in" && low[i] != "into":
		if !kind.bareName || !ambNameToken(low[i]) {
			return ambPieceReq{}, false
		}
		req.name = strings.Trim(fields[i], ",;:\"'()")
		i++
	}

	// The optional target tail: one preposition and the
	// target as typed (leading designators stripped, the
	// write boundary's rule). For the database claim, an
	// "in <bareword>" here is a function context, not a
	// target — that context is the book's statement form,
	// and the line is the book's. Anything else left in the
	// line — a behaviour clause most of all — is not these
	// shapes and keeps its current owner.
	if i < len(fields) {
		if low[i] != "to" && low[i] != "in" && low[i] != "into" {
			return ambPieceReq{}, false
		}
		prep := low[i]
		rest, restLow := fields[i+1:], low[i+1:]
		j := 0
		for j < len(rest) && (restLow[j] == "the" || restLow[j] == "file" || restLow[j] == "folder" || restLow[j] == "directory") {
			j++
		}
		if j >= len(rest) {
			return ambPieceReq{}, false
		}
		if kind.funcGuard && prep == "in" &&
			!strings.HasSuffix(restLow[j], ".go") && !strings.Contains(rest[j], "/") {
			return ambPieceReq{}, false
		}
		for _, w := range restLow[j:] {
			if w == "that" {
				return ambPieceReq{}, false
			}
		}
		req.target = strings.Join(rest[j:], " ")
	}

	// Resolve the candidates against the book in play. A
	// row the book doesn't carry can't be suggested; with
	// fewer than two survivors there is no ambiguity to
	// explain and the line keeps its standing owner.
	byID := make(map[string]analyze.ConceptEntry, len(entries))
	for _, e := range entries {
		byID[e.ID] = e
	}
	for _, ref := range kind.rows {
		if e, ok := byID[ref.id]; ok {
			req.rows = append(req.rows, ambPieceRow{entry: e, ref: ref})
		}
	}
	if len(req.rows) < 2 {
		return ambPieceReq{}, false
	}
	return req, true
}

// nameFor is the name a named-form sentence for row is
// spelled with: the user's when they gave one, else the
// kind's default, else the honest <Name> slot.
func (r ambPieceReq) nameFor(row ambPieceRow) string {
	if r.name != "" {
		return r.name
	}
	if r.kind.nameDefault != "" {
		return r.kind.nameDefault
	}
	return "<Name>"
}

// inlineFile is the file an inline-form sentence is spelled
// into: the user's target when it's a file, the target
// folder's main.go when it's a folder, and the standing
// example file when there is no target.
func (r ambPieceReq) inlineFile() string {
	if r.target == "" {
		return "main.go"
	}
	if strings.HasSuffix(strings.ToLower(r.target), ".go") {
		return r.target
	}
	return r.target + "/main.go"
}

// lineFor renders one candidate's exact working sentence
// from its book row, in the row's form.
func (r ambPieceReq) lineFor(row ambPieceRow) string {
	canon := row.sayName()
	art := conceptArticle(canon)
	if row.ref.form == ambLineInline {
		return fmt.Sprintf("add %s %s in %s to %s", art, canon, r.kind.funcName, r.inlineFile())
	}
	sentence := fmt.Sprintf("add %s %s named %s", art, canon, r.nameFor(row))
	if r.target != "" {
		sentence += " to " + r.target
	}
	return sentence
}

// sharedFolder is the folder every candidate lands in by
// default (the rows' folder field), or "" when they don't
// share one.
func (r ambPieceReq) sharedFolder() string {
	folder := ""
	for _, row := range r.rows {
		if row.entry.Folder == "" {
			return ""
		}
		if folder == "" {
			folder = row.entry.Folder
		} else if folder != row.entry.Folder {
			return ""
		}
	}
	return folder
}

// noTargetNote is the closing note of an untargeted answer,
// or "" — where the pieces land when the user named no file
// is part of the guidance, rendered from the table and the
// book: a kind whose candidates share a default folder names
// it; a candidate carrying a noteLabel is named with the
// exact file it would be created as.
func (r ambPieceReq) noTargetNote() string {
	if r.target != "" {
		return ""
	}
	for _, row := range r.rows {
		if row.ref.noteLabel != "" && row.entry.Folder != "" {
			return fmt.Sprintf("with no file named, %s is created as %s/%s.go",
				row.ref.noteLabel, row.entry.Folder, strings.ToLower(r.nameFor(row)))
		}
	}
	if r.kind.sharedFolderNote {
		if folder := r.sharedFolder(); folder != "" {
			return fmt.Sprintf("with no file named, the piece is created in a %s/ folder", folder)
		}
	}
	return ""
}

// answer renders the guidance for one claimed line: the
// ambiguity named plainly, each candidate with its one-phrase
// difference, and the exact working sentences with the user's
// own name and target carried in.
func (r ambPieceReq) answer() string {
	roster := make([]string, 0, len(r.rows))
	lines := make([]string, 0, len(r.rows))
	for _, row := range r.rows {
		label := row.ref.label
		if label == "" {
			label = conceptArticle(row.sayName()) + " " + row.sayName()
		}
		roster = append(roster, fmt.Sprintf("%s (%s)", label, row.ref.gloss))
		lines = append(lines, fmt.Sprintf("%q", r.lineFor(row)))
	}
	count := strconv.Itoa(len(r.rows))
	if len(r.rows) == 2 {
		count = "two"
	}
	out := fmt.Sprintf("%q names %s different pieces here, and I can't tell which one you mean: %s. Say %s",
		r.word, count, strings.Join(roster, " or "), strings.Join(lines, " or "))
	if note := r.noTargetNote(); note != "" {
		out += " — " + note
	}
	return out + "."
}

// tryAmbiguousPiece is the chat-loop entry for the chain-end
// claim (handler): it answers bare-kind-word adds the book
// never claimed with the disambiguation.
func tryAmbiguousPiece(line string) bool {
	return runAmbiguousPiece(line)
}

// runAmbiguousPiece answers one line when the chain-end
// claim takes it. It never asks anything and never touches
// a file.
func runAmbiguousPiece(line string) bool {
	return runAmbiguous(line, parseAmbiguousPiece)
}

// tryAmbiguousDatabase is the chat-loop entry for the
// pre-book claim (database): it answers the bare-database
// ask with the disambiguation before the concept book can
// claim the word for its statement row.
func tryAmbiguousDatabase(line string) bool {
	return runAmbiguousDatabase(line)
}

// runAmbiguousDatabase answers one line when the pre-book
// claim takes it. It never asks anything and never touches
// a file.
func runAmbiguousDatabase(line string) bool {
	return runAmbiguous(line, parseAmbiguousDatabase)
}

// runAmbiguous is the shared runner for both seats.
func runAmbiguous(line string, parse func(string) (ambPieceReq, bool)) bool {
	req, ok := parse(line)
	if !ok {
		return false
	}
	fmt.Printf("%s %s.\n", cTag("gollemer [composer]>"), strings.TrimSuffix(req.answer(), "."))
	recordBrain("composer", true, "ambiguous-piece guidance: "+req.kind.word+" (not performed)")
	return true
}
