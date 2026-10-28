package main

import (
	"fmt"
	"sort"
	"strings"

	"github.com/golangast/gollemer/internal/ai/training/decl"
)

// Constrained decoding for the decl model.
//
// The constraint is a token-level NFA built ONLY from the item's
// input string (see constrain.go): the declaration kind derived
// mechanically from the input text, and the identifier/type atoms
// that appear in it. A candidate next token is allowed iff stepping
// the NFA over the token's symbols leaves at least one live
// configuration — i.e. iff the partial output is still a prefix of
// some legal Go declaration of the item's kind whose identifiers all
// come from the input (plus Go's builtin type atoms, which are
// language constants, and the structural tag atoms; see
// buildConstraint).
//
// Model tokens are not grammar symbols: the trainer's tokenizer
// fuses ";" onto the preceding word ("string;", "io.writer;",
// `");"`), so each vocabulary word is first decomposed into symbols
// by symbolizeWord. The type-expression sub-automaton is a set of
// token templates covering exactly the shapes the decl generator
// emits (simple, slice, pointer, map, func types, generic
// instantiations, []*T, chan) — verified against every gold token
// stream in the held-out set during development.

type symKind uint8

const (
	sAtom symKind = iota
	sLBrace
	sRBrace
	sLBracket
	sRBracket
	sLParen
	sRParen
	sComma
	sColon
	sQuote
	sStar
	sSemi
	sTagOpen // `json
	sBacktick
	sEOS
	sInvalid
)

type sym struct {
	kind symKind
	text string // sAtom only
}

// symbolizeWord decomposes one vocabulary word into grammar symbols.
func symbolizeWord(word string) []sym {
	switch word {
	case "</s>":
		return []sym{{kind: sEOS}}
	case "<s>", "<pad>", "UNK":
		return []sym{{kind: sInvalid}}
	}
	var out []sym
	var atom strings.Builder
	flush := func() {
		if atom.Len() > 0 {
			out = append(out, sym{kind: sAtom, text: atom.String()})
			atom.Reset()
		}
	}
	runes := []rune(word)
	for i := 0; i < len(runes); i++ {
		switch r := runes[i]; r {
		case '{':
			flush()
			out = append(out, sym{kind: sLBrace})
		case '}':
			flush()
			out = append(out, sym{kind: sRBrace})
		case '[':
			flush()
			out = append(out, sym{kind: sLBracket})
		case ']':
			flush()
			out = append(out, sym{kind: sRBracket})
		case '(':
			flush()
			out = append(out, sym{kind: sLParen})
		case ')':
			flush()
			out = append(out, sym{kind: sRParen})
		case ',':
			flush()
			out = append(out, sym{kind: sComma})
		case ':':
			flush()
			out = append(out, sym{kind: sColon})
		case '"':
			flush()
			out = append(out, sym{kind: sQuote})
		case '*':
			flush()
			out = append(out, sym{kind: sStar})
		case ';':
			flush()
			out = append(out, sym{kind: sSemi})
		case '`':
			flush()
			if i+4 < len(runes) && string(runes[i+1:i+5]) == "json" {
				out = append(out, sym{kind: sTagOpen})
				i += 4
			} else {
				out = append(out, sym{kind: sBacktick})
			}
		default:
			atom.WriteRune(r)
		}
	}
	flush()
	return out
}

// --- type-expression templates -------------------------------------

type tElem struct {
	kind symKind
	text string // fixed atom text (kind == sAtom && !slot)
	slot bool   // a type-atom slot
}

func tSlot() tElem           { return tElem{kind: sAtom, slot: true} }
func tAtom(s string) tElem   { return tElem{kind: sAtom, text: s} }
func tPunct(k symKind) tElem { return tElem{kind: k} }

// typeTemplates are the type-expression shapes the decl generator
// emits, in symbol space. Atom slots accept any allowed type atom
// except the type keywords (map/func/chan head their own templates).
var typeTemplates = [][]tElem{
	{tSlot()}, // T
	{tPunct(sLBracket), tPunct(sRBracket), tSlot()},                                              // []T
	{tPunct(sStar), tSlot()},                                                                     // *T
	{tAtom("map"), tPunct(sLBracket), tSlot(), tPunct(sRBracket), tSlot()},                       // map[K]V
	{tAtom("func"), tPunct(sLParen), tPunct(sRParen), tSlot()},                                   // func() R
	{tAtom("func"), tPunct(sLParen), tSlot(), tPunct(sRParen), tSlot()},                          // func(A) R
	{tAtom("func"), tPunct(sLParen), tSlot(), tPunct(sComma), tSlot(), tPunct(sRParen), tSlot()}, // func(A, B) R
	{tSlot(), tPunct(sLBracket), tSlot(), tPunct(sRBracket)},                                     // G[T]
	{tSlot(), tPunct(sLBracket), tSlot(), tPunct(sComma), tSlot(), tPunct(sRBracket)},            // G[K, V]
	{tPunct(sLBracket), tPunct(sRBracket), tPunct(sStar), tSlot()},                               // []*T
	{tAtom("chan"), tSlot()},                                                                     // chan T
}

type tPos struct{ tmpl, pos int }

func (e tElem) matches(s sym, c *constraint) bool {
	if e.slot {
		return s.kind == sAtom && c.typeAtoms[s.text] &&
			s.text != "map" && s.text != "func" && s.text != "chan"
	}
	if e.kind == sAtom {
		return s.kind == sAtom && s.text == e.text
	}
	return s.kind == e.kind
}

// stepType advances every live template position over one symbol.
func stepType(set []tPos, s sym, c *constraint) []tPos {
	var out []tPos
	for _, p := range set {
		tmpl := typeTemplates[p.tmpl]
		if p.pos < len(tmpl) && tmpl[p.pos].matches(s, c) {
			out = append(out, tPos{p.tmpl, p.pos + 1})
		}
	}
	return out
}

func startTypeSet() []tPos {
	out := make([]tPos, len(typeTemplates))
	for i := range typeTemplates {
		out[i] = tPos{i, 0}
	}
	return out
}

func typeComplete(set []tPos) bool {
	for _, p := range set {
		if p.pos == len(typeTemplates[p.tmpl]) {
			return true
		}
	}
	return false
}

// --- declaration automaton -----------------------------------------

const (
	phStart = iota
	phDeclName
	phKindKw
	phBodyOpen
	phStructStart
	phStructAfterName
	phStructTypeEnd
	phTagColon
	phTagQuote
	phTagContent
	phTagAfterContent
	phTagOmit
	phTagQuote2
	phTagClosed
	phTagEnded
	phIfaceStart
	phIfaceAfterMethod
	phIfaceTypeEnd
	phParamsStart
	phParamsAfterName
	phParamItemEnd
	phFuncStart
	phRecvName
	phRecvType
	phRecvTypeAtom
	phRecvClose
	phFuncName
	phFuncParamsOpen
	phFuncResults
	phFuncResList
	phFuncResListEnd
	phFuncResEnd
	phFuncResDone
	phBodyStart
	phPanicOpen
	phPanicQuote1
	phPanicNot
	phPanicImpl
	phPanicQuote2
	phPanicClose
	phPanicEnd
	phMethodResults
	phMethodResList
	phMethodResListEnd
	phMethodResEnd
	phMethodResDone
	phDone
)

// config is one live automaton state. When inType is set, the type
// sub-automaton is mid-expression; phase then names the continuation
// the machine enters when the type completes and a symbol the type
// cannot consume arrives.
type config struct {
	phase  int
	exit   int // params machine exit phase
	inType bool
	tset   []tPos
	field  string // current struct field name ("" = none / embedded)
}

func (c config) key() string {
	var b strings.Builder
	fmt.Fprintf(&b, "%d/%d/%s/", c.phase, c.exit, c.field)
	if c.inType {
		pairs := append([]tPos(nil), c.tset...)
		sort.Slice(pairs, func(i, j int) bool {
			if pairs[i].tmpl != pairs[j].tmpl {
				return pairs[i].tmpl < pairs[j].tmpl
			}
			return pairs[i].pos < pairs[j].pos
		})
		b.WriteString("T")
		for _, p := range pairs {
			fmt.Fprintf(&b, ".%d:%d", p.tmpl, p.pos)
		}
	}
	return b.String()
}

func dedupe(cfgs []config) []config {
	seen := map[string]bool{}
	out := cfgs[:0]
	for _, c := range cfgs {
		k := c.key()
		if !seen[k] {
			seen[k] = true
			out = append(out, c)
		}
	}
	return out
}

// enterType starts the type sub-automaton with s as its first symbol,
// landing in continuation phase ret. Nil if no template consumes s.
func enterType(c config, s sym, ret int, con *constraint) []config {
	set := stepType(startTypeSet(), s, con)
	if len(set) == 0 {
		return nil
	}
	nc := c
	nc.phase = ret
	nc.inType = true
	nc.tset = set
	return []config{nc}
}

// stepAll advances every configuration over one symbol.
func stepAll(cfgs []config, s sym, con *constraint) []config {
	var out []config
	for _, c := range cfgs {
		if c.inType {
			if nt := stepType(c.tset, s, con); len(nt) > 0 {
				nc := c
				nc.tset = nt
				out = append(out, nc)
			}
			if typeComplete(c.tset) {
				popped := c
				popped.inType = false
				popped.tset = nil
				out = append(out, stepDecl(popped, s, con)...)
			}
			continue
		}
		out = append(out, stepDecl(c, s, con)...)
	}
	return dedupe(out)
}

func isName(c *constraint, s sym) bool {
	return s.kind == sAtom && c.names[s.text]
}

func isTypeAtom(c *constraint, s sym) bool {
	return s.kind == sAtom && c.typeAtoms[s.text]
}

// stepDecl is the declaration-level transition function for
// configurations that are not mid-type.
func stepDecl(c config, s sym, con *constraint) []config {
	one := func(phase int) []config {
		nc := c
		nc.phase = phase
		return []config{nc}
	}
	switch c.phase {
	case phStart:
		if s.kind == sAtom && ((con.kind == decl.KindFunc && s.text == "func") ||
			(con.kind != decl.KindFunc && s.text == "type")) {
			if con.kind == decl.KindFunc {
				return one(phFuncStart)
			}
			return one(phDeclName)
		}
	case phDeclName:
		if isName(con, s) {
			return one(phKindKw)
		}
	case phKindKw:
		if s.kind == sAtom && s.text == string(con.kind) {
			return one(phBodyOpen)
		}
	case phBodyOpen:
		if s.kind == sLBrace {
			if con.kind == decl.KindStruct {
				return one(phStructStart)
			}
			return one(phIfaceStart)
		}
	case phStructStart:
		base := c
		base.field = "" // a fresh member carries no field name yet
		switch {
		case s.kind == sRBrace:
			return one(phDone)
		case isName(con, s):
			out := []config{{phase: phStructAfterName, field: s.text}}
			out = append(out, enterType(base, s, phStructTypeEnd, con)...)
			return out
		default:
			return enterType(base, s, phStructTypeEnd, con)
		}
	case phStructAfterName:
		return enterType(c, s, phStructTypeEnd, con)
	case phStructTypeEnd:
		switch {
		case s.kind == sSemi:
			return one(phStructStart)
		case s.kind == sRBrace:
			return one(phDone)
		case s.kind == sTagOpen && c.field != "":
			return one(phTagColon)
		}
	case phTagColon:
		if s.kind == sColon {
			return one(phTagQuote)
		}
	case phTagQuote:
		if s.kind == sQuote {
			return one(phTagContent)
		}
	case phTagContent:
		if s.kind == sAtom && s.text == c.field {
			return one(phTagAfterContent)
		}
	case phTagAfterContent:
		switch s.kind {
		case sComma:
			return one(phTagOmit)
		case sQuote:
			return one(phTagClosed)
		}
	case phTagOmit:
		if s.kind == sAtom && s.text == "omitempty" {
			return one(phTagQuote2)
		}
	case phTagQuote2:
		if s.kind == sQuote {
			return one(phTagClosed)
		}
	case phTagClosed:
		if s.kind == sBacktick {
			return one(phTagEnded)
		}
	case phTagEnded:
		switch s.kind {
		case sSemi:
			return one(phStructStart)
		case sRBrace:
			return one(phDone)
		}
	case phIfaceStart:
		switch {
		case s.kind == sRBrace:
			return one(phDone)
		case isName(con, s):
			out := []config{{phase: phIfaceAfterMethod}}
			out = append(out, enterType(c, s, phIfaceTypeEnd, con)...)
			return out
		default:
			return enterType(c, s, phIfaceTypeEnd, con)
		}
	case phIfaceAfterMethod:
		if s.kind == sLParen {
			nc := c
			nc.phase = phParamsStart
			nc.exit = phMethodResults
			return []config{nc}
		}
	case phIfaceTypeEnd:
		switch s.kind {
		case sSemi:
			return one(phIfaceStart)
		case sRBrace:
			return one(phDone)
		}
	case phParamsStart:
		switch {
		case s.kind == sRParen:
			return one(c.exit)
		case isName(con, s):
			out := []config{{phase: phParamsAfterName, exit: c.exit}}
			nc := c
			out = append(out, enterType(nc, s, phParamItemEnd, con)...)
			return out
		default:
			return enterType(c, s, phParamItemEnd, con)
		}
	case phParamsAfterName:
		return enterType(c, s, phParamItemEnd, con)
	case phParamItemEnd:
		switch s.kind {
		case sComma:
			return one(phParamsStart)
		case sRParen:
			return one(c.exit)
		}
	case phFuncStart:
		switch {
		case s.kind == sLParen:
			return one(phRecvName)
		case isName(con, s):
			return one(phFuncParamsOpen)
		}
	case phRecvName:
		// The receiver NAME is the one identifier the phrasing
		// frames never carry (they name only the receiver type),
		// and the eval scores it only where the input gave it. The
		// position therefore admits any single-letter atom — a
		// structural constant of Go style, not item information.
		if isName(con, s) || (s.kind == sAtom && len(s.text) == 1 && s.text[0] >= 'a' && s.text[0] <= 'z') {
			return one(phRecvType)
		}
	case phRecvType:
		switch {
		case s.kind == sStar:
			return one(phRecvTypeAtom)
		case isTypeAtom(con, s):
			return one(phRecvClose)
		}
	case phRecvTypeAtom:
		if isTypeAtom(con, s) {
			return one(phRecvClose)
		}
	case phRecvClose:
		if s.kind == sRParen {
			return one(phFuncName)
		}
	case phFuncName:
		if isName(con, s) {
			return one(phFuncParamsOpen)
		}
	case phFuncParamsOpen:
		if s.kind == sLParen {
			nc := c
			nc.phase = phParamsStart
			nc.exit = phFuncResults
			return []config{nc}
		}
	case phFuncResults:
		switch {
		case s.kind == sLBrace:
			return one(phBodyStart)
		case s.kind == sLParen:
			return one(phFuncResList)
		default:
			return enterType(c, s, phFuncResEnd, con)
		}
	case phFuncResList:
		return enterType(c, s, phFuncResListEnd, con)
	case phFuncResListEnd:
		switch s.kind {
		case sComma:
			return one(phFuncResList)
		case sRParen:
			return one(phFuncResDone)
		}
	case phFuncResEnd, phFuncResDone:
		if s.kind == sLBrace {
			return one(phBodyStart)
		}
	case phBodyStart:
		switch {
		case s.kind == sRBrace:
			return one(phDone)
		case s.kind == sAtom && s.text == "panic":
			return one(phPanicOpen)
		}
	case phPanicOpen:
		if s.kind == sLParen {
			return one(phPanicQuote1)
		}
	case phPanicQuote1:
		if s.kind == sQuote {
			return one(phPanicNot)
		}
	case phPanicNot:
		if s.kind == sAtom && s.text == "not" {
			return one(phPanicImpl)
		}
	case phPanicImpl:
		if s.kind == sAtom && s.text == "implemented" {
			return one(phPanicQuote2)
		}
	case phPanicQuote2:
		if s.kind == sQuote {
			return one(phPanicClose)
		}
	case phPanicClose:
		if s.kind == sRParen {
			return one(phPanicEnd)
		}
	case phPanicEnd:
		if s.kind == sRBrace {
			return one(phDone)
		}
	case phMethodResults:
		switch {
		case s.kind == sSemi:
			return one(phIfaceStart)
		case s.kind == sRBrace:
			return one(phDone)
		case s.kind == sLParen:
			return one(phMethodResList)
		default:
			return enterType(c, s, phMethodResEnd, con)
		}
	case phMethodResList:
		return enterType(c, s, phMethodResListEnd, con)
	case phMethodResListEnd:
		switch s.kind {
		case sComma:
			return one(phMethodResList)
		case sRParen:
			return one(phMethodResDone)
		}
	case phMethodResEnd, phMethodResDone:
		switch s.kind {
		case sSemi:
			return one(phIfaceStart)
		case sRBrace:
			return one(phDone)
		}
	}
	return nil
}

// acceptsEOS reports whether the machine is in a completed
// declaration state, so the EOS token may be emitted.
func acceptsEOS(cfgs []config) bool {
	for _, c := range cfgs {
		if !c.inType && c.phase == phDone {
			return true
		}
	}
	return false
}

// acceptsTokens reports whether the symbol stream of a full token
// sequence (a gold target, in tests and validation) is accepted: each
// token steps the machine, and EOS must be reachable at the end.
func (c *constraint) acceptsTokens(toks []string, wordSyms map[string][]sym) bool {
	cfgs := []config{{phase: phStart}}
	for _, t := range toks {
		syms, ok := wordSyms[t]
		if !ok {
			syms = symbolizeWord(t)
		}
		for _, s := range syms {
			cfgs = stepAll(cfgs, s, c)
			if len(cfgs) == 0 {
				return false
			}
		}
	}
	return acceptsEOS(cfgs)
}
