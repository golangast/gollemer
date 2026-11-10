package analyze

// The wasm events piece (chat/wasmevent.go): one event on
// a component already mounted in a wasm app, by sentence —
//
//	when the button is clicked, show the card
//	when the save button is clicked, set the card's words to 'Saved'
//	when the switch is flipped, toggle the card
//
// An event is a handler in Go: a js.FuncOf callback in the
// app's events.go, registered after main.go mounts the
// components, acting on another component by its element
// id (the ids the component piece stamps — editwasm.go).
// The closed sets keep the promise honest: clicks belong
// to buttons and cards, changes to the switch and the
// text field, and what can happen is show, hide, toggle,
// or set a button's or card's words — the components whose
// words are the whole of what they say.
//
// Every callback is kept in events.go's package-level
// eventFuncs list for the page's whole life: a js.FuncOf
// value the Go side lets go of is garbage-collected, and
// a collected callback silently stops answering in the
// browser — the one gotcha this file's shape exists to
// keep a beginner from ever meeting.

import (
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"
)

// wasmEventsRel names the app's events file.
func wasmEventsRel(folder string) string {
	return path.Join(folder, "events.go")
}

// The events each kind can answer, and the kinds whose
// words an event can set — the piece's closed sets, in
// one place, so the planner's refusals and the generated
// code read the same table.
var (
	wasmClickKinds    = map[string]bool{"button": true, "tonal-button": true, "outlined-button": true, "text-button": true, "fab": true, "card": true}
	wasmChangeKinds   = map[string]bool{"switch": true, "text-field": true}
	wasmSetWordsKinds = map[string]bool{"button": true, "tonal-button": true, "outlined-button": true, "text-button": true, "card": true}
)

// wasmEventsHeader is the events file's head: the build
// tag, the callback-keeping list with the words that say
// why it exists, the three helpers every show/hide/toggle
// handler calls, and registerEvents with the markers the
// registrations land between — written once, when the
// first event lands.
const wasmEventsHeader = `//go:build js && wasm

package main

import "syscall/js"

// The events on this app's page: what happens when a
// component is clicked or changed — added by sentence
// ("when the button is clicked, show the card"), one
// handler each. main calls registerEvents after the
// components are mounted, so every handler finds its
// component on the page.

// eventFuncs holds every event callback the page
// registers, for the page's whole life. It exists because
// of how js.FuncOf works: a callback the Go side lets go
// of is garbage-collected, and a collected callback
// silently stops answering in the browser — so each one
// is kept here, and never let go.
var eventFuncs []js.Func

// showElement makes the element with this id visible
// again, undoing a hide.
func showElement(doc js.Value, id string) {
	el := doc.Call("getElementById", id)
	if el.Truthy() {
		el.Get("style").Set("display", "")
	}
}

// hideElement hides the element with this id — the rest
// of the page moves up into its place.
func hideElement(doc js.Value, id string) {
	el := doc.Call("getElementById", id)
	if el.Truthy() {
		el.Get("style").Set("display", "none")
	}
}

// toggleElement shows the element with this id when it
// is hidden, and hides it when it is showing.
func toggleElement(doc js.Value, id string) {
	el := doc.Call("getElementById", id)
	if !el.Truthy() {
		return
	}
	if el.Get("style").Get("display").String() == "none" {
		showElement(doc, id)
	} else {
		hideElement(doc, id)
	}
}

// registerEvents wires every event handler in this file
// to its component. It runs once, after the components
// are mounted; the handlers are called between the
// markers, in the order they were asked for.
func registerEvents(doc js.Value) {
	// gollemer:events-begin
	// gollemer:events-end
}

`

// WasmComponentRefWords renders one component reference
// the way a sentence says it: "the save button" for a
// named component, "the button" for a kind alone.
func WasmComponentRefWords(kind, name string) string {
	if name != "" {
		if kind == "" {
			return "the " + name
		}
		return "the " + name + " " + MaterialComponentWords(kind)
	}
	return "the " + MaterialComponentWords(kind)
}

// wasmEventVerbWords renders an event's verb phrase for
// the sentence: a switch is flipped, a text field
// changes, everything else in the change set is changed.
func wasmEventVerbWords(req EditRequest) string {
	if req.EventName == "click" {
		return "is clicked"
	}
	switch req.EventSourceKind {
	case "switch":
		return "is flipped"
	case "text-field":
		return "changes"
	}
	return "is changed"
}

// WasmEventSentence renders one event request as the
// sentence that asks for it — the planner's summaries and
// refusals, the generated handler's comment, and the
// landing confirmation all speak this one rendering, so
// they cannot drift apart.
func WasmEventSentence(req EditRequest) string {
	s := "when " + WasmComponentRefWords(req.EventSourceKind, req.EventSourceName) +
		" " + wasmEventVerbWords(req) + ", "
	target := WasmComponentRefWords(req.EventTargetKind, req.EventTargetName)
	switch req.EventAction {
	case "show":
		s += "show " + target
	case "hide":
		s += "hide " + target
	case "toggle":
		s += "toggle " + target
	case "setwords":
		s += "set " + target + "'s words to '" + req.EventWords + "'"
	}
	return s
}

// ResolveWasmComponentRef settles one component reference
// — a kind, a name, or both — against an app's mounted
// instances. A name matches the component that carries it
// (as its name or, for a reference the clarifying answer
// rebuilt, its element id); a kind alone matches the
// kind's one instance. Exactly one fit returns it; more
// than one returns the candidates (the chat asks which);
// none returns neither.
func ResolveWasmComponentRef(instances []WasmComponentInstance, kind, name string) (match WasmComponentInstance, candidates []WasmComponentInstance) {
	var fits []WasmComponentInstance
	for _, in := range instances {
		if kind != "" && in.Kind != kind {
			continue
		}
		if name != "" && in.Name != name && in.ID != name {
			continue
		}
		fits = append(fits, in)
	}
	switch {
	case len(fits) == 1:
		return fits[0], nil
	case len(fits) > 1:
		return WasmComponentInstance{}, fits
	}
	return WasmComponentInstance{}, nil
}

// wasmInstanceList renders an app's mounted components
// for refusal prose: "a card, a button called save".
func wasmInstanceList(instances []WasmComponentInstance) string {
	parts := []string{}
	for _, in := range instances {
		s := "a " + MaterialComponentWords(in.Kind)
		if in.Name != "" {
			s += " called " + in.Name
		}
		parts = append(parts, s)
	}
	return strings.Join(parts, ", ")
}

// wasmCamelID renders an element id as a Go name part:
// "card-1" becomes Card1, "save" becomes Save.
func wasmCamelID(id string) string {
	parts := strings.Split(id, "-")
	for i, p := range parts {
		if p != "" {
			parts[i] = strings.ToUpper(p[:1]) + p[1:]
		}
	}
	return strings.Join(parts, "")
}

// wasmEventHandlerName names one event's handler from the
// components it joins: onSaveClickSetCardWords — the
// source, the event, the action, the target. The name is
// the event's identity in the file: the same event asked
// again computes the same name, which is how a repeat ask
// is recognised.
func wasmEventHandlerName(src WasmComponentInstance, eventName, action string, tgt WasmComponentInstance) string {
	ev := "Click"
	if eventName == "change" {
		ev = "Change"
	}
	switch action {
	case "show":
		return "on" + wasmCamelID(src.ID) + ev + "Show" + wasmCamelID(tgt.ID)
	case "hide":
		return "on" + wasmCamelID(src.ID) + ev + "Hide" + wasmCamelID(tgt.ID)
	case "toggle":
		return "on" + wasmCamelID(src.ID) + ev + "Toggle" + wasmCamelID(tgt.ID)
	case "setwords":
		return "on" + wasmCamelID(src.ID) + ev + "Set" + wasmCamelID(tgt.ID) + "Words"
	}
	return ""
}

// wasmEventHandler renders one handler: its comment (the
// sentence that asked for it), the callback it builds,
// the keep-alive append, and the registration on its
// component. show/hide/toggle call the file's helpers;
// setting words writes the target's text directly, with
// a guard for a page the user has since edited by hand.
func wasmEventHandler(fnName, comment string, src WasmComponentInstance, eventName, action string, tgt WasmComponentInstance, words string) string {
	var body string
	switch action {
	case "show":
		body = "\t\tshowElement(doc, " + strconv.Quote(tgt.ID) + ")\n"
	case "hide":
		body = "\t\thideElement(doc, " + strconv.Quote(tgt.ID) + ")\n"
	case "toggle":
		body = "\t\ttoggleElement(doc, " + strconv.Quote(tgt.ID) + ")\n"
	case "setwords":
		body = "\t\ttarget := doc.Call(\"getElementById\", " + strconv.Quote(tgt.ID) + ")\n" +
			"\t\tif target.Truthy() {\n" +
			"\t\t\ttarget.Set(\"textContent\", " + strconv.Quote(words) + ")\n" +
			"\t\t}\n"
	}
	return "// " + comment + "\n" +
		"func " + fnName + "(doc js.Value) {\n" +
		"\tel := doc.Call(\"getElementById\", " + strconv.Quote(src.ID) + ")\n" +
		"\tfn := js.FuncOf(func(this js.Value, args []js.Value) interface{} {\n" +
		body +
		"\t\treturn nil\n" +
		"\t})\n" +
		"\teventFuncs = append(eventFuncs, fn)\n" +
		"\tel.Call(\"addEventListener\", " + strconv.Quote(eventName) + ", fn)\n" +
		"}\n"
}

// wasmHandlerWords reads the words an existing set-words
// handler sets, out of events.go content — the repeat
// ask's comparison: the same words are the already-there
// answer, different words are a conflict, both named.
func wasmHandlerWords(eventsContent, fnName string) (string, bool) {
	i := strings.Index(eventsContent, "func "+fnName+"(")
	if i < 0 {
		return "", false
	}
	body := eventsContent[i:]
	if j := strings.Index(body, "\n}\n"); j >= 0 {
		body = body[:j]
	}
	const probe = `Set("textContent", `
	k := strings.Index(body, probe)
	if k < 0 {
		return "", false
	}
	rest := body[k+len(probe):]
	if rest == "" || rest[0] != '"' {
		return "", false
	}
	end := 1
	for end < len(rest) {
		if rest[end] == '\\' {
			end += 2
			continue
		}
		if rest[end] == '"' {
			break
		}
		end++
	}
	if end >= len(rest) {
		return "", false
	}
	words, err := strconv.Unquote(rest[:end+1])
	if err != nil {
		return "", false
	}
	return words, true
}

// stampExistingWasmBuilder writes the id stamp onto one
// builder already in components.go — the backfill for a
// builder the component piece wrote before builders
// carried ids. The stamp lands under the builder's root
// creation line, exactly where stampWasmBuilderID puts
// it on a new builder; a pre-stamp chip builder, whose
// root is its return line, is first given the variable
// every other builder keeps its root in. A builder whose
// root line is neither shape reports false — the caller
// refuses rather than guess at hand-shaped code.
func stampExistingWasmBuilder(compContent, fn, kindID, elemID string) (string, bool) {
	start := strings.Index(compContent, "func "+fn+"(")
	if start < 0 {
		return compContent, false
	}
	lines := strings.SplitAfter(compContent, "\n")
	// The line offset where this builder starts and ends:
	// its func line, and the "}" line that closes it.
	startLine, endLine := -1, -1
	pos := 0
	for i, ln := range lines {
		if pos+len(ln) > start && startLine < 0 {
			startLine = i
		}
		if startLine >= 0 && (ln == "}\n" || ln == "}") {
			endLine = i
			break
		}
		pos += len(ln)
	}
	if startLine < 0 || endLine < 0 {
		return compContent, false
	}
	rootVar := wasmBuilderRootVar[kindID]
	for i := startLine; i <= endLine; i++ {
		ln := lines[i]
		if strings.Contains(ln, rootVar+`.Set("id", `) {
			return compContent, true // the root is stamped already — nothing to do
		}
		if strings.Contains(ln, ":= el(doc,") {
			stamp := "\t" + rootVar + ".Set(\"id\", " + strconv.Quote(elemID) + ")\n"
			out := append([]string{}, lines[:i+1]...)
			out = append(out, stamp)
			out = append(out, lines[i+1:]...)
			return strings.Join(out, ""), true
		}
		if trimmed := strings.TrimSpace(ln); strings.HasPrefix(trimmed, "return el(doc,") {
			indent := ln[:len(ln)-len(strings.TrimLeft(ln, " \t"))]
			creation := strings.TrimPrefix(trimmed, "return ")
			out := append([]string{}, lines[:i]...)
			out = append(out,
				indent+rootVar+" := "+creation+"\n",
				indent+rootVar+".Set(\"id\", "+strconv.Quote(elemID)+")\n",
				indent+"return "+rootVar+"\n")
			out = append(out, lines[i+1:]...)
			return strings.Join(out, ""), true
		}
	}
	return compContent, false
}

// planWasmEvent assembles one event: the handler joins
// the app's events.go (created with its header on first
// use), its registration lands between the register
// markers in ask order, main.go gains the registerEvents
// call right after the mounts when it doesn't call it
// yet, and the two components the event joins get their
// id stamps backfilled when their builders predate ids —
// a handler finds its components by id, so an id the page
// doesn't carry would be a promise the code can't keep.
func (p *Project) planWasmEvent(req EditRequest) (*EditPlan, string) {
	folder := path.Clean(filepath.ToSlash(strings.TrimSpace(req.WasmFolder)))
	_, mainRel, _, _, _ := wasmAppFiles(folder)
	compRel := wasmComponentsRel(folder)
	eventsRel := wasmEventsRel(folder)
	mainBefore, err := os.ReadFile(filepath.Join(p.Root, filepath.FromSlash(mainRel)))
	if err != nil {
		return nil, fmt.Sprintf("there's no wasm app in %s yet — say \"add a wasm app in %s\" first, then the event can go into it.", folder, folder)
	}
	main := string(mainBefore)
	if !wasmAppMainContent(main) {
		return nil, fmt.Sprintf("%s isn't a wasm app's main file — it doesn't carry the js && wasm build tag a wasm app's main wears — so the event has no app to join. Say \"add a wasm app in %s\" to make one.", p.LinkPath(mainRel), folder)
	}
	if !strings.Contains(main, "mountComponents(doc, app)") {
		return nil, fmt.Sprintf("%s doesn't call mountComponents the way the wasm piece writes it — an event's handler registers right after the components mount, and without that call it has nowhere honest to register. Nothing was written.", p.LinkPath(mainRel))
	}
	compBefore, _ := os.ReadFile(filepath.Join(p.Root, filepath.FromSlash(compRel)))
	compContent := string(compBefore)
	instances := wasmComponentInstances(main, compContent)
	for _, rel := range []string{mainRel, compRel, eventsRel} {
		if why, blocked := p.fileBlocked(rel); blocked {
			return nil, fmt.Sprintf("I won't write %s — %s.", p.LinkPath(rel), why)
		}
	}

	// Both references resolve against what's mounted —
	// the planner re-reads them here, at plan time, so a
	// sentence completed from a clarifying answer faces
	// the app as it stands now, not as it stood when the
	// question was asked.
	src, srcCands := ResolveWasmComponentRef(instances, req.EventSourceKind, req.EventSourceName)
	if len(srcCands) > 0 {
		return nil, fmt.Sprintf("the wasm app in %s has more than one %s — %s — and the sentence doesn't say which one the event starts from. A name settles it: \"add a button called save\" gives a button a name an event can use.", folder, MaterialComponentWords(req.EventSourceKind), wasmInstanceList(srcCands))
	}
	if src.Builder == "" {
		if req.EventSourceName != "" {
			return nil, fmt.Sprintf("the wasm app in %s has no component called %s — the components it has are: %s. An event starts from a component that's there.", folder, req.EventSourceName, wasmInstanceListOrNone(instances))
		}
		return nil, fmt.Sprintf("the wasm app in %s has no %s in it yet — an event starts from a component that's there. Add one first: \"add a %s to %s\".", folder, MaterialComponentWords(req.EventSourceKind), MaterialComponentWords(req.EventSourceKind), folder)
	}
	tgt, tgtCands := ResolveWasmComponentRef(instances, req.EventTargetKind, req.EventTargetName)
	if len(tgtCands) > 0 {
		return nil, fmt.Sprintf("the wasm app in %s has more than one %s — %s — and the sentence doesn't say which one the event acts on. A name settles it: \"add a button called save\" gives a button a name an event can use.", folder, MaterialComponentWords(req.EventTargetKind), wasmInstanceList(tgtCands))
	}
	if tgt.Builder == "" {
		if req.EventTargetName != "" {
			return nil, fmt.Sprintf("the wasm app in %s has no component called %s — the components it has are: %s. An event acts on a component that's there.", folder, req.EventTargetName, wasmInstanceListOrNone(instances))
		}
		return nil, fmt.Sprintf("the wasm app in %s has no %s in it yet — the event needs a %s that's there to act on. Add one first: \"add a %s to %s\".", folder, MaterialComponentWords(req.EventTargetKind), MaterialComponentWords(req.EventTargetKind), MaterialComponentWords(req.EventTargetKind), folder)
	}

	// The closed sets: the event must be one the source
	// can answer, and set-words must land on a component
	// whose words are its whole content.
	switch req.EventName {
	case "click":
		if !wasmClickKinds[src.Kind] {
			if wasmChangeKinds[src.Kind] {
				return nil, fmt.Sprintf("a %s has no click to answer — it answers to being changed instead: \"when %s, …\" is its event. Clicks belong to buttons and cards.", MaterialComponentWords(src.Kind), WasmComponentRefWords(src.Kind, src.Name)+" "+wasmEventVerbWords(EditRequest{EventName: "change", EventSourceKind: src.Kind}))
			}
			return nil, fmt.Sprintf("a %s doesn't take a click in this set — clicks belong to buttons and cards, and a switch or a text field answers to being changed instead.", MaterialComponentWords(src.Kind))
		}
	case "change":
		if !wasmChangeKinds[src.Kind] {
			return nil, fmt.Sprintf("a %s doesn't change in this set — changing belongs to the switch and the text field. A button or a card answers to a click instead: \"when %s is clicked, …\".", MaterialComponentWords(src.Kind), WasmComponentRefWords(src.Kind, src.Name))
		}
	default:
		return nil, "that event isn't one of this set: a click, or a change."
	}
	switch req.EventAction {
	case "show", "hide", "toggle":
	case "setwords":
		if !wasmSetWordsKinds[tgt.Kind] {
			return nil, fmt.Sprintf("only a button's or a card's words can be set — their words are the whole of what they say. A %s has more inside it than one line of words, so its words stay its own.", MaterialComponentWords(tgt.Kind))
		}
	default:
		return nil, "that isn't something an event can do in this set: show, hide, toggle, or set a component's words."
	}

	// The resolved request renders every word the plan
	// speaks — summary, handler comment, confirmation —
	// from the components as they actually are.
	res := req
	res.EventSourceKind, res.EventTargetKind = src.Kind, tgt.Kind
	sentence := WasmEventSentence(res)
	handlerName := wasmEventHandlerName(src, req.EventName, req.EventAction, tgt)

	eventsBefore, _ := os.ReadFile(filepath.Join(p.Root, filepath.FromSlash(eventsRel)))
	events := string(eventsBefore)
	if events != "" && (!strings.Contains(events, "// gollemer:events-begin") || !strings.Contains(events, "// gollemer:events-end")) {
		return nil, fmt.Sprintf("%s doesn't have the events block — the two gollemer:events marker comments in its register function — so this event has nowhere honest to land. Nothing was written.", p.LinkPath(eventsRel))
	}
	if strings.Contains(events, "func "+handlerName+"(") {
		if req.EventAction == "setwords" {
			if existing, ok := wasmHandlerWords(events, handlerName); ok {
				if existing == req.EventWords {
					return nil, fmt.Sprintf("the wasm app in %s already does that — %s — nothing to add.", folder, sentence)
				}
				return nil, fmt.Sprintf("the wasm app in %s already answers that event by setting %s's words to '%s' — one event sets one set of words, so nothing was added.", folder, WasmComponentRefWords(tgt.Kind, tgt.Name), existing)
			}
		}
		return nil, fmt.Sprintf("the wasm app in %s already does that — %s — nothing to add.", folder, sentence)
	}

	// The handler and its registration.
	comment := "When " + strings.TrimPrefix(sentence, "when ")
	handler := wasmEventHandler(handlerName, comment, src, req.EventName, req.EventAction, tgt, req.EventWords)
	base := events
	if base == "" {
		base = wasmEventsHeader
	}
	end := strings.Index(base, "// gollemer:events-end")
	endLine := strings.LastIndex(base[:end], "\n") + 1
	newEvents := base[:endLine] + "\t" + handlerName + "(doc)\n" + base[endLine:]
	newEvents = strings.TrimRight(newEvents, "\n") + "\n\n" + handler

	// main.go gains the register call right after the
	// mounts, the first time an event lands.
	newMain := main
	mainTouched := false
	if !strings.Contains(main, "registerEvents(doc)") {
		i := strings.Index(main, "mountComponents(doc, app)")
		lineEnd := strings.Index(main[i:], "\n") + i
		newMain = main[:lineEnd+1] + "\tregisterEvents(doc)\n" + main[lineEnd+1:]
		mainTouched = true
	}

	// The id backfill: the two components this event
	// joins, when their builders predate id stamps.
	newComp := compContent
	compTouched := false
	var backfilled []WasmComponentInstance
	for _, inst := range []WasmComponentInstance{src, tgt} {
		if _, stamped := wasmBuilderStamp(newComp, inst.Builder); stamped {
			continue
		}
		updated, ok := stampExistingWasmBuilder(newComp, inst.Builder, inst.Kind, inst.ID)
		if !ok {
			return nil, fmt.Sprintf("the builder for %s in %s isn't shaped the way the wasm piece writes it — its root element has no line an id can be stamped on — so the event would have no way to find it. Nothing was written.", WasmComponentRefWords(inst.Kind, inst.Name), p.LinkPath(compRel))
		}
		if updated != newComp {
			newComp = updated
			compTouched = true
			backfilled = append(backfilled, inst)
		}
	}

	files := []EditFile{finishFile(eventsRel, eventsBefore, []byte(newEvents))}
	if mainTouched {
		files = append(files, finishFile(mainRel, mainBefore, []byte(newMain)))
	}
	if compTouched {
		files = append(files, finishFile(compRel, compBefore, []byte(newComp)))
	}
	notes := []string{}
	if len(eventsBefore) == 0 {
		notes = append(notes, p.LinkPath(eventsRel)+" is new: it holds the app's event handlers, and the eventFuncs list that keeps every callback alive for the page's whole life — a callback Go lets go of is garbage-collected, and a collected callback silently stops answering.")
	}
	if len(backfilled) > 0 {
		parts := []string{}
		for _, inst := range backfilled {
			parts = append(parts, fmt.Sprintf("%s (id %q)", WasmComponentRefWords(inst.Kind, inst.Name), inst.ID))
		}
		notes = append(notes, "The event finds its components by id, and "+strings.Join(parts, " and ")+" had no id yet — their builders in "+p.LinkPath(compRel)+" carry one now.")
	}
	notes = append(notes, "Rebuild the app in "+folder+" ("+WasmBuildCommand+") and the event does its work in the browser.")
	summary := fmt.Sprintf("Give the wasm app in %s an event: %s. The handler %s joins %s, registered after the components mount in %s.", folder, sentence, handlerName, p.LinkPath(eventsRel), p.LinkPath(mainRel))
	return &EditPlan{
		Summary:    summary,
		VerifyDesc: fmt.Sprintf("the wasm app in %s now answers: %s", folder, sentence),
		Site:       p.LinkPath(eventsRel),
		Files:      files,
		Notes:      notes,
		verify: func(after *Project) bool {
			b, err := os.ReadFile(filepath.Join(after.Root, filepath.FromSlash(eventsRel)))
			if err != nil || string(b) != newEvents {
				return false
			}
			m, err := os.ReadFile(filepath.Join(after.Root, filepath.FromSlash(mainRel)))
			if err != nil || string(m) != newMain {
				return false
			}
			if compTouched {
				c, err := os.ReadFile(filepath.Join(after.Root, filepath.FromSlash(compRel)))
				if err != nil || string(c) != newComp {
					return false
				}
			}
			dir := folder
			if dir == "." {
				dir = ""
			}
			for _, pkg := range after.Packages {
				if pkg.Dir != dir {
					continue
				}
				for _, f := range pkg.Funcs {
					if f.Receiver == "" && f.Name == handlerName {
						return true
					}
				}
			}
			return false
		},
	}, ""
}

// wasmInstanceListOrNone renders the mounted list for a
// refusal, or the honest word when the app has none.
func wasmInstanceListOrNone(instances []WasmComponentInstance) string {
	if len(instances) == 0 {
		return "none yet"
	}
	return wasmInstanceList(instances)
}
