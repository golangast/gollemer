package analyze

// Taught components: the teach seat's component writer.
// When the user teaches the chat a component from a
// builder example in their own project ("learn this
// component as a price tag from PriceTag in
// jim/components.go"), the chat layer stores the builder's
// source as a template with two slots — the builder's own
// name, and the string in its root element's id stamp. A
// later ask ("add a price tag to jim", "add a price tag
// called total to jim") arrives here as data and lands
// exactly the way a registry component lands (planComponentWasm
// and its part twin): the builder — stamped with the
// instance's builder name and element id, and headed by
// the marker line that says what it is — joins the app's
// components.go (or the part's), one mount line lands in
// ask order, and the same name and already-there rules
// guard the app. The template is the user's own code,
// carried verbatim; nothing here invents code.
//
// Two honest differences from a registry landing, both
// stated in the plan's own words: no styles travel with a
// taught component in this version (the registry's chunks
// are keyed by registry kind; a taught builder's classes
// come with it, but the host page is untouched), and the
// builder carries the taught marker, so the app's files
// themselves record the kind and the root variable every
// later reading (instances, events) works from.

import (
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"
)

// planTaughtComponent assembles one taught component into
// a wasm app: the body, or one part of it when
// ComponentPart names one.
func (p *Project) planTaughtComponent(req EditRequest) (*EditPlan, string) {
	if strings.TrimSpace(req.TaughtComponentTemplate) == "" || strings.TrimSpace(req.TaughtComponentRootVar) == "" {
		return nil, "that edit needs a taught component to add, and none was supplied."
	}
	if req.ComponentPart != "" {
		return p.planTaughtComponentWasmPart(req)
	}
	return p.planTaughtComponentWasm(req)
}

// taughtComponentWords is the plan's name for the kind:
// the user's own phrase, defaulted like planTaughtDecl's.
func taughtComponentWords(req EditRequest) string {
	words := strings.TrimSpace(req.TaughtKindPhrase)
	if words == "" {
		return "component"
	}
	return words
}

// renderTaughtBuilder stamps one instance out of the
// taught template: the marker line that says what the
// builder is, then the template with its name slot taken
// by this instance's builder name and its id slot by this
// instance's element id.
func renderTaughtBuilder(req EditRequest, fn, elemID string) string {
	text := strings.ReplaceAll(req.TaughtComponentTemplate, TaughtNameSlot, fn)
	text = strings.ReplaceAll(text, TaughtIDSlot, elemID)
	return taughtComponentMarkerLine(taughtComponentWords(req), req.TaughtComponentRootVar) + "\n" +
		strings.TrimRight(text, "\n") + "\n"
}

// taughtComponentNotes is the plan's notes for one taught
// landing: the provenance first (the teach seat's line,
// when the chat supplied one), then the render note every
// wasm component carries, then — when the example's
// builder gives its elements classes — the styles
// honesty: classes come with the builder, styles do not.
func taughtComponentNotes(req EditRequest, words, elemID string) []string {
	var notes []string
	if req.TaughtNote != "" {
		notes = append(notes, req.TaughtNote)
	}
	notes = append(notes, "The "+words+" renders when the app is built and the page loads — it is a Go builder in the app, not HTML in the page. Its element carries the id \""+elemID+"\", the name an event finds it by: \"when the button is clicked, show the "+words+"\" is how one is added.")
	if req.TaughtComponentStyled {
		notes = append(notes, "Its classes come with it, exactly as your example wrote them, but no styles travel with a taught component in this version — the page shows it styled only where its own styles already cover those classes.")
	}
	return notes
}

// planTaughtComponentWasm is the taught component's body
// mode — planComponentWasm's discipline, scoped to a
// taught kind: the builder joins the app's components.go
// (created with the shared helper on first use), one mount
// line lands between main.go's markers in ask order, and
// the host page is never touched (no styles travel). A
// named ask is a distinct instance — its own builder, its
// name as its element id; an unnamed ask is the kind's one
// unnamed instance in the body, with a positional id: the
// kind's slug and its place among its kind in the app.
func (p *Project) planTaughtComponentWasm(req EditRequest) (*EditPlan, string) {
	words := taughtComponentWords(req)
	folder := path.Clean(filepath.ToSlash(strings.TrimSpace(req.WasmFolder)))
	_, mainRel, _, indexRel, _ := wasmAppFiles(folder)
	compRel := wasmComponentsRel(folder)
	mainBefore, err := os.ReadFile(filepath.Join(p.Root, filepath.FromSlash(mainRel)))
	if err != nil {
		return nil, fmt.Sprintf("there's no wasm app in %s yet — say \"add a wasm app in %s\" first, then the %s can go into it.", folder, folder, words)
	}
	main := string(mainBefore)
	if !wasmAppMainContent(main) {
		return nil, fmt.Sprintf("%s isn't a wasm app's main file — it doesn't carry the js && wasm build tag a wasm app's main wears — so the %s has no app to join. Say \"add a wasm app in %s\" to make one, or add the %s to a page instead.", p.LinkPath(mainRel), words, folder, words)
	}
	begin := strings.Index(main, "// gollemer:components-begin")
	end := strings.Index(main, "// gollemer:components-end")
	if begin < 0 || end < 0 || end < begin {
		return nil, fmt.Sprintf("%s doesn't have the components block the wasm piece mounts into — the two gollemer:components marker comments in its mount function — so the %s has nowhere honest to land. Nothing was written.", p.LinkPath(mainRel), words)
	}
	if _, err := os.ReadFile(filepath.Join(p.Root, filepath.FromSlash(indexRel))); err != nil {
		return nil, fmt.Sprintf("%s isn't there — the host page of the app in %s is missing, so this isn't the app the wasm piece makes. Nothing was written.", p.LinkPath(indexRel), folder)
	}
	for _, rel := range []string{mainRel, compRel} {
		if why, blocked := p.fileBlocked(rel); blocked {
			return nil, fmt.Sprintf("I won't write %s — %s.", p.LinkPath(rel), why)
		}
	}
	compBefore, _ := os.ReadFile(filepath.Join(p.Root, filepath.FromSlash(compRel)))
	compContent := string(compBefore)
	allInstances := p.wasmAppAllInstances(folder, main, compContent)
	name := strings.TrimSpace(req.ComponentName)
	var fn, elemID string
	if name != "" {
		for _, in := range allInstances {
			if in.Name != name && in.ID == name {
				return nil, fmt.Sprintf("the wasm app in %s already has a %s whose id is %s — an event would find that one by this name, so it can't be given to another component. Pick another name.", folder, MaterialComponentWords(in.Kind), name)
			}
		}
		for _, in := range allInstances {
			if in.Name != name {
				continue
			}
			if in.Kind == words {
				return nil, fmt.Sprintf("the wasm app in %s already has a %s called %s — nothing to add.", folder, words, name)
			}
			return nil, fmt.Sprintf("the wasm app in %s already has a component called %s — its %s — and a name belongs to one component. Pick another name, or add the %s without one.", folder, name, MaterialComponentWords(in.Kind), words)
		}
		fn = wasmTaughtNamedBuilderName(name, words)
		if strings.Contains(compContent, "func "+fn+"(") {
			return nil, fmt.Sprintf("%s already declares %s — the builder name a %s called %s would take — so nothing was written. Rename that function, or pick another name.", p.LinkPath(compRel), fn, words, name)
		}
		elemID = name
	} else {
		fn = wasmTaughtBuilderName(words)
		for _, in := range allInstances {
			if in.Taught && in.Kind == words && in.Part == "" && in.Name == "" {
				return nil, fmt.Sprintf("the wasm app in %s already has a %s in it — nothing to add.", folder, words)
			}
		}
		if strings.Contains(compContent, "func "+fn+"(") {
			return nil, fmt.Sprintf("%s already declares %s — the builder name a %s would take — but no %s is mounted in the app. Nothing was written. Rename that function, or give the %s a name: \"add a %s called <name> to %s\".", p.LinkPath(compRel), fn, words, words, words, words, folder)
		}
		elemID = wasmTaughtKindSlug(words) + "-" + strconv.Itoa(wasmKindCount(allInstances, words)+1)
	}
	builder := renderTaughtBuilder(req, fn, elemID)
	endLine := strings.LastIndex(main[:end], "\n") + 1
	newMain := main[:endLine] + "\tapp.Call(\"appendChild\", " + fn + "(doc))\n" + main[endLine:]
	newComp := ""
	if compContent == "" {
		newComp = wasmComponentsHeader + builder
	} else {
		newComp = strings.TrimRight(compContent, "\n") + "\n\n" + builder
	}
	files := []EditFile{finishFile(compRel, compBefore, []byte(newComp)), finishFile(mainRel, mainBefore, []byte(newMain))}
	what := words
	if name != "" {
		what += " called " + name
	}
	summary := fmt.Sprintf("Put a %s in the wasm app in %s: %s joins %s and is mounted on the page in %s", what, folder, fn, p.LinkPath(compRel), p.LinkPath(mainRel))
	return &EditPlan{
		Summary:    summary,
		VerifyDesc: fmt.Sprintf("the wasm app in %s now has a %s mounted on its page", folder, what),
		Site:       p.LinkPath(mainRel),
		Files:      files,
		Notes:      taughtComponentNotes(req, words, elemID),
		verify: func(after *Project) bool {
			b, err := os.ReadFile(filepath.Join(after.Root, filepath.FromSlash(compRel)))
			if err != nil || string(b) != newComp {
				return false
			}
			m, err := os.ReadFile(filepath.Join(after.Root, filepath.FromSlash(mainRel)))
			if err != nil || string(m) != newMain {
				return false
			}
			dir := folder
			if dir == "." {
				dir = "" // the analyzer names the root package's dir ""
			}
			for _, pkg := range after.Packages {
				if pkg.Dir != dir {
					continue
				}
				for _, f := range pkg.Funcs {
					if f.Receiver == "" && f.Name == fn {
						return true
					}
				}
			}
			return false
		},
	}, ""
}

// planTaughtComponentWasmPart is the taught component's
// part mode — planComponentWasmPart's discipline, scoped
// to a taught kind: the builder joins the part's
// components.go (created with the part's header on first
// use) and one mount line lands between the part's
// markers in ask order. Names still belong to one
// component in the whole app; an unnamed instance is per
// part, and its id counts its kind across the whole app.
func (p *Project) planTaughtComponentWasmPart(req EditRequest) (*EditPlan, string) {
	words := taughtComponentWords(req)
	folder := path.Clean(filepath.ToSlash(strings.TrimSpace(req.WasmFolder)))
	part := req.ComponentPart
	if part != "header" && part != "footer" {
		return nil, "that edit names no part of the app — a header or a footer."
	}
	_, mainRel, _, indexRel, _ := wasmAppFiles(folder)
	compRel := wasmComponentsRel(folder)
	partRel := wasmPartFileRel(folder, part)
	partCompRel := wasmPartComponentsRel(folder, part)
	mainBefore, err := os.ReadFile(filepath.Join(p.Root, filepath.FromSlash(mainRel)))
	if err != nil {
		return nil, fmt.Sprintf("there's no wasm app in %s yet — say \"add a wasm app in %s\" first, then the %s can go into its %s.", folder, folder, words, part)
	}
	main := string(mainBefore)
	if !wasmAppMainContent(main) {
		return nil, fmt.Sprintf("%s isn't a wasm app's main file — it doesn't carry the js && wasm build tag a wasm app's main wears — so the %s has no app to join. Say \"add a wasm app in %s\" to make one, or add the %s to a page instead.", p.LinkPath(mainRel), words, folder, words)
	}
	if _, err := os.ReadFile(filepath.Join(p.Root, filepath.FromSlash(indexRel))); err != nil {
		return nil, fmt.Sprintf("%s isn't there — the host page of the app in %s is missing, so this isn't the app the wasm piece makes. Nothing was written.", p.LinkPath(indexRel), folder)
	}
	partBefore, err := os.ReadFile(filepath.Join(p.Root, filepath.FromSlash(partRel)))
	if err != nil {
		return nil, fmt.Sprintf("the wasm app in %s has no %s yet — say \"add a %s to the wasm app in %s\" first, then the %s can go into it.", folder, part, part, folder, words)
	}
	partContent := string(partBefore)
	if !strings.Contains(partContent, "package "+part) || !strings.Contains(partContent, "func "+wasmPartExported(part)+"(") {
		return nil, fmt.Sprintf("%s isn't the app's %s part — it doesn't carry the %s package and its %s builder the parts piece writes — so the %s has no part to join. Nothing was written.", p.LinkPath(partRel), part, part, wasmPartExported(part), words)
	}
	beginMark, endMark := wasmPartMarkers(part)
	begin := strings.Index(partContent, beginMark)
	end := strings.Index(partContent, endMark)
	if begin < 0 || end < 0 || end < begin {
		return nil, fmt.Sprintf("%s doesn't have the %s components block — the two %s marker comments in its mount function — so the %s has nowhere honest to land. Nothing was written.", p.LinkPath(partRel), part, part, words)
	}
	for _, rel := range []string{partRel, partCompRel} {
		if why, blocked := p.fileBlocked(rel); blocked {
			return nil, fmt.Sprintf("I won't write %s — %s.", p.LinkPath(rel), why)
		}
	}
	partCompBefore, _ := os.ReadFile(filepath.Join(p.Root, filepath.FromSlash(partCompRel)))
	partCompContent := string(partCompBefore)
	compBefore, _ := os.ReadFile(filepath.Join(p.Root, filepath.FromSlash(compRel)))
	partInstances := wasmPartInstances(partContent, partCompContent, part)
	allInstances := p.wasmAppAllInstances(folder, main, string(compBefore))
	name := strings.TrimSpace(req.ComponentName)
	var fn, elemID string
	if name != "" {
		for _, in := range allInstances {
			if in.Name != name && in.ID == name {
				return nil, fmt.Sprintf("the wasm app in %s already has a %s whose id is %s — an event would find that one by this name, so it can't be given to another component. Pick another name.", folder, MaterialComponentWords(in.Kind), name)
			}
		}
		for _, in := range allInstances {
			if in.Name != name {
				continue
			}
			if in.Kind == words {
				return nil, fmt.Sprintf("the wasm app in %s already has a %s called %s — nothing to add.", folder, words, name)
			}
			return nil, fmt.Sprintf("the wasm app in %s already has a component called %s — its %s — and a name belongs to one component. Pick another name, or add the %s without one.", folder, name, MaterialComponentWords(in.Kind), words)
		}
		fn = wasmTaughtNamedBuilderName(name, words)
		if strings.Contains(partCompContent, "func "+fn+"(") || strings.Contains(partContent, "func "+fn+"(") {
			return nil, fmt.Sprintf("the %s part of the wasm app in %s already declares %s — the builder name a %s called %s would take — so nothing was written. Rename that function, or pick another name.", part, folder, fn, words, name)
		}
		elemID = name
	} else {
		fn = wasmTaughtBuilderName(words)
		taken := false
		for _, in := range partInstances {
			if in.Taught && in.Kind == words && in.Name == "" {
				taken = true
			}
		}
		if taken {
			return nil, fmt.Sprintf("the %s of the wasm app in %s already has a %s in it — nothing to add.", part, folder, words)
		}
		if strings.Contains(partCompContent, "func "+fn+"(") || strings.Contains(partContent, "func "+fn+"(") {
			return nil, fmt.Sprintf("the %s part of the wasm app in %s already declares %s — the builder name a %s would take — but no %s is mounted in the %s. Nothing was written. Rename that function, or give the %s a name: \"add a %s called <name> to the %s in %s\".", part, folder, fn, words, words, part, words, words, part, folder)
		}
		elemID = wasmTaughtKindSlug(words) + "-" + strconv.Itoa(wasmKindCount(allInstances, words)+1)
	}
	builder := renderTaughtBuilder(req, fn, elemID)
	endLine := strings.LastIndex(partContent[:end], "\n") + 1
	newPart := partContent[:endLine] + "\troot.Call(\"appendChild\", " + fn + "(doc))\n" + partContent[endLine:]
	newPartComp := ""
	if partCompContent == "" {
		newPartComp = wasmPartComponentsHeader(part) + builder
	} else {
		newPartComp = strings.TrimRight(partCompContent, "\n") + "\n\n" + builder
	}
	files := []EditFile{finishFile(partCompRel, partCompBefore, []byte(newPartComp)), finishFile(partRel, partBefore, []byte(newPart))}
	what := words
	if name != "" {
		what += " called " + name
	}
	summary := fmt.Sprintf("Put a %s in the %s of the wasm app in %s: %s joins %s and is mounted inside the %s in %s", what, part, folder, fn, p.LinkPath(partCompRel), part, p.LinkPath(partRel))
	return &EditPlan{
		Summary:    summary,
		VerifyDesc: fmt.Sprintf("the %s of the wasm app in %s now has a %s mounted inside it", part, folder, what),
		Site:       p.LinkPath(partRel),
		Files:      files,
		Notes:      taughtComponentNotes(req, words, elemID),
		verify: func(after *Project) bool {
			b, err := os.ReadFile(filepath.Join(after.Root, filepath.FromSlash(partCompRel)))
			if err != nil || string(b) != newPartComp {
				return false
			}
			m, err := os.ReadFile(filepath.Join(after.Root, filepath.FromSlash(partRel)))
			if err != nil || string(m) != newPart {
				return false
			}
			dir := wasmPartDirRel(folder, part)
			for _, pkg := range after.Packages {
				if pkg.Dir != dir {
					continue
				}
				for _, f := range pkg.Funcs {
					if f.Receiver == "" && f.Name == fn {
						return true
					}
				}
			}
			return false
		},
	}, ""
}
