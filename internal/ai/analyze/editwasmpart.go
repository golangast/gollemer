package analyze

// The wasm parts piece (chat/wasmpart.go): a wasm app's
// header and footer as parts of the app — each its own
// package inside the app's folder:
//
//	<app>/header/header.go   package header, Header(doc)
//	<app>/footer/footer.go   package footer, Footer(doc)
//
// A part is a package, not a block in main.go, for the
// same reason the app is its own module: the part can
// grow — its own components land in the part's own
// components.go, mounted inside the part between its own
// markers — without crowding the app's main program, and
// main.go stays the small thing that mounts header, body,
// footer in page order:
//
//	app.Call("appendChild", header.Header(doc))
//	mountComponents(doc, app)
//	app.Call("appendChild", footer.Footer(doc))
//
// A part's standard content IS the component it replaces:
// the header part's root is the app-bar builder, the
// footer part's root is the footer builder — the same
// templates, classes, and TODO words as their body twins.
// So when the body already mounts an app bar (or a
// footer), giving the app a header (or a footer) MOVES
// that component into the part: its builder relocates
// into the part file with its content — and its name and
// id — preserved, its body mount line goes away, and
// nothing renders twice. A move that can't be read safely
// refuses the part plainly rather than duplicating.
//
// Components land in parts through the component kind
// with ComponentPart set (planComponentWasmPart below):
// the builder joins the part's components.go, the mount
// lands inside the part's marker block, and the styles
// join the host page exactly once, as in the body. The
// events planner reads body + parts as one app
// (wasmAppAllInstances), so an event reaches a component
// wherever in the app it lives, by the same ids.

import (
	"fmt"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// wasmPartNames is the closed set of parts, in page
// order: the header mounts before the body components,
// the footer after them.
var wasmPartNames = []string{"header", "footer"}

// wasmPartRootKind is a part's root component kind: the
// header is the app bar, the footer is the footer.
func wasmPartRootKind(part string) string {
	if part == "header" {
		return "app-bar"
	}
	return "footer"
}

// wasmPartExported is a part's exported builder: the
// function main.go calls to get the part's root element.
func wasmPartExported(part string) string {
	if part == "header" {
		return "Header"
	}
	return "Footer"
}

// wasmPartMountFunc is a part's internal mount function:
// the one that puts the part's own added components
// inside its root, between the part's markers.
func wasmPartMountFunc(part string) string {
	return "mount" + wasmPartExported(part) + "Components"
}

// wasmPartMarkers are a part's component markers — the
// part-scoped twin of main's gollemer:components markers.
func wasmPartMarkers(part string) (begin, end string) {
	return "// gollemer:" + part + "-components-begin", "// gollemer:" + part + "-components-end"
}

// wasmPartDirRel names a part's folder inside the app.
func wasmPartDirRel(folder, part string) string {
	return path.Join(folder, part)
}

// wasmPartFileRel names a part's own file: header.go in
// header/, footer.go in footer/.
func wasmPartFileRel(folder, part string) string {
	return path.Join(folder, part, part+".go")
}

// wasmPartComponentsRel names a part's components file —
// created on first use, when a component is first asked
// into the part.
func wasmPartComponentsRel(folder, part string) string {
	return path.Join(folder, part, "components.go")
}

// wasmPartFile renders a part's own file: the build tag,
// the package with the words that say what the part is,
// the el helper its builders share, the exported builder
// (which builds the root and mounts the part's own
// components inside it), the marker block those mounts
// land between, and the root builder itself — the kind's
// template for a fresh part, or the body's own builder,
// moved, with its content untouched.
func wasmPartFile(part, rootFn, rootBuilder string) string {
	exported := wasmPartExported(part)
	mountFn := wasmPartMountFunc(part)
	begin, end := wasmPartMarkers(part)
	var words, rootWords string
	if part == "header" {
		words = `// Package header is the wasm app's header: the part
// at the top of the page, in its own package inside
// the app, so the app's main program stays small and
// the part can grow its own components (components.go,
// beside this file) without crowding it. main.go
// mounts it before the app's own components.
`
		rootWords = `// Header builds the app's header — the app bar, with
// the page's title — and mounts the components this
// part has been given inside it, in the order they
// were asked for.
`
	} else {
		words = `// Package footer is the wasm app's footer: the part
// at the foot of the page, in its own package inside
// the app, so the app's main program stays small and
// the part can grow its own components (components.go,
// beside this file) without crowding it. main.go
// mounts it after the app's own components.
`
		rootWords = `// Footer builds the app's footer — the footer bar,
// with its small print and links — and mounts the
// components this part has been given inside it, in
// the order they were asked for.
`
	}
	return `//go:build js && wasm

` + words + `package ` + part + `

import "syscall/js"

// el makes one element: its tag, its class when it has
// one, and its text when it has some.
func el(doc js.Value, tag, class, text string) js.Value {
	e := doc.Call("createElement", tag)
	if class != "" {
		e.Set("className", class)
	}
	if text != "" {
		e.Set("textContent", text)
	}
	return e
}

` + rootWords + `func ` + exported + `(doc js.Value) js.Value {
	root := ` + rootFn + `(doc)
	` + mountFn + `(doc, root)
	return root
}

// ` + mountFn + ` puts the components the chat adds to
// this part on it — ask for one ("add a chip to the ` + part + `
// in this app's folder") and its mount line lands
// between the markers. Leave the markers where they are.
func ` + mountFn + `(doc, root js.Value) {
	` + begin + `
	` + end + `
}

` + rootBuilder
}

// wasmPartComponentsHeader is a part's components file
// head: the build tag, the part's package, and the words
// that say what the file is — written once, when the
// first component lands in the part. The el helper is
// NOT repeated here: it lives in the part's own file,
// beside this one, in the same package.
func wasmPartComponentsHeader(part string) string {
	return `//go:build js && wasm

package ` + part + `

import "syscall/js"

// The components inside this app's ` + part + `, one builder
// each — the Material set the chat adds by sentence
// ("add a chip to the ` + part + ` in this app's folder"), in
// Go instead of HTML, with the same classes and the
// same starter words as their twins in a plain page.
// ` + wasmPartFileName(part) + ` mounts them inside the part, in the
// order they were asked for. Each root element carries
// an id — the name it was given, or its place among its
// kind — because the app's events (the app's events.go)
// find their components by id. The el helper the
// builders use lives in ` + wasmPartFileName(part) + `, beside this
// file.

`
}

// wasmPartFileName is a part's own file name, for prose.
func wasmPartFileName(part string) string {
	return part + ".go"
}

// wasmPartRootRe reads a part file's root builder call:
// the "root := <builder>(doc)" line inside the exported
// builder — how the part's root component is recognised.
var wasmPartRootRe = regexp.MustCompile(`root := (\w+)\(doc\)`)

// wasmPartMountRe reads one mount line inside a part's
// marker block — the part twin of wasmMountRe, on the
// part mount function's root parameter.
var wasmPartMountRe = regexp.MustCompile(`root\.Call\("appendChild", (\w+)\(doc\)\)`)

// wasmPartInstances reads one part's components, in the
// order they appear in the part: its root first (the
// builder the exported function calls), then the mounts
// between the part's markers, their builders read from
// the part's components.go. Ids and names read exactly as
// the body's do (wasmBuilderStamp / wasmBuilderKind), so
// one reading serves the whole app.
func wasmPartInstances(partContent, partCompContent, part string) []WasmComponentInstance {
	if partContent == "" {
		return nil
	}
	var out []WasmComponentInstance
	exp := wasmPartExported(part)
	if i := strings.Index(partContent, "func "+exp+"("); i >= 0 {
		body := partContent[i:]
		if j := strings.Index(body, "\n}\n"); j >= 0 {
			body = body[:j]
		}
		if m := wasmPartRootRe.FindStringSubmatch(body); m != nil {
			fn := m[1]
			if kind, prefix := wasmBuilderKind(fn); kind != "" {
				inst := WasmComponentInstance{Kind: kind, Builder: fn, Part: part, Root: true}
				if id, stamped := wasmBuilderStamp(partContent, fn); stamped {
					inst.ID = id
					if prefix != "" {
						inst.Name = id
					}
				} else {
					inst.ID = kind + "-1"
					inst.Name = prefix
				}
				out = append(out, inst)
			}
		}
	}
	beginMark, endMark := wasmPartMarkers(part)
	begin := strings.Index(partContent, beginMark)
	end := strings.Index(partContent, endMark)
	if begin < 0 || end < 0 || end < begin {
		return out
	}
	counts := map[string]int{}
	for _, inst := range out {
		counts[inst.Kind]++
	}
	for _, m := range wasmPartMountRe.FindAllStringSubmatch(partContent[begin:end], -1) {
		fn := m[1]
		// A taught builder is read from its marker first,
		// as in the body's reading (wasmComponentInstances).
		if inst, ok := taughtInstanceFor(fn, partCompContent, counts); ok {
			inst.Part = part
			out = append(out, inst)
			continue
		}
		kind, prefix := wasmBuilderKind(fn)
		if kind == "" {
			continue
		}
		counts[kind]++
		inst := WasmComponentInstance{Kind: kind, Builder: fn, Part: part}
		id, stamped := wasmBuilderStamp(partCompContent, fn)
		if stamped {
			inst.ID = id
		} else {
			inst.ID = kind + "-" + strconv.Itoa(counts[kind])
		}
		if prefix != "" {
			inst.Name = prefix
			if stamped {
				inst.Name = id
			}
		}
		out = append(out, inst)
	}
	return out
}

// wasmAppAllInstances reads every component mounted in
// the app in folder — the header part's, the body's, the
// footer part's, in the order they appear on the page —
// from files the caller may already hold (the body's) and
// the part files read here. This is the one aggregation
// the component, event, and part planners share: body +
// parts are one app.
func (p *Project) wasmAppAllInstances(folder, mainContent, compContent string) []WasmComponentInstance {
	var out []WasmComponentInstance
	read := func(rel string) string {
		b, _ := os.ReadFile(filepath.Join(p.Root, filepath.FromSlash(rel)))
		return string(b)
	}
	for _, part := range wasmPartNames {
		partContent := read(wasmPartFileRel(folder, part))
		partComp := read(wasmPartComponentsRel(folder, part))
		insts := wasmPartInstances(partContent, partComp, part)
		if part == "header" {
			out = append(out, insts...)
			out = append(out, wasmComponentInstances(mainContent, compContent)...)
		} else {
			out = append(out, insts...)
		}
	}
	return out
}

// wasmBuilderBlock locates one builder's block in a
// components file's content: its doc comment (the
// contiguous // lines directly above the func) through
// the closing brace at column zero — the span a move
// lifts out whole, so the moved builder keeps every word
// it had. Byte offsets, end exclusive.
func wasmBuilderBlock(content, fn string) (int, int, bool) {
	i := strings.Index(content, "func "+fn+"(")
	if i < 0 {
		return 0, 0, false
	}
	start := i
	// Walk back over the comment lines directly above.
	for start > 0 {
		prevEnd := start - 1 // the newline ending the line above
		if prevEnd < 0 || content[prevEnd] != '\n' {
			break
		}
		lineStart := strings.LastIndex(content[:prevEnd], "\n") + 1
		if !strings.HasPrefix(content[lineStart:prevEnd], "//") {
			break
		}
		start = lineStart
	}
	j := strings.Index(content[i:], "\n}\n")
	if j < 0 {
		// A builder at the very end of the file, its
		// closing brace the last bytes.
		if strings.HasSuffix(content[i:], "\n}") {
			return start, len(content), true
		}
		return 0, 0, false
	}
	return start, i + j + len("\n}\n"), true
}

// extractWasmBuilder lifts one builder's block out of a
// components file: the block itself (ending with its
// closing brace and newline) and the file without it —
// the separators normalised so what remains is exactly a
// file the piece could have written. ok=false when the
// builder isn't there in a shape that lifts safely.
func extractWasmBuilder(content, fn string) (block, rest string, ok bool) {
	start, end, found := wasmBuilderBlock(content, fn)
	if !found {
		return "", "", false
	}
	block = content[start:end]
	rest = content[:start] + content[end:]
	// The builders are joined by one blank line; the
	// removal can leave two, or a blank line at the head
	// of the builders — normalise both.
	rest = strings.ReplaceAll(rest, "\n\n\n", "\n\n")
	if i := strings.Index(rest, "func "); i >= 0 {
		head := rest[:i]
		if strings.HasSuffix(head, "\n\n\n") {
			rest = head[:len(head)-1] + rest[i:]
		}
	}
	rest = strings.TrimRight(rest, "\n") + "\n"
	if !strings.HasSuffix(block, "\n") {
		block += "\n"
	}
	return block, rest, true
}

// wasmModulePath reads an app module's path out of its
// go.mod content — the "module <path>" line the import
// of a part package is built from. "" when the line
// isn't there to read.
func wasmModulePath(goMod string) string {
	for _, ln := range strings.Split(goMod, "\n") {
		ln = strings.TrimSpace(ln)
		if strings.HasPrefix(ln, "module ") {
			return strings.TrimSpace(strings.TrimPrefix(ln, "module "))
		}
	}
	return ""
}

// planWasmPart assembles the header and/or footer parts
// for the wasm app in WasmFolder: one new package per
// part asked for (its file, with the part's root builder
// — fresh from the kind's template, or MOVED from the
// body when the body already mounts that component), and
// main.go edited to import the part packages and mount
// them around the body components block — the header
// before it, the footer after it, both ahead of any
// registerEvents call so events on part components find
// them mounted. A part that's already there is the
// family's already-there answer (a note when the other
// part still lands; the refusal when neither does). The
// root kinds' styles join the host page whole-chunk, as
// they do when the components land in the body.
func (p *Project) planWasmPart(req EditRequest) (*EditPlan, string) {
	folder := path.Clean(filepath.ToSlash(strings.TrimSpace(req.WasmFolder)))
	if folder == "" || folder == "." {
		return nil, "a wasm app's parts live in the app's folder, and the app can't be the project root — its go.mod makes the folder a module of its own. Say \"add a header and footer to the wasm app in <folder>\" with the app's folder."
	}
	goModRel, mainRel, _, indexRel, _ := wasmAppFiles(folder)
	compRel := wasmComponentsRel(folder)
	mainBefore, err := os.ReadFile(filepath.Join(p.Root, filepath.FromSlash(mainRel)))
	if err != nil {
		return nil, fmt.Sprintf("there's no wasm app in %s yet — say \"add a wasm app in %s\" first, then it can have a header and a footer.", folder, folder)
	}
	main := string(mainBefore)
	if !wasmAppMainContent(main) {
		return nil, fmt.Sprintf("%s isn't a wasm app's main file — it doesn't carry the js && wasm build tag a wasm app's main wears — so there's no app here for a part to join. Say \"add a wasm app in %s\" to make one.", p.LinkPath(mainRel), folder)
	}
	if !strings.Contains(main, "mountComponents(doc, app)") {
		return nil, fmt.Sprintf("%s doesn't call mountComponents the way the wasm piece writes it — the parts mount around that call, and without it they have nowhere honest to land. Nothing was written.", p.LinkPath(mainRel))
	}
	goModBefore, err := os.ReadFile(filepath.Join(p.Root, filepath.FromSlash(goModRel)))
	if err != nil {
		return nil, fmt.Sprintf("the wasm app in %s has no %s — its own module file — so the part packages have no module path to be imported by. Nothing was written.", folder, p.LinkPath(goModRel))
	}
	module := wasmModulePath(string(goModBefore))
	if module == "" {
		return nil, fmt.Sprintf("%s doesn't carry the module line the wasm piece writes — the part packages are imported by the app's module path, and I can't read it. Nothing was written.", p.LinkPath(goModRel))
	}
	indexBefore, err := os.ReadFile(filepath.Join(p.Root, filepath.FromSlash(indexRel)))
	if err != nil {
		return nil, fmt.Sprintf("%s isn't there — the host page whose style block the parts' styles join is missing, so this isn't the app the wasm piece makes. Nothing was written.", p.LinkPath(indexRel))
	}
	index := string(indexBefore)
	compBefore, _ := os.ReadFile(filepath.Join(p.Root, filepath.FromSlash(compRel)))
	compContent := string(compBefore)
	bodyInstances := wasmComponentInstances(main, compContent)
	allInstances := p.wasmAppAllInstances(folder, main, compContent)

	var parts []string
	if req.WasmPartHeader {
		parts = append(parts, "header")
	}
	if req.WasmPartFooter {
		parts = append(parts, "footer")
	}
	if len(parts) == 0 {
		return nil, "that edit names no part — a header, a footer, or both."
	}

	// What each part's landing needs, settled before any
	// content is built: the part file's state, and the
	// body's component of the part's kind, if one is
	// mounted there to move.
	type partWork struct {
		part      string
		rel       string
		exists    bool // the part file is there and IS this part
		moveFn    string
		moveBlock string
		rootFn    string
		rootB     string
		rootKind  string
	}
	var works []*partWork
	var notes []string
	landed := 0
	newComp := compContent
	newMain := main
	for _, part := range parts {
		w := &partWork{part: part, rel: wasmPartFileRel(folder, part), rootKind: wasmPartRootKind(part)}
		works = append(works, w)
		partAbs := filepath.Join(p.Root, filepath.FromSlash(w.rel))
		if b, rerr := os.ReadFile(partAbs); rerr == nil {
			content := string(b)
			if strings.Contains(content, "package "+part) && strings.Contains(content, "func "+wasmPartExported(part)+"(") {
				w.exists = true
				continue
			}
			return nil, fmt.Sprintf("%s is already there and isn't a %s part the wasm piece recognises — I never overwrite what's yours, so nothing was written.", p.LinkPath(w.rel), part)
		}
		if why, blocked := p.fileBlocked(w.rel); blocked {
			return nil, fmt.Sprintf("I won't write %s — %s.", p.LinkPath(w.rel), why)
		}
		// The move: the body mounts a component of the
		// part's kind — the first one, in mount order.
		for _, in := range bodyInstances {
			if in.Kind == w.rootKind {
				w.moveFn = in.Builder
				break
			}
		}
		if w.moveFn != "" {
			block, rest, ok := extractWasmBuilder(newComp, w.moveFn)
			if !ok {
				return nil, fmt.Sprintf("the wasm app in %s mounts a %s in its body, but its builder isn't in %s in a shape I can lift out whole — moving it could lose it or leave it rendering twice, so I'm not adding the %s this way. Nothing was written.", folder, MaterialComponentWords(w.rootKind), p.LinkPath(compRel), part)
			}
			w.moveBlock = block
			newComp = rest
			// The moved component's body mount line goes
			// away — the part mounts it now.
			mount := "\tapp.Call(\"appendChild\", " + w.moveFn + "(doc))\n"
			if !strings.Contains(newMain, mount) {
				return nil, fmt.Sprintf("the wasm app in %s is said to mount a %s in its body, but the mount line isn't in %s in the shape the wasm piece writes — moving it could leave it rendering twice, so I'm not adding the %s this way. Nothing was written.", folder, MaterialComponentWords(w.rootKind), p.LinkPath(mainRel), part)
			}
			newMain = strings.Replace(newMain, mount, "", 1)
			w.rootFn = w.moveFn
			w.rootB = block
		} else {
			w.rootFn = wasmBuilderName(w.rootKind)
			elemID := w.rootKind + "-" + strconv.Itoa(wasmKindCount(allInstances, w.rootKind)+1)
			builder, ok := wasmComponentBuilder(w.rootKind, w.rootFn, elemID)
			if !ok {
				return nil, "that edit needs one of the material components: " + componentFormsText() + "."
			}
			w.rootB = builder
		}
		landed++
	}
	if landed == 0 {
		// Every part asked for is already there.
		if len(parts) == 2 {
			return nil, fmt.Sprintf("the wasm app in %s already has a header and a footer — nothing to add.", folder)
		}
		return nil, fmt.Sprintf("the wasm app in %s already has a %s — nothing to add.", folder, parts[0])
	}
	for _, w := range works {
		if w.exists {
			notes = append(notes, fmt.Sprintf("The %s is already in the app — %s stays exactly as it is.", w.part, p.LinkPath(w.rel)))
		}
	}

	// main.go: the imports and the mounts. The imports go
	// into the import block in sorted order (the shape the
	// skeleton writes — plain quoted paths, one per line);
	// anything else refuses rather than rewriting by guess.
	var newImports []string
	for _, w := range works {
		if !w.exists {
			newImports = append(newImports, module+"/"+w.part)
		}
	}
	mountHeader, mountFooter := false, false
	for _, w := range works {
		if w.exists {
			continue
		}
		if w.part == "header" {
			mountHeader = true
		} else {
			mountFooter = true
		}
	}
	editedMain, refusal := wasmMainWithPartMounts(newMain, newImports, mountHeader, mountFooter)
	if refusal != "" {
		return nil, refusal
	}
	newMain = editedMain

	// The root kinds' styles join the host page, whole
	// chunk or not at all — the component piece's
	// discipline, so a part that lands styled in the body
	// stays styled in its part.
	needCSS := ""
	if !strings.Contains(index, MaterialStyleMarker) {
		needCSS = MaterialCSSVariables()
	}
	for _, w := range works {
		if w.exists {
			continue
		}
		comp, _ := MaterialComponentByID(w.rootKind)
		if !strings.Contains(index, comp.CSS) && !strings.Contains(needCSS, comp.CSS) {
			needCSS += comp.CSS
		}
	}
	newIndex := index
	if needCSS != "" {
		styleEnd := strings.Index(index, "</style>")
		if styleEnd < 0 {
			return nil, fmt.Sprintf("%s has no style block for the parts' styles to join — it isn't the app's host page I recognise.", p.LinkPath(indexRel))
		}
		newIndex = index[:styleEnd] + needCSS + index[styleEnd:]
	}

	for _, rel := range []string{mainRel, compRel, indexRel} {
		if why, blocked := p.fileBlocked(rel); blocked {
			return nil, fmt.Sprintf("I won't write %s — %s.", p.LinkPath(rel), why)
		}
	}
	files := []EditFile{}
	want := map[string]string{}
	for _, w := range works {
		if w.exists {
			continue
		}
		content := wasmPartFile(w.part, w.rootFn, w.rootB)
		files = append(files, finishFile(w.rel, nil, []byte(content)))
		want[w.rel] = content
		if w.moveFn != "" {
			notes = append(notes, fmt.Sprintf("The %s your app already had moves into the %s — its builder, its words, and its id are unchanged; it renders once, from the part now.", MaterialComponentWords(w.rootKind), w.part))
		}
	}
	files = append(files, finishFile(mainRel, mainBefore, []byte(newMain)))
	want[mainRel] = newMain
	if newComp != compContent {
		files = append(files, finishFile(compRel, compBefore, []byte(newComp)))
		want[compRel] = newComp
	}
	if needCSS != "" {
		files = append(files, finishFile(indexRel, indexBefore, []byte(newIndex)))
		want[indexRel] = newIndex
	}
	landedParts := []string{}
	for _, w := range works {
		if !w.exists {
			landedParts = append(landedParts, w.part)
		}
	}
	summary := fmt.Sprintf("Give the wasm app in %s %s: %s. %s mounts the header before the app's components and the footer after them.",
		folder, wasmPartListWords(landedParts), wasmPartFilesWords(p, folder, landedParts), p.LinkPath(mainRel))
	verifyDesc := fmt.Sprintf("the wasm app in %s now has %s", folder, wasmPartListWords(landedParts))
	return &EditPlan{
		Summary:    summary,
		VerifyDesc: verifyDesc,
		Site:       p.LinkPath(mainRel),
		Files:      files,
		Notes:      notes,
		verify: func(after *Project) bool {
			for rel, content := range want {
				b, err := os.ReadFile(filepath.Join(after.Root, filepath.FromSlash(rel)))
				if err != nil || string(b) != content {
					return false
				}
			}
			for _, w := range works {
				if w.exists {
					continue
				}
				dir := wasmPartDirRel(folder, w.part)
				found := false
				for _, pkg := range after.Packages {
					if pkg.Dir != dir {
						continue
					}
					for _, fn := range pkg.Funcs {
						if fn.Receiver == "" && fn.Name == wasmPartExported(w.part) {
							found = true
						}
					}
				}
				if !found {
					return false
				}
			}
			return true
		},
	}, ""
}

// wasmPartListWords renders the parts a sentence names:
// "a header", "a footer", "a header and a footer".
func wasmPartListWords(parts []string) string {
	if len(parts) == 2 {
		return "a header and a footer"
	}
	return "a " + parts[0]
}

// wasmPartFilesWords renders where the parts land, for
// the plan summary: "header/header.go (its own package,
// package header)" per part.
func wasmPartFilesWords(p *Project, folder string, parts []string) string {
	var out []string
	for _, part := range parts {
		out = append(out, fmt.Sprintf("%s (the %s part, its own package)", p.LinkPath(wasmPartFileRel(folder, part)), part))
	}
	return strings.Join(out, " and ")
}

// planComponentWasmPart is the component piece's part
// mode: the target is one part of a wasm app ("add a
// navigation bar to the header in jim"), so the component
// lands in the part — its builder joins the part's
// components.go (created with the part's header on first
// use), one mount line lands between the part's markers
// in ask order, and the styles it needs join the host
// page exactly once, whole chunk or not at all — the
// body mode's discipline, scoped to the part. The
// already-there rules are per part (a card in the body is
// not a card in the header), while a component's NAME
// still belongs to one component in the whole app, and a
// new unnamed instance's id counts its kind across the
// whole app, exactly as in the body mode.
func (p *Project) planComponentWasmPart(req EditRequest, comp MaterialComponent) (*EditPlan, string) {
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
		return nil, fmt.Sprintf("there's no wasm app in %s yet — say \"add a wasm app in %s\" first, then the %s can go into its %s.", folder, folder, comp.Words, part)
	}
	main := string(mainBefore)
	if !wasmAppMainContent(main) {
		return nil, fmt.Sprintf("%s isn't a wasm app's main file — it doesn't carry the js && wasm build tag a wasm app's main wears — so the %s has no app to join. Say \"add a wasm app in %s\" to make one, or add the %s to a page instead.", p.LinkPath(mainRel), comp.Words, folder, comp.Words)
	}
	indexBefore, err := os.ReadFile(filepath.Join(p.Root, filepath.FromSlash(indexRel)))
	if err != nil {
		return nil, fmt.Sprintf("%s isn't there — the host page whose style block the %s's styles join is missing, so this isn't the app the wasm piece makes. Nothing was written.", p.LinkPath(indexRel), comp.Words)
	}
	index := string(indexBefore)
	partBefore, err := os.ReadFile(filepath.Join(p.Root, filepath.FromSlash(partRel)))
	if err != nil {
		return nil, fmt.Sprintf("the wasm app in %s has no %s yet — say \"add a %s to the wasm app in %s\" first, then the %s can go into it.", folder, part, part, folder, comp.Words)
	}
	partContent := string(partBefore)
	if !strings.Contains(partContent, "package "+part) || !strings.Contains(partContent, "func "+wasmPartExported(part)+"(") {
		return nil, fmt.Sprintf("%s isn't the app's %s part — it doesn't carry the %s package and its %s builder the parts piece writes — so the %s has no part to join. Nothing was written.", p.LinkPath(partRel), part, part, wasmPartExported(part), comp.Words)
	}
	beginMark, endMark := wasmPartMarkers(part)
	begin := strings.Index(partContent, beginMark)
	end := strings.Index(partContent, endMark)
	if begin < 0 || end < 0 || end < begin {
		return nil, fmt.Sprintf("%s doesn't have the %s components block — the two %s marker comments in its mount function — so the %s has nowhere honest to land. Nothing was written.", p.LinkPath(partRel), part, part, comp.Words)
	}
	for _, rel := range []string{partRel, partCompRel, indexRel} {
		if why, blocked := p.fileBlocked(rel); blocked {
			return nil, fmt.Sprintf("I won't write %s — %s.", p.LinkPath(rel), why)
		}
	}
	partCompBefore, _ := os.ReadFile(filepath.Join(p.Root, filepath.FromSlash(partCompRel)))
	partCompContent := string(partCompBefore)
	compBefore, _ := os.ReadFile(filepath.Join(p.Root, filepath.FromSlash(compRel)))
	partInstances := wasmPartInstances(partContent, partCompContent, part)
	allInstances := p.wasmAppAllInstances(folder, main, string(compBefore))
	// The instance this ask adds — the body mode's rules,
	// scoped: names are unique across the app, unnamed
	// instances are per part.
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
			if in.Kind == comp.ID {
				return nil, fmt.Sprintf("the wasm app in %s already has a %s called %s — nothing to add.", folder, comp.Words, name)
			}
			return nil, fmt.Sprintf("the wasm app in %s already has a component called %s — its %s — and a name belongs to one component. Pick another name, or add the %s without one.", folder, name, MaterialComponentWords(in.Kind), comp.Words)
		}
		fn = wasmNamedBuilderName(name, comp.ID)
		if clashKind, _ := wasmBuilderKind(fn); clashKind != "" && clashKind != comp.ID {
			return nil, fmt.Sprintf("a %s called %s would be built by %s — the %s's own builder name — so the two could never be told apart in the app's code. Pick another name.", comp.Words, name, fn, MaterialComponentWords(clashKind))
		}
		if strings.Contains(partCompContent, "func "+fn+"(") || strings.Contains(partContent, "func "+fn+"(") {
			return nil, fmt.Sprintf("the %s part of the wasm app in %s already declares %s — the builder name a %s called %s would take — so nothing was written. Rename that function, or pick another name.", part, folder, fn, comp.Words, name)
		}
		elemID = name
	} else {
		fn = wasmBuilderName(comp.ID)
		taken := strings.Contains(partCompContent, "func "+fn+"(")
		for _, in := range partInstances {
			if in.Builder == fn {
				taken = true
			}
		}
		if taken {
			return nil, fmt.Sprintf("the %s of the wasm app in %s already has a %s in it — nothing to add.", part, folder, comp.Words)
		}
		elemID = comp.ID + "-" + strconv.Itoa(wasmKindCount(allInstances, comp.ID)+1)
	}
	builder, ok := wasmComponentBuilder(comp.ID, fn, elemID)
	if !ok {
		return nil, "that edit needs one of the material components: " + componentFormsText() + "."
	}
	needCSS := ""
	if !strings.Contains(index, MaterialStyleMarker) {
		needCSS = MaterialCSSVariables()
	}
	if !strings.Contains(index, comp.CSS) {
		needCSS += comp.CSS
	}
	newIndex := index
	if needCSS != "" {
		styleEnd := strings.Index(index, "</style>")
		if styleEnd < 0 {
			return nil, fmt.Sprintf("%s has no style block for the %s's styles to join — it isn't the app's host page I recognise.", p.LinkPath(indexRel), comp.Words)
		}
		newIndex = index[:styleEnd] + needCSS + index[styleEnd:]
	}
	// The mount line lands just before the part's end
	// marker's line, below the mounts already there — the
	// part's components accumulate in ask order.
	endLine := strings.LastIndex(partContent[:end], "\n") + 1
	newPart := partContent[:endLine] + "\troot.Call(\"appendChild\", " + fn + "(doc))\n" + partContent[endLine:]
	newPartComp := ""
	if partCompContent == "" {
		newPartComp = wasmPartComponentsHeader(part) + builder
	} else {
		newPartComp = strings.TrimRight(partCompContent, "\n") + "\n\n" + builder
	}
	files := []EditFile{finishFile(partCompRel, partCompBefore, []byte(newPartComp)), finishFile(partRel, partBefore, []byte(newPart))}
	if needCSS != "" {
		files = append(files, finishFile(indexRel, indexBefore, []byte(newIndex)))
	}
	what := comp.Words
	if name != "" {
		what += " called " + name
	}
	summary := fmt.Sprintf("Put a %s in the %s of the wasm app in %s: %s joins %s and is mounted inside the %s in %s", what, part, folder, fn, p.LinkPath(partCompRel), part, p.LinkPath(partRel))
	verifyDesc := fmt.Sprintf("the %s of the wasm app in %s now has a %s mounted inside it", part, folder, what)
	if needCSS != "" {
		summary += ", with the Material styles it needs joining the host page's style block in " + p.LinkPath(indexRel)
		verifyDesc += ", and the styles it needs are in the host page"
	} else {
		summary += " — its Material styles are already in the host page"
	}
	return &EditPlan{
		Summary:    summary,
		VerifyDesc: verifyDesc,
		Site:       p.LinkPath(partRel),
		Files:      files,
		Notes: []string{
			"The " + comp.Words + " renders inside the app's " + part + " when the app is built and the page loads — it is a Go builder in the " + part + "'s own package, not HTML in the page. Its element carries the id \"" + elemID + "\", the name an event finds it by: \"when the button is clicked, show the card\" is how one is added.",
		},
		verify: func(after *Project) bool {
			b, err := os.ReadFile(filepath.Join(after.Root, filepath.FromSlash(partCompRel)))
			if err != nil || string(b) != newPartComp {
				return false
			}
			m, err := os.ReadFile(filepath.Join(after.Root, filepath.FromSlash(partRel)))
			if err != nil || string(m) != newPart {
				return false
			}
			if needCSS != "" {
				h, err := os.ReadFile(filepath.Join(after.Root, filepath.FromSlash(indexRel)))
				if err != nil || string(h) != newIndex {
					return false
				}
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

// wasmMainWithPartMounts edits an app's main.go for the
// parts being landed: their packages join the import
// block in sorted order (the shape the skeleton writes —
// plain quoted paths, one per line; anything else refuses
// rather than rewriting by guess), the header's mount
// line lands immediately before the mountComponents
// call — or before the home page's mount line, when the
// main carries one, so the page order stays header, home
// page, components — and the footer's immediately after
// the mountComponents call, which is also before any
// registerEvents call the events piece put there, so
// events keep finding every component mounted. A main.go
// already mounting a part takes no second mount line.
func wasmMainWithPartMounts(main string, imports []string, mountHeader, mountFooter bool) (string, string) {
	out := main
	if len(imports) > 0 {
		begin := strings.Index(out, "import (\n")
		if begin < 0 {
			return "", "the app's main.go doesn't have the import block the wasm piece writes — the part packages are imported there, and I won't rewrite an import block I don't recognise. Nothing was written."
		}
		rest := out[begin+len("import (\n"):]
		end := strings.Index(rest, "\n)\n")
		if end < 0 {
			return "", "the app's main.go doesn't have the import block the wasm piece writes — the part packages are imported there, and I won't rewrite an import block I don't recognise. Nothing was written."
		}
		var paths []string
		seen := map[string]bool{}
		for _, ln := range strings.Split(rest[:end], "\n") {
			ln = strings.TrimSpace(ln)
			if ln == "" {
				continue
			}
			if len(ln) < 2 || ln[0] != '"' || ln[len(ln)-1] != '"' {
				return "", "the app's main.go import block carries a line that isn't a plain quoted path (" + ln + ") — the part packages join that block in sorted order, and I won't rewrite an import block I don't recognise. Nothing was written."
			}
			pth, err := strconv.Unquote(ln)
			if err != nil {
				return "", "the app's main.go import block carries a line that isn't a plain quoted path (" + ln + ") — the part packages join that block in sorted order, and I won't rewrite an import block I don't recognise. Nothing was written."
			}
			if !seen[pth] {
				seen[pth] = true
				paths = append(paths, pth)
			}
		}
		for _, imp := range imports {
			if !seen[imp] {
				seen[imp] = true
				paths = append(paths, imp)
			}
		}
		sort.Strings(paths)
		var block strings.Builder
		block.WriteString("import (\n")
		for _, pth := range paths {
			block.WriteString("\t" + strconv.Quote(pth) + "\n")
		}
		block.WriteString(")\n")
		out = out[:begin] + block.String() + rest[end+len("\n)\n"):]
	}
	mountCall := "mountComponents(doc, app)"
	i := strings.Index(out, mountCall)
	if i < 0 {
		return "", "the app's main.go doesn't call mountComponents the way the wasm piece writes it — the parts mount around that call, and without it they have nowhere honest to land. Nothing was written."
	}
	lineStart := strings.LastIndex(out[:i], "\n") + 1
	lineEnd := strings.Index(out[i:], "\n") + i + 1
	if mountHeader && !strings.Contains(out, "header.Header(doc)") {
		// The header mounts above the home page when the
		// main carries one: the home mount chunk (its
		// comment and its line) sits between the #app
		// lookup and the components, and the header
		// belongs above the whole chunk.
		insertAt := lineStart
		if h := strings.Index(out, "Home(doc, app)"); h >= 0 {
			anchor := h
			if c := strings.Index(out, "// The home page (home.go):"); c >= 0 && c < h {
				anchor = c
			}
			if as := strings.LastIndex(out[:anchor], "\n") + 1; as < lineStart {
				insertAt = as
			}
		}
		out = out[:insertAt] + "\tapp.Call(\"appendChild\", header.Header(doc))\n" + out[insertAt:]
		lineEnd += len("\tapp.Call(\"appendChild\", header.Header(doc))\n")
	}
	if mountFooter && !strings.Contains(out, "footer.Footer(doc)") {
		out = out[:lineEnd] + "\tapp.Call(\"appendChild\", footer.Footer(doc))\n" + out[lineEnd:]
	}
	return out, ""
}
