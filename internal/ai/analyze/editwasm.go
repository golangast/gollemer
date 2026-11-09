package analyze

// The wasm piece (chat/wasm.go): a whole browser app in one
// folder — Go compiled to WebAssembly, driving its page
// through syscall/js — plus the component piece's wasm
// mode, where a Material component lands in the app as a
// Go DOM builder instead of an HTML snippet.
//
// The app is its own module: the go.mod the skeleton writes
// pins Go 1.27, and that pin is load-bearing twice. The
// portable simd package the app's vector unit uses exists
// for wasm targets only in Go 1.27 (on 1.26 the simd
// experiment is AMD64-only), and the nested module boundary
// keeps the app's build-tagged files out of the host
// project's own build and Go version entirely.
//
// The garbage collector's GOGC knob is set low in the
// generated main.go, in code (runtime/debug.SetGCPercent):
// a browser page has no environment to pass flags through,
// so the app sets the flag itself, first thing, with the
// comment naming what it is.
//
// Components render and mount only — no event handlers, no
// js.FuncOf anywhere. A switch in a wasm app is a switch
// that sits there; nothing in this piece promises a click.

import (
	"fmt"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
)

// WasmAppGoVersion is the Go version the app's own module
// pins — the pin that makes the portable simd package
// available to the wasm build.
const WasmAppGoVersion = "1.27.0"

// WasmBuildCommand is the exact command that builds the
// app, stated in the host page's comment, the plan, and
// the landing confirmation — one string, so they cannot
// drift apart.
const WasmBuildCommand = "GOEXPERIMENT=simd GOOS=js GOARCH=wasm go build -o main.wasm ."

// wasmAppFiles names the app's files in the folder: its
// module, its main, its vector unit, its host page, and the
// runtime glue copied from the Go install.
func wasmAppFiles(folder string) (goMod, main, vector, index, execJS string) {
	return path.Join(folder, "go.mod"),
		path.Join(folder, "main.go"),
		path.Join(folder, "vector.go"),
		path.Join(folder, "index.html"),
		path.Join(folder, "wasm_exec.js")
}

// wasmAppGoMod is the app's own module: the folder's base
// name as the module path, and the Go 1.27 pin.
func wasmAppGoMod(folder string) string {
	return "module " + path.Base(folder) + "\n\ngo " + WasmAppGoVersion + "\n"
}

// wasmAppMain is the app itself: it sets the collector's
// knob, builds the welcome heading and the vector demo's
// result into the page's #app element, mounts whatever
// components the chat has added (between the markers the
// component piece writes into), and then blocks forever —
// a wasm program whose main returns is finished, and the
// browser takes its page down with it.
func wasmAppMain() string {
	return `//go:build js && wasm

// This app runs in the browser: Go compiled to WebAssembly
// (a .wasm file), building its page through syscall/js. It
// lives in its own module — the go.mod beside it pins Go
// 1.27 — so the project around it keeps its own Go version,
// and the js && wasm build tag above keeps these files out
// of every ordinary build. The one build that includes
// them is the browser build its host page names:
//
//	` + WasmBuildCommand + `

package main

import (
	"fmt"
	"runtime/debug"
	"syscall/js"
)

func main() {
	// The garbage collector's GOGC setting, turned down in
	// code: a browser page has no environment to pass flags
	// through, so the app sets the flag itself, first thing.
	// 50 collects at half the usual heap growth — a smaller
	// footprint in the tab, at a little CPU. Raise the
	// number here if the app would rather trade memory for
	// speed.
	debug.SetGCPercent(50)

	doc := js.Global().Get("document")
	app := doc.Call("getElementById", "app")

	heading := doc.Call("createElement", "h1")
	heading.Set("className", "headline")
	heading.Set("textContent", "TODO: your app's name")
	app.Call("appendChild", heading)

	// The vector demo (vector.go): two rows of numbers the
	// simd unit added, printed on the page as the proof it
	// ran.
	sums := AddVectors(
		[]float32{1, 2, 3, 4, 5, 6, 7, 8},
		[]float32{10, 20, 30, 40, 50, 60, 70, 80},
	)
	demo := doc.Call("createElement", "p")
	demo.Set("textContent", fmt.Sprintf("The vector demo added the two rows of numbers: %v", sums))
	app.Call("appendChild", demo)

	mountComponents(doc, app)

	// A wasm program whose main returns is finished, and
	// the browser takes the page down with it — so block
	// here forever and let the page live.
	select {}
}

// mountComponents puts the components the chat adds to
// this app on the page, in the order they were asked for:
// ask for one ("add a card to this app's folder") and its
// mount line lands between the markers. Leave the markers
// where they are.
func mountComponents(doc, app js.Value) {
	// gollemer:components-begin
	// gollemer:components-end
}
`
}

// wasmAppVector is the app's simd unit: whole rows of
// numbers added one vector at a time on the portable simd
// package — the package Go 1.27 opened to every target,
// wasm included — with the tail (fewer numbers than one
// vector holds) taken by the Part loads.
func wasmAppVector() string {
	return `//go:build js && wasm

package main

import "simd"

// AddVectors adds two rows of numbers, one vector at a
// time: the portable simd package (Go 1.27's — the reason
// this app's go.mod pins 1.27) loads a whole vector of
// numbers, adds the matching vector from the other row,
// and stores the result, so the machine does several
// additions in a single step. It builds with
// GOEXPERIMENT=simd, the flag in the build command on the
// host page. b is at least as long as a; the result is as
// long as a.
func AddVectors(a, b []float32) []float32 {
	out := make([]float32, len(a))
	// One vector's worth of numbers at a time, while a
	// whole vector fits in what's left. (A zero vector's
	// Len is how many numbers one vector holds here.)
	step := simd.BroadcastFloat32s(0).Len()
	i := 0
	for ; i+step <= len(a); i += step {
		simd.LoadFloat32s(a[i:]).Add(simd.LoadFloat32s(b[i:])).Store(out[i:])
	}
	// The tail: fewer numbers than one vector holds. The
	// Part loads take what's there, and StorePart writes
	// exactly that much back.
	if i < len(a) {
		va, _ := simd.LoadFloat32sPart(a[i:])
		vb, _ := simd.LoadFloat32sPart(b[i:])
		va.Add(vb).StorePart(out[i:])
	}
	return out
}
`
}

// wasmAppIndex is the host page: the Material styles (the
// shared variables and base, the same single source as the
// pages), the #app element the program builds into, and
// the standard loader — wasm_exec.js (the runtime glue
// from the Go install) plus the instantiation of
// main.wasm. The build command is stated in the comment at
// the top, where a beginner meets it first.
func wasmAppIndex() string {
	return `<!DOCTYPE html>
<!--
	This page hosts a Go WebAssembly app. Build the app from
	the Go files in this folder:

	` + WasmBuildCommand + `

	Run it in this folder — the app's go.mod pins Go 1.27,
	and a first build fetches that toolchain by itself.
	Then serve this folder with any static file server and
	open this page: the app's words appear below, built by
	the Go program in the browser.
-->
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TODO: your app's name</title>
<style>
` + MaterialCSSVariables() + MaterialCSSBase() + `</style>
</head>
<body>
<div id="app"></div>
<script src="wasm_exec.js"></script>
<script>
	// The loader: wasm_exec.js (above) is the Go runtime's
	// glue; this asks the browser to fetch the built
	// main.wasm and start it.
	if (!WebAssembly.instantiateStreaming) {
		WebAssembly.instantiateStreaming = async (resp, importObject) => {
			const source = await (await resp).arrayBuffer();
			return await WebAssembly.instantiate(source, importObject);
		};
	}
	const go = new Go();
	WebAssembly.instantiateStreaming(fetch("main.wasm"), go.importObject).then((result) => {
		go.run(result.instance);
	});
</script>
</body>
</html>
`
}

// findWasmExecJS locates this machine's Go 1.27
// wasm_exec.js — the runtime glue the host page loads —
// and returns its bytes and version. The search reads the
// toolchain trees the go command itself keeps in the
// module cache (GOMODCACHE, then the default ~/go/pkg/mod)
// and takes the highest 1.27.x there; the running
// program's own GOROOT counts when it is a 1.27 tree. The
// file is copied, never embedded: a stale copy in the repo
// would rot against the runtime it glues to.
func findWasmExecJS() (content []byte, version string, ok bool) {
	type candidate struct {
		version string
		path    string
	}
	var cands []candidate
	var roots []string
	if mc := os.Getenv("GOMODCACHE"); mc != "" {
		roots = append(roots, mc)
	}
	if home, err := os.UserHomeDir(); err == nil {
		roots = append(roots, filepath.Join(home, "go", "pkg", "mod"))
	}
	for _, root := range roots {
		entries, err := os.ReadDir(filepath.Join(root, "golang.org"))
		if err != nil {
			continue
		}
		for _, e := range entries {
			name := e.Name()
			if !strings.HasPrefix(name, "toolchain@") {
				continue
			}
			// toolchain@v0.0.1-go1.27.2.linux-amd64 — the
			// version is the run of digits and dots after
			// "go", up to the platform suffix.
			i := strings.Index(name, "-go")
			if i < 0 {
				continue
			}
			rest := name[i+len("-go"):]
			j := 0
			for j < len(rest) && (rest[j] >= '0' && rest[j] <= '9' || rest[j] == '.') {
				j++
			}
			ver := strings.TrimSuffix(rest[:j], ".")
			if !strings.HasPrefix(ver, "1.27.") {
				continue
			}
			cands = append(cands, candidate{ver, filepath.Join(root, "golang.org", name, "lib", "wasm", "wasm_exec.js")})
		}
	}
	if rv := runtime.Version(); strings.HasPrefix(rv, "go1.27.") {
		if goroot := runtime.GOROOT(); goroot != "" {
			cands = append(cands, candidate{strings.TrimPrefix(rv, "go"), filepath.Join(goroot, "lib", "wasm", "wasm_exec.js")})
		}
	}
	sort.Slice(cands, func(i, j int) bool { return wasmVersionLess(cands[j].version, cands[i].version) })
	for _, c := range cands {
		if b, err := os.ReadFile(c.path); err == nil && len(b) > 0 {
			return b, c.version, true
		}
	}
	return nil, "", false
}

// wasmVersionLess compares two dotted versions ("1.27.2")
// by their numbers.
func wasmVersionLess(a, b string) bool {
	pa, pb := strings.Split(a, "."), strings.Split(b, ".")
	for i := 0; i < len(pa) && i < len(pb); i++ {
		na, _ := strconv.Atoi(pa[i])
		nb, _ := strconv.Atoi(pb[i])
		if na != nb {
			return na < nb
		}
	}
	return len(pa) < len(pb)
}

// wasmAppMainContent reports whether file content is a
// wasm app's main: the js && wasm build tag and the
// syscall/js import. Content is the recognition, as
// everywhere in this family.
func wasmAppMainContent(content string) bool {
	return strings.Contains(content, "js && wasm") && strings.Contains(content, "\"syscall/js\"")
}

// planWasmApp assembles the skeleton: the app's module,
// main, vector unit, and host page as new files — each
// refused if it already exists, because a file that is
// there is the user's — plus the wasm_exec.js copied from
// this machine's Go 1.27 install when one is found (a
// wasm_exec.js already in the folder is left exactly as it
// is). When no 1.27 install is found, the plan still lands
// and its note names the one manual copy, in the open.
func (p *Project) planWasmApp(req EditRequest) (*EditPlan, string) {
	folder := path.Clean(filepath.ToSlash(strings.TrimSpace(req.WasmFolder)))
	if folder == "" || folder == "." {
		return nil, "a wasm app needs a folder of its own — its go.mod makes the folder a module of its own, apart from the project's build, so the app can't be the project root. Say \"add a wasm app in <folder>\" with a folder of its own."
	}
	if st, err := os.Stat(filepath.Join(p.Root, filepath.FromSlash(folder))); err != nil || !st.IsDir() {
		if err == nil {
			return nil, fmt.Sprintf("%s isn't a folder — a wasm app belongs to a folder, as its own module.", folder)
		}
		return nil, fmt.Sprintf("there's no folder %s in this project for the app to live in — say \"create a folder %s\" first, then ask again.", folder, folder)
	}
	goModRel, mainRel, vectorRel, indexRel, execRel := wasmAppFiles(folder)
	contents := []struct {
		rel  string
		body string
	}{
		{goModRel, wasmAppGoMod(folder)},
		{mainRel, wasmAppMain()},
		{vectorRel, wasmAppVector()},
		{indexRel, wasmAppIndex()},
	}
	files := []EditFile{}
	want := map[string]string{}
	for _, c := range contents {
		if why, blocked := p.fileBlocked(c.rel); blocked {
			return nil, fmt.Sprintf("I won't write %s — %s.", p.LinkPath(c.rel), why)
		}
		if _, err := os.Lstat(filepath.Join(p.Root, filepath.FromSlash(c.rel))); err == nil {
			return nil, fmt.Sprintf("%s is already there — I never overwrite what's yours, so nothing was written.", p.LinkPath(c.rel))
		}
		files = append(files, finishFile(c.rel, nil, []byte(c.body)))
		want[c.rel] = c.body
	}
	notes := []string{
		"The app is its own module: " + p.LinkPath(goModRel) + " pins Go " + WasmAppGoVersion + ", so the project around it keeps its own Go version, and the app's files stay out of the project's build.",
		"The app sets the garbage collector's GOGC knob low in code — runtime/debug.SetGCPercent(50) in " + p.LinkPath(mainRel) + " — because a browser page has no environment to pass the flag through.",
		"Build it in " + folder + ": " + WasmBuildCommand + " — a first build fetches the Go 1.27 toolchain by itself. Then any static file server can serve the folder (\"serve static files\" points this project's server at a folder).",
	}
	execNote := ""
	if _, err := os.Lstat(filepath.Join(p.Root, filepath.FromSlash(execRel))); err == nil {
		execNote = p.LinkPath(execRel) + " is already in the folder, so it stays exactly as it is."
	} else if b, ver, found := findWasmExecJS(); found {
		files = append(files, finishFile(execRel, nil, b))
		want[execRel] = string(b)
		execNote = p.LinkPath(execRel) + " is copied from this machine's Go " + ver + " install — the runtime glue the host page loads."
	} else {
		execNote = "wasm_exec.js is NOT in this plan: no Go 1.27 install was found on this machine to copy it from. Copy lib/wasm/wasm_exec.js from any Go 1.27 install into " + folder + " — the page can't start the app without it."
	}
	notes = append(notes, execNote)
	summary := fmt.Sprintf("Create the wasm app in %s: %s (its own module, pinned to Go %s), %s (the app — it builds the page in the browser through syscall/js), %s (a small simd unit that adds rows of numbers a vector at a time), and %s (the host page, with the Material styles, that loads the built app).",
		folder, p.LinkPath(goModRel), WasmAppGoVersion, p.LinkPath(mainRel), p.LinkPath(vectorRel), p.LinkPath(indexRel))
	return &EditPlan{
		Summary:    summary,
		VerifyDesc: fmt.Sprintf("the wasm app is in %s", folder),
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
			for _, pkg := range after.Packages {
				if pkg.Dir != folder {
					continue
				}
				for _, fn := range pkg.Funcs {
					if fn.Receiver == "" && fn.Name == "main" {
						return true
					}
				}
			}
			return false
		},
	}, ""
}

// wasmComponentsRel names the app's components file.
func wasmComponentsRel(folder string) string {
	return path.Join(folder, "components.go")
}

// wasmBuilderName is a component's builder function name:
// "card" builds cardComponent, "tonal-button" builds
// tonalButtonComponent.
func wasmBuilderName(id string) string {
	parts := strings.Split(id, "-")
	for i, w := range parts {
		if i > 0 && w != "" {
			parts[i] = strings.ToUpper(w[:1]) + w[1:]
		}
	}
	return strings.Join(parts, "") + "Component"
}

// wasmComponentsHeader is the components file's head: the
// build tag, the package, the one helper every builder
// uses, and the words that say what the file is — written
// once, when the first component lands.
const wasmComponentsHeader = `//go:build js && wasm

package main

import "syscall/js"

// The components on this app's page, one builder each —
// the Material set the chat adds by sentence ("add a card
// to this app's folder"), in Go instead of HTML: a wasm
// app builds its page in the browser, so each component is
// a function that makes the same elements, with the same
// classes and the same starter words, as its twin in a
// plain page. main.go mounts them in the order they were
// asked for. They render and sit there — this set has no
// click handlers: nothing here listens for anything.

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

`

// wasmComponentBuilder renders one component's builder:
// the same structure and TODO words as the component's
// HTML snippet (MaterialComponentSnippet), built element
// by element. The registry has no builder for an id it
// doesn't know — the second return value says so.
func wasmComponentBuilder(id string) (string, bool) {
	switch id {
	case "button":
		return `// The filled button.
func buttonComponent(doc js.Value) js.Value {
	b := el(doc, "button", "btn btn-filled", "TODO: the button's words")
	b.Set("type", "button")
	return b
}
`, true
	case "tonal-button":
		return `// The tonal button.
func tonalButtonComponent(doc js.Value) js.Value {
	b := el(doc, "button", "btn btn-tonal", "TODO: the button's words")
	b.Set("type", "button")
	return b
}
`, true
	case "outlined-button":
		return `// The outlined button.
func outlinedButtonComponent(doc js.Value) js.Value {
	b := el(doc, "button", "btn btn-outlined", "TODO: the button's words")
	b.Set("type", "button")
	return b
}
`, true
	case "text-button":
		return `// The text button.
func textButtonComponent(doc js.Value) js.Value {
	b := el(doc, "button", "btn btn-text", "TODO: the button's words")
	b.Set("type", "button")
	return b
}
`, true
	case "card":
		return `// A card: a heading and its words, grouped in a
// raised box.
func cardComponent(doc js.Value) js.Value {
	card := el(doc, "div", "card", "")
	card.Call("appendChild", el(doc, "p", "card-title", "TODO: the card's heading"))
	card.Call("appendChild", el(doc, "p", "", "TODO: the card's words — a card groups one thing: a person, a product, a short note."))
	return card
}
`, true
	case "app-bar":
		return `// The app bar: the bar at the top, speaking the
// page's title.
func appBarComponent(doc js.Value) js.Value {
	title := doc.Get("title").String()
	if title == "" {
		title = "TODO: the page's name"
	}
	bar := el(doc, "header", "app-bar", "")
	bar.Call("appendChild", el(doc, "span", "app-bar-title", title))
	return bar
}
`, true
	case "navigation-bar":
		return `// The navigation bar: the brand, the small-screen
// menu switch, and the links.
func navigationBarComponent(doc js.Value) js.Value {
	nav := el(doc, "nav", "site-nav", "")
	nav.Call("setAttribute", "aria-label", "Main")
	inner := el(doc, "div", "site-nav-inner", "")
	brand := el(doc, "a", "site-nav-brand", "TODO: your site's name")
	brand.Set("href", "#")
	inner.Call("appendChild", brand)
	// The menu checkbox is the small-screen menu switch:
	// tapping Menu checks it, and the navigation bar
	// styles then show the links. No JavaScript is
	// involved.
	toggle := el(doc, "input", "site-nav-toggle", "")
	toggle.Set("id", "site-nav-toggle")
	toggle.Set("type", "checkbox")
	inner.Call("appendChild", toggle)
	menu := el(doc, "label", "site-nav-menu-button", "☰ Menu")
	menu.Set("htmlFor", "site-nav-toggle")
	inner.Call("appendChild", menu)
	links := el(doc, "ul", "site-nav-links", "")
	for _, name := range []string{"Home", "About", "Contact"} {
		item := el(doc, "li", "", "")
		link := el(doc, "a", "", name)
		link.Set("href", "#")
		item.Call("appendChild", link)
		links.Call("appendChild", item)
	}
	inner.Call("appendChild", links)
	nav.Call("appendChild", inner)
	return nav
}
`, true
	case "fab":
		return `// The floating button: the round button fixed to
// the corner.
func fabComponent(doc js.Value) js.Value {
	b := el(doc, "button", "fab", "+")
	b.Set("type", "button")
	b.Call("setAttribute", "aria-label", "TODO: say what this button does")
	return b
}
`, true
	case "chip":
		return `// A chip: one small labeled pill.
func chipComponent(doc js.Value) js.Value {
	return el(doc, "span", "chip", "TODO: the chip's words")
}
`, true
	case "text-field":
		return `// A text field: its label over its input.
func textFieldComponent(doc js.Value) js.Value {
	field := el(doc, "div", "field", "")
	field.Call("appendChild", el(doc, "label", "", "TODO: the field's label"))
	input := el(doc, "input", "", "")
	input.Set("type", "text")
	input.Set("placeholder", "TODO: a hint for what to type")
	field.Call("appendChild", input)
	return field
}
`, true
	case "switch":
		return `// A switch: the sliding on/off control, in its
// labeled row. It renders switched off; this set has no
// handlers, so nothing answers when it's flipped.
func switchComponent(doc js.Value) js.Value {
	row := el(doc, "label", "switch-row", "")
	input := el(doc, "input", "switch", "")
	input.Set("type", "checkbox")
	row.Call("appendChild", input)
	row.Call("appendChild", doc.Call("createTextNode", " TODO: what this switch turns on"))
	return row
}
`, true
	case "footer":
		return `// The footer bar: the small print and its links,
// at the foot of the page.
func footerComponent(doc js.Value) js.Value {
	foot := el(doc, "footer", "site-footer", "")
	foot.Call("appendChild", el(doc, "p", "", "TODO: your footer's line — the small print that belongs at the bottom of every page."))
	links := el(doc, "ul", "site-footer-links", "")
	for _, name := range []string{"Home", "About", "Contact"} {
		item := el(doc, "li", "", "")
		link := el(doc, "a", "", name)
		link.Set("href", "#")
		item.Call("appendChild", link)
		links.Call("appendChild", item)
	}
	foot.Call("appendChild", links)
	return foot
}
`, true
	}
	return "", false
}

// planComponentWasm is the component piece's wasm mode:
// the target is an app in a folder, not a page, so the
// component lands as code — its builder joins the app's
// components.go (created with the shared helper on first
// use), one mount line lands between main.go's markers in
// ask order, and the styles it needs join the host page's
// style block exactly once, the variables first when their
// marker is missing, the component's chunk whole or not at
// all.
func (p *Project) planComponentWasm(req EditRequest, comp MaterialComponent) (*EditPlan, string) {
	folder := path.Clean(filepath.ToSlash(strings.TrimSpace(req.WasmFolder)))
	_, mainRel, _, indexRel, _ := wasmAppFiles(folder)
	compRel := wasmComponentsRel(folder)
	mainBefore, err := os.ReadFile(filepath.Join(p.Root, filepath.FromSlash(mainRel)))
	if err != nil {
		return nil, fmt.Sprintf("there's no wasm app in %s yet — say \"add a wasm app in %s\" first, then the %s can go into it.", folder, folder, comp.Words)
	}
	main := string(mainBefore)
	if !wasmAppMainContent(main) {
		return nil, fmt.Sprintf("%s isn't a wasm app's main file — it doesn't carry the js && wasm build tag a wasm app's main wears — so the %s has no app to join. Say \"add a wasm app in %s\" to make one, or add the %s to a page instead.", p.LinkPath(mainRel), comp.Words, folder, comp.Words)
	}
	begin := strings.Index(main, "// gollemer:components-begin")
	end := strings.Index(main, "// gollemer:components-end")
	if begin < 0 || end < 0 || end < begin {
		return nil, fmt.Sprintf("%s doesn't have the components block the wasm piece mounts into — the two gollemer:components marker comments in its mount function — so the %s has nowhere honest to land. Nothing was written.", p.LinkPath(mainRel), comp.Words)
	}
	indexBefore, err := os.ReadFile(filepath.Join(p.Root, filepath.FromSlash(indexRel)))
	if err != nil {
		return nil, fmt.Sprintf("%s isn't there — the host page whose style block the %s's styles join is missing, so this isn't the app the wasm piece makes. Nothing was written.", p.LinkPath(indexRel), comp.Words)
	}
	index := string(indexBefore)
	for _, rel := range []string{mainRel, compRel, indexRel} {
		if why, blocked := p.fileBlocked(rel); blocked {
			return nil, fmt.Sprintf("I won't write %s — %s.", p.LinkPath(rel), why)
		}
	}
	builder, ok := wasmComponentBuilder(comp.ID)
	if !ok {
		return nil, "that edit needs one of the material components: " + componentFormsText() + "."
	}
	fn := wasmBuilderName(comp.ID)
	compBefore, _ := os.ReadFile(filepath.Join(p.Root, filepath.FromSlash(compRel)))
	compContent := string(compBefore)
	if strings.Contains(compContent, "func "+fn+"(") || strings.Contains(main, fn+"(doc)") {
		return nil, fmt.Sprintf("the wasm app in %s already has a %s in it — nothing to add.", folder, comp.Words)
	}
	// The styles this host page still needs: the variables
	// when the marker is absent, then the component's chunk
	// when the page does not carry it whole.
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
	// The mount line lands just before the end marker's
	// line, below the mounts already there — components
	// accumulate in ask order.
	endLine := strings.LastIndex(main[:end], "\n") + 1
	newMain := main[:endLine] + "\tapp.Call(\"appendChild\", " + fn + "(doc))\n" + main[endLine:]
	newComp := ""
	if compContent == "" {
		newComp = wasmComponentsHeader + builder
	} else {
		newComp = strings.TrimRight(compContent, "\n") + "\n\n" + builder
	}
	files := []EditFile{finishFile(compRel, compBefore, []byte(newComp)), finishFile(mainRel, mainBefore, []byte(newMain))}
	if needCSS != "" {
		files = append(files, finishFile(indexRel, indexBefore, []byte(newIndex)))
	}
	summary := fmt.Sprintf("Put a %s in the wasm app in %s: %s joins %s and is mounted on the page in %s", comp.Words, folder, fn, p.LinkPath(compRel), p.LinkPath(mainRel))
	verifyDesc := fmt.Sprintf("the wasm app in %s now has a %s mounted on its page", folder, comp.Words)
	if needCSS != "" {
		summary += ", with the Material styles it needs joining the host page's style block in " + p.LinkPath(indexRel)
		verifyDesc += ", and the styles it needs are in the host page"
	} else {
		summary += " — its Material styles are already in the host page"
	}
	return &EditPlan{
		Summary:    summary,
		VerifyDesc: verifyDesc,
		Site:       p.LinkPath(mainRel),
		Files:      files,
		Notes: []string{
			"The " + comp.Words + " renders when the app is built and the page loads — it is a Go builder in the app, not HTML in the page, and this set mounts components without click handlers.",
		},
		verify: func(after *Project) bool {
			b, err := os.ReadFile(filepath.Join(after.Root, filepath.FromSlash(compRel)))
			if err != nil || string(b) != newComp {
				return false
			}
			m, err := os.ReadFile(filepath.Join(after.Root, filepath.FromSlash(mainRel)))
			if err != nil || string(m) != newMain {
				return false
			}
			if needCSS != "" {
				h, err := os.ReadFile(filepath.Join(after.Root, filepath.FromSlash(indexRel)))
				if err != nil || string(h) != newIndex {
					return false
				}
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
