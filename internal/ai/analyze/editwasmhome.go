package analyze

// The wasm home piece (chat/wasmhome.go): a wasm app's
// home page as a file of its own — <app>/home.go, whose
// Home builder carries the welcome heading and the
// vector demo. Apps the skeleton makes carry one from
// birth; this piece gives one to an app made before they
// did, and it works by MOVING, the parts piece's honesty:
// the standard inline welcome block the old skeleton
// wrote in main.go is lifted out whole, home.go is
// created holding that same content as the Home builder,
// and main mounts Home exactly where the block stood —
// so the page renders exactly as it did, and nothing
// renders twice. A main whose welcome is not the
// standard block — edited past recognition — refuses
// plainly rather than being rewritten by guess, and a
// home.go that is already there is never overwritten.

import (
	"fmt"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// wasmLegacyHomeBlock is the welcome block the skeleton
// wrote INLINE in main.go before the home page became a
// file of its own: the exact bytes, heading through
// demo, as the old wasmAppMain wrote them. The home
// piece lifts exactly this out of an older app's main;
// anything else there is the user's own welcome, refused
// rather than rewritten.
const wasmLegacyHomeBlock = "\theading := doc.Call(\"createElement\", \"h1\")\n" +
	"\theading.Set(\"className\", \"headline\")\n" +
	"\theading.Set(\"textContent\", \"TODO: your app's name\")\n" +
	"\tapp.Call(\"appendChild\", heading)\n" +
	"\n" +
	"\t// The vector demo (vector.go): two rows of numbers the\n" +
	"\t// simd unit added, printed on the page as the proof it\n" +
	"\t// ran.\n" +
	"\tsums := AddVectors(\n" +
	"\t\t[]float32{1, 2, 3, 4, 5, 6, 7, 8},\n" +
	"\t\t[]float32{10, 20, 30, 40, 50, 60, 70, 80},\n" +
	"\t)\n" +
	"\tdemo := doc.Call(\"createElement\", \"p\")\n" +
	"\tdemo.Set(\"textContent\", fmt.Sprintf(\"The vector demo added the two rows of numbers: %v\", sums))\n" +
	"\tapp.Call(\"appendChild\", demo)\n"

// wasmMainWithHomeMount writes the home mount chunk into
// main immediately before the mountComponents call —
// after a header part's mount line (that line sits
// immediately before the same call, so the chunk lands
// below it) and before the components: the page order
// every main keeps.
func wasmMainWithHomeMount(main string) (string, bool) {
	i := strings.Index(main, "mountComponents(doc, app)")
	if i < 0 {
		return "", false
	}
	lineStart := strings.LastIndex(main[:i], "\n") + 1
	return main[:lineStart] + wasmHomeMountChunk + "\n" + main[lineStart:], true
}

// wasmMainDropImport removes one plain quoted path from
// main's import block, the block rebuilt in its own shape
// — the wasmMainWithPartMounts discipline: plain quoted
// paths, one per line, sorted. ok=false when the block
// isn't in that shape, so the caller refuses rather than
// rewriting an import block it doesn't recognise. A path
// the block doesn't carry leaves the file unchanged.
func wasmMainDropImport(main, drop string) (string, bool) {
	begin := strings.Index(main, "import (\n")
	if begin < 0 {
		return "", false
	}
	rest := main[begin+len("import (\n"):]
	end := strings.Index(rest, "\n)\n")
	if end < 0 {
		return "", false
	}
	var paths []string
	found := false
	for _, ln := range strings.Split(rest[:end], "\n") {
		ln = strings.TrimSpace(ln)
		if ln == "" {
			continue
		}
		if len(ln) < 2 || ln[0] != '"' || ln[len(ln)-1] != '"' {
			return "", false
		}
		pth, err := strconv.Unquote(ln)
		if err != nil {
			return "", false
		}
		if pth == drop {
			found = true
			continue
		}
		paths = append(paths, pth)
	}
	if !found {
		return main, true
	}
	sort.Strings(paths)
	var block strings.Builder
	block.WriteString("import (\n")
	for _, pth := range paths {
		block.WriteString("\t" + strconv.Quote(pth) + "\n")
	}
	block.WriteString(")\n")
	return main[:begin] + block.String() + rest[end+len("\n)\n"):], true
}

// planWasmHome gives the wasm app in WasmFolder a home
// page of its own (<app>/home.go, the Home builder). The
// states it meets, each answered for what it is:
//
//   - home.go there and mounted: already-there.
//   - home.go there, carrying Home, but not mounted: the
//     mount alone lands.
//   - home.go there without a Home builder: a squatter —
//     refused, never overwritten.
//   - no home.go, but main already mounts Home: the file
//     alone is created.
//   - no home.go, and main carries the standard inline
//     welcome block: the move — the block lifts out,
//     home.go is created, the mount lands where the
//     block stood, and the fmt import the block was the
//     last user of goes with it.
//   - no home.go, and main's welcome is anything else:
//     refused plainly, nothing written.
func (p *Project) planWasmHome(req EditRequest) (*EditPlan, string) {
	folder := path.Clean(filepath.ToSlash(strings.TrimSpace(req.WasmFolder)))
	if folder == "" || folder == "." {
		return nil, "a wasm app's home page lives in the app's folder, and the app can't be the project root — its go.mod makes the folder a module of its own. Say \"add a home page to the wasm app in <folder>\" with the app's folder."
	}
	_, mainRel, _, _, _ := wasmAppFiles(folder)
	homeRel := wasmHomeRel(folder)
	mainBefore, err := os.ReadFile(filepath.Join(p.Root, filepath.FromSlash(mainRel)))
	if err != nil {
		return nil, fmt.Sprintf("there's no wasm app in %s yet — say \"add a wasm app in %s\" first, then it can have a home page.", folder, folder)
	}
	main := string(mainBefore)
	if !wasmAppMainContent(main) {
		return nil, fmt.Sprintf("%s isn't a wasm app's main file — it doesn't carry the js && wasm build tag a wasm app's main wears — so there's no app here for a home page to join. Say \"add a wasm app in %s\" to make one.", p.LinkPath(mainRel), folder)
	}
	if !strings.Contains(main, "mountComponents(doc, app)") {
		return nil, fmt.Sprintf("%s doesn't call mountComponents the way the wasm piece writes it — the home page mounts just before that call, and without it it has nowhere honest to land. Nothing was written.", p.LinkPath(mainRel))
	}
	for _, rel := range []string{mainRel, homeRel} {
		if why, blocked := p.fileBlocked(rel); blocked {
			return nil, fmt.Sprintf("I won't write %s — %s.", p.LinkPath(rel), why)
		}
	}
	mountsHome := strings.Contains(main, "Home(doc, app)")
	verify := func(want map[string]string) func(*Project) bool {
		return func(after *Project) bool {
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
					if fn.Receiver == "" && fn.Name == "Home" {
						return true
					}
				}
			}
			return false
		}
	}

	homeBefore, homeErr := os.ReadFile(filepath.Join(p.Root, filepath.FromSlash(homeRel)))
	if homeErr == nil {
		content := string(homeBefore)
		if !strings.Contains(content, "func Home(") {
			return nil, fmt.Sprintf("%s is already there and doesn't carry the Home builder a home page is — I never overwrite what's yours, so nothing was written.", p.LinkPath(homeRel))
		}
		if mountsHome {
			return nil, fmt.Sprintf("the wasm app in %s already has a home page — %s, mounted in %s. Nothing to add.", folder, p.LinkPath(homeRel), p.LinkPath(mainRel))
		}
		newMain, ok := wasmMainWithHomeMount(main)
		if !ok {
			return nil, fmt.Sprintf("%s doesn't call mountComponents the way the wasm piece writes it — the home page mounts just before that call, and without it it has nowhere honest to land. Nothing was written.", p.LinkPath(mainRel))
		}
		return &EditPlan{
			Summary:    fmt.Sprintf("Mount the home page the wasm app in %s already has: %s is called from %s just before the components mount, so the welcome leads the page.", folder, p.LinkPath(homeRel), p.LinkPath(mainRel)),
			VerifyDesc: fmt.Sprintf("the wasm app in %s's home page is mounted in main.go", folder),
			Site:       p.LinkPath(mainRel),
			Files:      []EditFile{finishFile(mainRel, mainBefore, []byte(newMain))},
			Notes:      []string{fmt.Sprintf("%s stays exactly as it is — only its mount was missing.", p.LinkPath(homeRel))},
			verify:     verify(map[string]string{mainRel: newMain}),
		}, ""
	}

	if mountsHome {
		content := wasmAppHome()
		return &EditPlan{
			Summary:    fmt.Sprintf("Give the wasm app in %s its home page file: %s (the Home builder — the welcome heading and the vector demo). %s already mounts it; only the file was missing.", folder, p.LinkPath(homeRel), p.LinkPath(mainRel)),
			VerifyDesc: fmt.Sprintf("the wasm app in %s now has a home page in home.go", folder),
			Site:       p.LinkPath(mainRel),
			Files:      []EditFile{finishFile(homeRel, nil, []byte(content))},
			verify:     verify(map[string]string{homeRel: content}),
		}, ""
	}

	if !strings.Contains(main, wasmLegacyHomeBlock) {
		return nil, fmt.Sprintf("the welcome in %s isn't the standard block the wasm piece used to write inline — it has been edited into a shape I can't lift out whole, and moving it could lose your words or leave the welcome rendering twice, so I'm not giving the app a home page this way. Nothing was written.", p.LinkPath(mainRel))
	}
	newMain := strings.Replace(main, wasmLegacyHomeBlock, wasmHomeMountChunk, 1)
	if !strings.Contains(newMain, "fmt.") {
		trimmed, ok := wasmMainDropImport(newMain, "fmt")
		if !ok {
			return nil, fmt.Sprintf("%s's import block isn't in the shape the wasm piece writes — plain quoted paths, one per line — and the move would leave its fmt import unused, so I won't rewrite the block by guess. Nothing was written.", p.LinkPath(mainRel))
		}
		newMain = trimmed
	}
	content := wasmAppHome()
	return &EditPlan{
		Summary:    fmt.Sprintf("Give the wasm app in %s a home page of its own: the welcome moves out of %s into %s (the Home builder — the same heading and the same vector demo), and %s mounts it first, before the components — the page renders exactly as it did.", folder, p.LinkPath(mainRel), p.LinkPath(homeRel), p.LinkPath(mainRel)),
		VerifyDesc: fmt.Sprintf("the wasm app in %s now has a home page in home.go", folder),
		Site:       p.LinkPath(mainRel),
		Files:      []EditFile{finishFile(homeRel, nil, []byte(content)), finishFile(mainRel, mainBefore, []byte(newMain))},
		Notes: []string{
			"Nothing renders twice: the inline welcome comes out of " + p.LinkPath(mainRel) + " in the same plan that creates " + p.LinkPath(homeRel) + ", and the page's words are unchanged — only their file is new.",
		},
		verify: verify(map[string]string{homeRel: content, mainRel: newMain}),
	}, ""
}
