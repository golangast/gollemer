package chat

import (
	"fmt"
	"path"
	"strings"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// confirm.go — partner-style landing confirmations.
//
// When a piece lands, the chat used to answer with machinery:
// "Done — main.go now declares the form handler Jake in
// package main, at jim/main.go:19. I built the project and
// checked that it's really there." The facts were there, but
// they read like a build log. Now the landing line leads
// with a partner's confirmation, rendered entirely from the
// verified record of the transaction itself:
//
//   - sentence 1 says what landed and where, from the request
//     slots and the executed plan ("The form handler Jake is
//     in jim/main.go.", "Jake is in handlers.go, registered
//     at /jake.");
//   - the spine's verification sentence follows ("I built
//     the project and checked that it's really there." — or,
//     with no go.mod, the re-read disclosure): the word
//     "Done" has always carried that contract, and the suite
//     pins it, so the confirmation keeps it as its own
//     sentence rather than dropping the fact;
//   - a final sentence, ONLY where the landing owns a next
//     step in its own vocabulary, names the exact follow-up
//     line the user can type — and only when the follow-up
//     is itself verified: the wiring planner must accept the
//     handler on the re-analyzed project before the
//     confirmation may promise that "connect the Jake
//     handler to the webserver" will do it. A promise the
//     planner refuses is never printed.
//
// The confirmation prints only after the transaction's build
// verification has passed (it replaces the spine's Done line
// at exactly that point), so the word "Done" keeps its
// contract: built (or, with no go.mod, re-read) and verified.
// It never prints on a refusal, a gate-out, or a rolled-back
// edit — those paths return before the print, unchanged.
//
// Facts come from the landing record — the EditRequest, the
// executed EditPlan, and the freshly re-analyzed project —
// never from the user's sentence and never from a model. The
// session context is not consulted and not duplicated: the
// spine's own record is the fresher source, and
// noteSpineLanded has already fed the session summary from
// the same facts.

// landingFacts is the verified record of one landed plan:
// the request that was carried, the plan that executed, the
// registrations the project had before the edit (snapshotted
// before anything was written — the wiring scan reads the
// disk live, so the list cannot be re-read afterwards), and
// the project as re-analyzed after the landing.
type landingFacts struct {
	req       analyze.EditRequest
	plan      *analyze.EditPlan
	beforeReg []analyze.WireRegistration
	fresh     *analyze.Project
	built     bool   // a build ran and passed; false = the re-read check only (no go.mod)
	wireName  string // wire kind only: the handler's name when the sentence named it ("" = bare)
}

// renderLandingConfirmation renders the Done line for a
// landed edit-spine or delegate plan. Kinds with a dedicated
// template get the name-first partner grammar; every other
// kind keeps the plan's own verified clause (VerifyDesc is a
// fact the planner proved, in plain words) as sentence 1.
// Either way the verification sentence follows, and a
// verified next step, where one exists, closes the line.
func renderLandingConfirmation(f landingFacts) string {
	rel := landingPrimaryRel(f.req, f.plan)
	s1, s2 := "", ""
	switch f.req.Kind {
	case analyze.EditWireHandler:
		s1 = confirmWire(f)
	case analyze.EditPage:
		s1, s2 = confirmPage(f, rel)
	case analyze.EditServeHTML:
		s1 = confirmServeHTML(f, rel)
	case analyze.EditWebpage:
		s1 = fmt.Sprintf("The %s is in %s", analyze.WebpageTemplateWords(f.req.WebpageTemplate), f.fresh.LinkPath(rel))
	case analyze.EditServeStatic:
		s1 = fmt.Sprintf("The server in %s now serves the %s folder's files at /static/", f.fresh.LinkPath(rel), f.req.StaticFolder)
	case analyze.EditComponent:
		if f.req.WasmFolder != "" {
			if f.req.ComponentName != "" {
				s1 = fmt.Sprintf("The %s called %s is in the wasm app in %s", analyze.MaterialComponentWords(f.req.Component), f.req.ComponentName, f.fresh.LinkPath(f.req.WasmFolder))
			} else {
				s1 = fmt.Sprintf("The %s is in the wasm app in %s", analyze.MaterialComponentWords(f.req.Component), f.fresh.LinkPath(f.req.WasmFolder))
			}
		} else {
			s1 = fmt.Sprintf("The %s is in %s", analyze.MaterialComponentWords(f.req.Component), f.fresh.LinkPath(rel))
		}
	case analyze.EditWasmEvent:
		s1 = confirmWasmEvent(f)
	case analyze.EditWasmApp:
		s1 = fmt.Sprintf("The wasm app is in %s", f.fresh.LinkPath(f.req.WasmFolder))
		s2 = fmt.Sprintf("It builds with Go %s — in %s, run: %s (a first build fetches that toolchain by itself) — and once main.wasm is built, any static server can serve the folder.", analyze.WasmAppGoVersion, f.req.WasmFolder, analyze.WasmBuildCommand)
	case analyze.EditTemplatedPage:
		s1, s2 = confirmTemplatedPage(f)
	case analyze.EditScaffoldSection:
		s1 = fmt.Sprintf("The %s page's body in %s now has a section of %s",
			strings.TrimSuffix(path.Base(f.req.SiteBody), ".html"), f.fresh.LinkPath(f.req.SiteBody),
			analyze.ScaffoldSectionWords(f.req.SiteCols, f.req.SiteRows))
	case analyze.EditAddServer:
		s1 = confirmAddServer(f, rel)
	case analyze.EditConcept:
		s1, s2 = confirmConcept(f, rel)
	case analyze.EditTaughtDecl:
		s1 = fmt.Sprintf("The %s %s is in %s", f.req.TaughtKindPhrase, f.req.TaughtName, f.fresh.LinkPath(rel))
		s2 = handlerNextStep(f.fresh, f.req.TaughtName, rel)
	case analyze.EditProposedDecl:
		kind := f.req.ProposedKind
		if kind == "func" {
			kind = "function"
		}
		s1 = fmt.Sprintf("The %s %s is in %s", kind, f.req.ProposedName, f.fresh.LinkPath(rel))
		if f.req.ProposedKind == "func" {
			s2 = handlerNextStep(f.fresh, f.req.ProposedName, rel)
		}
	case analyze.EditDelegate:
		s1, s2 = confirmDelegate(f, rel)
	default:
		// Statements, fields, cases, instantiations, make-and-
		// call, signature changes and the rest of the assembled
		// edits: the plan's verified clause is already a plain
		// factual sentence ("main.go now has the print line in
		// main", "Helper now exists in a.go and is called from
		// b.go") — it stands as sentence 1 unchanged.
		s1 = f.plan.VerifyDesc
	}
	if s1 == "" {
		s1 = f.plan.VerifyDesc
	}
	out := "Done — " + s1
	if !strings.HasSuffix(out, ".") {
		out += "."
	}
	if f.built {
		out += " I built the project and checked that it's really there."
	} else {
		out += " I checked by re-reading the project — there's no go.mod at its root, so no build to run."
	}
	if s2 != "" {
		out += " " + s2
	}
	return out
}

// landingPrimaryRel settles the file a landing is "in", by
// the session context's own rule (noteSpineLanded): the
// request's named target, the delegate's file when that is
// the request's shape, else the plan's first file.
func landingPrimaryRel(req analyze.EditRequest, plan *analyze.EditPlan) string {
	if req.File != "" {
		return req.File
	}
	if req.DelegateFile != "" {
		return req.DelegateFile
	}
	if plan != nil && len(plan.Files) > 0 {
		return plan.Files[0].Rel
	}
	return ""
}

// planFileCreated reports whether the plan created rel (it
// had no original bytes) rather than editing it.
func planFileCreated(plan *analyze.EditPlan, rel string) bool {
	if plan == nil {
		return false
	}
	for _, ef := range plan.Files {
		if ef.Rel == rel {
			return len(ef.Before) == 0
		}
	}
	return false
}

// confirmWasmEvent renders the wasm event landing: the
// event itself, in the sentence that asked for it — with
// the references resolved against the freshly re-analyzed
// app first, so a component the sentence named by name
// alone is spoken of by its kind as well ("the save
// button"), the way the planner's own rendering speaks.
func confirmWasmEvent(f landingFacts) string {
	req := f.req
	if insts := f.fresh.WasmAppComponents(req.WasmFolder); len(insts) > 0 {
		if src, _ := analyze.ResolveWasmComponentRef(insts, req.EventSourceKind, req.EventSourceName); src.Builder != "" {
			req.EventSourceKind = src.Kind
		}
		if tgt, _ := analyze.ResolveWasmComponentRef(insts, req.EventTargetKind, req.EventTargetName); tgt.Builder != "" {
			req.EventTargetKind = tgt.Kind
		}
	}
	return fmt.Sprintf("The event is in the wasm app in %s: %s", f.fresh.LinkPath(req.WasmFolder), analyze.WasmEventSentence(req))
}

// confirmWire renders the wiring landing: the handler, its
// own file, and the route it now answers at. The facts come
// from the one registration this plan added (diffed between
// the before and after projects) and from the handler scan
// of the after project; anything less certain falls back to
// the plan's verified clause.
func confirmWire(f landingFacts) string {
	reg, ok := newPlanRegistration(f.beforeReg, f.fresh, f.plan)
	if !ok {
		return f.plan.VerifyDesc
	}
	name := f.wireName
	if name == "" {
		name = reg.Name
	}
	if name == "" {
		return f.plan.VerifyDesc
	}
	file := reg.File
	if hf := f.fresh.HandlerFacts(name, reg.HandlerDir); hf.IsHandler {
		file = hf.File
	}
	return fmt.Sprintf("%s is in %s, registered at %s", name, f.fresh.LinkPath(file), reg.Route)
}

// newPlanRegistration finds the one registration a plan
// added: present in the after project, absent from the
// before snapshot, and living in a file the plan touched.
// Zero or several candidates is not a fact, and reports
// not-ok.
func newPlanRegistration(before []analyze.WireRegistration, fresh *analyze.Project, plan *analyze.EditPlan) (analyze.WireRegistration, bool) {
	var none analyze.WireRegistration
	if fresh == nil || plan == nil {
		return none, false
	}
	inPlan := map[string]bool{}
	for _, ef := range plan.Files {
		inPlan[ef.Rel] = true
	}
	old := map[analyze.WireRegistration]bool{}
	for _, r := range before {
		old[r] = true
	}
	var found *analyze.WireRegistration
	for _, r := range fresh.WireRegistrations() {
		if !inPlan[r.File] || old[r] || r.Name == "" {
			continue
		}
		if found != nil {
			return none, false
		}
		reg := r
		found = &reg
	}
	if found == nil {
		return none, false
	}
	return *found, true
}

// confirmPage renders the page-family landing: the page, its
// handler, and — when the plan composed the registration —
// the route the handler now answers at. An unregistered page
// (no composable server) names its handler's next step the
// same way a handler declaration does.
func confirmPage(f landingFacts, rel string) (string, string) {
	name := f.req.PageName
	display := f.req.PageKind
	if info, ok := pageKindTable[f.req.PageKind]; ok {
		display = info.display
	}
	page := f.fresh.LinkPath(f.req.PageHTML)
	hf := f.fresh.HandlerFacts(name, slashDir(rel))
	if hf.Registered {
		return fmt.Sprintf("The %s is in %s — %s answers at %s", display, page, name, hf.Route), ""
	}
	s1 := fmt.Sprintf("The %s is in %s, and its handler %s is in %s", display, page, name, f.fresh.LinkPath(rel))
	return s1, handlerNextStep(f.fresh, name, rel)
}

// confirmServeHTML renders the serve-html landing: the file
// that serves, and the page it serves — a created program
// says so.
func confirmServeHTML(f landingFacts, rel string) string {
	page := f.fresh.LinkPath(f.req.ServeHTML)
	if planFileCreated(f.plan, rel) {
		return fmt.Sprintf("%s is a new server program, serving %s", f.fresh.LinkPath(rel), page)
	}
	return fmt.Sprintf("%s now serves %s", f.fresh.LinkPath(rel), page)
}

// confirmAddServer renders the webserver landing: a file
// that became the whole program (it was created by the
// plan, or it held nothing but a bare package clause — the
// planner's own test for the state where the file becomes
// the component whole) versus a component spliced into an
// existing main.
func confirmAddServer(f landingFacts, rel string) string {
	if f.plan != nil {
		for _, ef := range f.plan.Files {
			if ef.Rel == rel && (len(ef.Before) == 0 || strings.TrimSpace(string(ef.Before)) == "package main") {
				return fmt.Sprintf("%s is a webserver program now", f.fresh.LinkPath(rel))
			}
		}
	}
	return fmt.Sprintf("%s has a main that serves HTTP now", f.fresh.LinkPath(rel))
}

// confirmConcept renders a concept-book landing. A
// declaration gets the name-first grammar ("The form handler
// Jake is in jim/main.go.") and a handler's next step when it
// has one; a statement keeps the plan's verified clause.
func confirmConcept(f landingFacts, rel string) (string, string) {
	if f.req.Concept != nil && f.req.Concept.Kind == analyze.ConceptDeclaration && f.req.ConceptName != "" && f.req.Concept.Canonical() != "" {
		s1 := fmt.Sprintf("The %s %s is in %s", f.req.Concept.Canonical(), f.req.ConceptName, f.fresh.LinkPath(rel))
		return s1, handlerNextStep(f.fresh, f.req.ConceptName, rel)
	}
	return f.plan.VerifyDesc, ""
}

// confirmDelegate renders the delegate landing. The wrapper
// named in the sentence is a fact the chat holds; a derived
// name (the sentence named none) is the planner's to state,
// so that case keeps the plan's verified clause.
func confirmDelegate(f landingFacts, rel string) (string, string) {
	if f.req.WrapperName == "" {
		return f.plan.VerifyDesc, ""
	}
	s1 := fmt.Sprintf("%s is in %s — it calls %s", f.req.WrapperName, f.fresh.LinkPath(rel), f.req.SourceName)
	return s1, handlerNextStep(f.fresh, f.req.WrapperName, rel)
}

// confirmTemplatedPage renders the scaffold landing: the
// page's body, the shared header and footer it is joined
// to, and the handler that renders it — then the wiring
// step, but only when the wiring planner verifies it
// would land (handlerNextStep's own rule).
func confirmTemplatedPage(f landingFacts) (string, string) {
	header, footer, body, handler := analyze.TemplatedPageFiles(f.req.SiteFolder, f.req.SiteName)
	name := analyze.TemplatedPageHandlerName(f.req.SiteName)
	s1 := fmt.Sprintf("The %s templated page is ready: its body is in %s, the shared header in %s, the shared footer in %s, and its handler %s in %s",
		f.req.SiteName, f.fresh.LinkPath(body), f.fresh.LinkPath(header), f.fresh.LinkPath(footer), name, f.fresh.LinkPath(handler))
	return s1, handlerNextStep(f.fresh, name, handler)
}

// handlerNextStep names the follow-up line for a handler
// that has just landed unwired — but only a line that is
// verified to work. The wiring planner gets the first say on
// the re-analyzed project: if it would plan the wiring, the
// confirmation may promise the wiring sentence. If it would
// not (no server, a custom server, an unreachable handler)
// but the handler's own file can take a webserver, the
// add-a-webserver sentence is the honest step, and it claims
// no more than it does. Anything else is silence: no filler,
// no invented next step.
func handlerNextStep(fresh *analyze.Project, name, file string) string {
	if fresh == nil || name == "" {
		return ""
	}
	hf := fresh.HandlerFacts(name, slashDir(file))
	if !hf.IsHandler || hf.Registered {
		return ""
	}
	if plan, refusal := fresh.PlanEdit(analyze.EditRequest{
		Kind:     analyze.EditWireHandler,
		WireName: name,
	}); plan != nil && refusal == "" {
		return fmt.Sprintf("It isn't wired to the server yet — \"connect the %s handler to the webserver\" will do it.", name)
	}
	if plan, refusal := fresh.PlanEdit(analyze.EditRequest{
		Kind: analyze.EditAddServer,
		File: file,
	}); plan != nil && refusal == "" {
		return fmt.Sprintf("It has no server of its own yet — \"add a webserver to %s\" will add one.", fresh.LinkPath(file))
	}
	return ""
}

// renderCreateConfirmation renders the create family's Done
// line. A plain creation keeps its verification sentence
// (the create family's own how: built / parses / it's
// there). A webserver starter is a complete program, and
// its confirmation says the one thing that completes the
// picture: the exact line that starts it.
func renderCreateConfirmation(kind string, created []string, fileRel, how string) string {
	out := "Done — created " + strings.Join(created, " and ") + "."
	if kind == "webserver" {
		return out + fmt.Sprintf(" It's the webserver starter and it builds — \"run the program %s\" will start it.", fileRel)
	}
	if how != "" {
		out += " " + how
	}
	return out
}
