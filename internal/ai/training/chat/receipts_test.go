package chat

import (
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

// The landing receipts (receipts.go): durable, structured
// records of verified landings in a sibling store, and the
// work-context resume they make possible across sessions.
//
// TestMain isolates the whole package's HOME: receipts are
// written by every landing hook, so a test run must never
// append fixture-project receipts to the developer's real
// ~/.gollemer_receipts.json (the sibling stores' tests
// isolate their own paths for the same reason).
func TestMain(m *testing.M) {
	home, err := os.MkdirTemp("", "gollemer-chat-home-")
	if err != nil {
		panic(err)
	}
	_ = os.Setenv("HOME", home)
	code := m.Run()
	_ = os.RemoveAll(home)
	os.Exit(code)
}

func isolateReceiptStore(t *testing.T) string {
	t.Helper()
	home := t.TempDir()
	oldPath := receiptStorePathOverride
	oldPending, oldAnnounced := receiptResumePending, receiptResumeAnnounced
	receiptStorePathOverride = filepath.Join(home, ".gollemer_receipts.json")
	receiptResumePending = receiptResumeState{}
	receiptResumeAnnounced = map[string]bool{}
	resetShortTermMemory()
	t.Cleanup(func() {
		receiptStorePathOverride = oldPath
		receiptResumePending, receiptResumeAnnounced = oldPending, oldAnnounced
	})
	return home
}

func mustReceipts(t *testing.T, root string) []landingReceipt {
	t.Helper()
	recs, err := landingReceiptsForRoot(root)
	if err != nil {
		t.Fatal(err)
	}
	return recs
}

func lastReceiptOfKind(recs []landingReceipt, kind string) (landingReceipt, bool) {
	for i := len(recs) - 1; i >= 0; i-- {
		if recs[i].Kind == kind {
			return recs[i], true
		}
	}
	return landingReceipt{}, false
}

func assertReceiptFiles(t *testing.T, rec landingReceipt, want map[string]string) {
	t.Helper()
	got := map[string]string{}
	for _, f := range rec.Files {
		got[f.Path] = f.Change
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("receipt %s files = %v, want %v", rec.Kind, got, want)
	}
}

func projectHashes(t *testing.T, root string) map[string]string {
	t.Helper()
	out := map[string]string{}
	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			if info.Name() == ".git" {
				return filepath.SkipDir
			}
			return nil
		}
		data, rerr := os.ReadFile(path)
		if rerr != nil {
			return rerr
		}
		rel, rerr := filepath.Rel(root, path)
		if rerr != nil {
			return rerr
		}
		out[filepath.ToSlash(rel)] = fmt.Sprintf("%x", sha256.Sum256(data))
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return out
}

// Store discipline: 0600, per-project isolation, the cap
// enforced with the oldest dropped, no temp litter.
func TestReceiptStoreDiscipline(t *testing.T) {
	home := isolateReceiptStore(t)
	rootA := t.TempDir()
	rootB := t.TempDir()
	at := time.Date(2026, time.November, 5, 12, 0, 0, 0, time.UTC)
	for i := 0; i < landingReceiptCap+5; i++ {
		rec := landingReceipt{
			At: at, Kind: "concept", Piece: "struct", Name: "Profile",
			PrimaryFile: "main.go", Verification: receiptVerificationBuilt,
			Verify: fmt.Sprintf("V%d", i),
		}
		if err := appendLandingReceipt(rootA, rec); err != nil {
			t.Fatal(err)
		}
	}
	if err := appendLandingReceipt(rootB, landingReceipt{At: at, Kind: "create_file", Verify: "created b.go", Verification: receiptVerificationReRead}); err != nil {
		t.Fatal(err)
	}

	fi, err := os.Stat(filepath.Join(home, ".gollemer_receipts.json"))
	if err != nil {
		t.Fatal(err)
	}
	if fi.Mode().Perm() != 0o600 {
		t.Errorf("receipt store mode = %o, want 600", fi.Mode().Perm())
	}
	entries, err := os.ReadDir(home)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 || entries[0].Name() != ".gollemer_receipts.json" {
		t.Errorf("store dir holds %v — the atomic write left litter", entries)
	}

	recsA := mustReceipts(t, rootA)
	if len(recsA) != landingReceiptCap {
		t.Fatalf("project A holds %d receipts, want the cap %d", len(recsA), landingReceiptCap)
	}
	if recsA[0].Verify != "V5" || recsA[len(recsA)-1].Verify != fmt.Sprintf("V%d", landingReceiptCap+4) {
		t.Errorf("cap dropped the wrong end: first %q last %q", recsA[0].Verify, recsA[len(recsA)-1].Verify)
	}
	recsB := mustReceipts(t, rootB)
	if len(recsB) != 1 || recsB[0].Verify != "created b.go" {
		t.Errorf("project B receipts = %+v — isolation broken", recsB)
	}
}

// Field-by-field receipt correctness across the families
// the acceptance names — create (folder + file), concept,
// add-server, wire, page — driven end to end through the
// claim chain: exactly one receipt per landing, and none
// for a gate-out or a refusal.
func TestReceiptsFromLandingsFieldByField(t *testing.T) {
	isolateReceiptStore(t)
	root := seedCtxProject(t)
	withChatSession(t, root, true)
	t.Chdir(root)
	profile := &UserProfile{}

	steps := []struct {
		line string
		kind string
	}{
		{"create a folder site", "create_folder"},
		{"create file site/main.go", "create_file"},
		{"add a form handler named Jake to site/main.go", "concept"},
		{"add a webserver to site/main.go", "add_server"},
		{"connect the Jake handler to the webserver", "wire_handler"},
		{"create an about page named About to site/main.go", "page"},
	}
	for i, step := range steps {
		handled, out := ctxRun(t, step.line, yes, profile)
		if !handled {
			t.Fatalf("step %d %q fell through:\n%s", i, step.line, out)
		}
		recs := mustReceipts(t, root)
		if len(recs) != i+1 {
			t.Fatalf("after step %d (%q): %d receipts, want exactly %d (one per landing)", i, step.line, len(recs), i+1)
		}
		if recs[i].Kind != step.kind {
			t.Errorf("step %d receipt kind = %q, want %q", i, recs[i].Kind, step.kind)
		}
		if recs[i].At.IsZero() {
			t.Errorf("step %d receipt carries no timestamp", i)
		}
	}
	recs := mustReceipts(t, root)

	folder, _ := lastReceiptOfKind(recs, "create_folder")
	if folder.Folder != "site" || folder.Verify != "created folder site" || folder.Verification != receiptVerificationReRead || len(folder.Files) != 0 {
		t.Errorf("folder receipt = %+v", folder)
	}
	file, _ := lastReceiptOfKind(recs, "create_file")
	if file.PrimaryFile != "site/main.go" || file.Verification != receiptVerificationBuilt || file.Verify != "created site/main.go" {
		t.Errorf("file receipt = %+v", file)
	}
	assertReceiptFiles(t, file, map[string]string{"site/main.go": "created"})
	if file.Snapshot == nil || file.Snapshot.Family != "create" || file.Snapshot.File != "site/main.go" || file.Snapshot.Folder != "site" {
		t.Errorf("file receipt snapshot = %+v", file.Snapshot)
	}

	concept, _ := lastReceiptOfKind(recs, "concept")
	if concept.Name != "Jake" || concept.Piece != "form handler" || concept.PrimaryFile != "site/main.go" ||
		concept.Verification != receiptVerificationBuilt || !strings.Contains(concept.Verify, "now declares the form handler Jake") {
		t.Errorf("concept receipt = %+v", concept)
	}
	assertReceiptFiles(t, concept, map[string]string{"site/main.go": "edited"})
	if concept.Snapshot == nil || concept.Snapshot.Family != "concept" || concept.Snapshot.Name != "Jake" ||
		concept.Snapshot.File != "site/main.go" || concept.Snapshot.ConceptID == "" {
		t.Errorf("concept receipt snapshot = %+v", concept.Snapshot)
	}

	server, _ := lastReceiptOfKind(recs, "add_server")
	if server.PrimaryFile != "site/main.go" || server.Snapshot == nil || server.Snapshot.Family != "server" {
		t.Errorf("server receipt = %+v", server)
	}

	wire, _ := lastReceiptOfKind(recs, "wire_handler")
	if wire.Name != "Jake" || wire.Route != "/jake" || wire.PrimaryFile != "site/main.go" || wire.Verification != receiptVerificationBuilt {
		t.Errorf("wire receipt = %+v", wire)
	}
	assertReceiptFiles(t, wire, map[string]string{"site/main.go": "edited"})
	if wire.Snapshot == nil || wire.Snapshot.Family != "wire" || wire.Snapshot.Name != "Jake" {
		t.Errorf("wire receipt snapshot = %+v", wire.Snapshot)
	}

	page, _ := lastReceiptOfKind(recs, "page")
	if page.Name != "About" || page.Route != "/about" || page.PrimaryFile != "site/main.go" {
		t.Errorf("page receipt = %+v", page)
	}
	assertReceiptFiles(t, page, map[string]string{"site/main.go": "edited", "site/about.html": "created"})
	if page.Snapshot == nil || page.Snapshot.Family != "page" || page.Snapshot.PageKind != "about" || page.Snapshot.Name != "About" {
		t.Errorf("page receipt snapshot = %+v", page.Snapshot)
	}

	// A declined gate writes no receipt.
	handled, out := ctxRun(t, "add a struct named Profile to site/main.go", ctxNo, profile)
	if !handled || !strings.Contains(out, "Not applied") {
		t.Fatalf("the gate-out did not decline:\n%s", out)
	}
	if got := len(mustReceipts(t, root)); got != len(steps) {
		t.Errorf("a declined gate wrote a receipt: %d, want %d", got, len(steps))
	}
	// A refusal writes no receipt either.
	handled, out = ctxRun(t, "add a form handler named Jake to site/main.go", yes, profile)
	if !handled {
		t.Fatalf("the refusal line fell through:\n%s", out)
	}
	if got := len(mustReceipts(t, root)); got != len(steps) {
		t.Errorf("a refusal wrote a receipt: %d, want %d", got, len(steps))
	}
}

// A landing the build check rolls back writes no receipt:
// the receipt hook sits after the spine's verification,
// with the note hooks it shares.
func TestReceiptsRollbackWritesNone(t *testing.T) {
	isolateReceiptStore(t)
	root := seedCtxProject(t)
	// A parseable but unbuildable file elsewhere in the
	// module: the spine's whole-project build fails, the
	// edit rolls back.
	writeFile(t, root, "broken.go", "package main\n\nfunc Broken() { Undefined() }\n")
	withChatSession(t, root, true)
	t.Chdir(root)
	profile := &UserProfile{}
	if handled, out := ctxRun(t, "create a folder api", yes, profile); !handled {
		t.Fatalf("folder create fell through:\n%s", out)
	}
	if handled, out := ctxRun(t, "create file api/one.go", yes, profile); !handled {
		t.Fatalf("file create fell through:\n%s", out)
	}
	before := len(mustReceipts(t, root))
	handled, out := ctxRun(t, "add a struct named Profile to api/one.go", yes, profile)
	if !handled || !strings.Contains(out, "restored your files") {
		t.Fatalf("the edit did not roll back:\n%s", out)
	}
	if got := len(mustReceipts(t, root)); got != before {
		t.Errorf("a rolled-back edit wrote a receipt: %d, want %d", got, before)
	}
	if got := readBack(t, root, "api/one.go"); strings.Contains(got, "Profile") {
		t.Errorf("the rolled-back edit is on disk anyway:\n%s", got)
	}
}

// Cross-session resume: session 1 lands pieces in two
// files; a fresh process state (context and short-term
// memory reset) restores the context from the receipts —
// disclosure spoken, query answered from receipts, a
// follow-up resolving against the restored files — and
// the restore itself writes nothing to the project.
func TestReceiptResumeAcrossSessions(t *testing.T) {
	isolateReceiptStore(t)
	root := seedCtxProject(t)
	withChatSession(t, root, true)
	t.Chdir(root)
	profile := &UserProfile{}
	for _, line := range []string{
		"create a folder api",
		"create file api/one.go",
		"create file api/two.go",
		"add a struct named Profile to api/one.go",
		"add a struct named Settings to api/two.go",
	} {
		if handled, out := ctxRun(t, line, yes, profile); !handled {
			t.Fatalf("session 1 line %q fell through:\n%s", line, out)
		}
	}
	hashes := projectHashes(t, root)

	// Session 2: the process state a restart resets. The
	// continuity line is asked for FIRST — at chat startup
	// it precedes the first project read — and must still
	// restore and disclose.
	resetWorkContext()
	resetShortTermMemory()
	line := beginMemorySession(root)
	withChatSession(t, root, true)
	for _, want := range []string{"Resumed from the build receipts", "Settings", "api/two.go", "api/one.go before it"} {
		if !strings.Contains(line, want) {
			t.Errorf("continuity line missing %q:\n%s", want, line)
		}
	}
	if workCtx.lastFile != "api/two.go" || workCtx.lastHow != "edited" || workCtx.prevFile != "api/one.go" ||
		workCtx.folder != "api" || workCtx.pkg != "api" || workCtx.piece != "struct" || workCtx.pieceName != "Settings" {
		t.Errorf("restored context = %+v", workCtx)
	}
	if workCtx.landed == nil || workCtx.landed.family != "concept" || workCtx.landed.name != "Settings" || workCtx.landed.file != "api/two.go" {
		t.Errorf("restored snapshot = %+v", workCtx.landed)
	}
	if !strings.Contains(sessionSummaryText(), "Settings") {
		t.Errorf("short-term built summary did not ride the restore: %q", sessionSummaryText())
	}

	// The receipts query, through the state-answer arm:
	// most recent first, with dates and verification.
	ans, ok := talkStateAnswer(TalkContext{State: systemState{Root: root}}, "what have you built here")
	if !ok {
		t.Fatal("the receipts query was not claimed by the state arm")
	}
	iSettings, iProfile := strings.Index(ans, "Settings"), strings.Index(ans, "Profile")
	if iSettings < 0 || iProfile < 0 || iSettings > iProfile {
		t.Errorf("query is not the receipts list, newest first:\n%s", ans)
	}
	if !strings.Contains(ans, "Verification: built and passed") || !strings.Contains(ans, "2026") && !strings.Contains(ans, "2025") {
		t.Errorf("query lacks verification or dates:\n%s", ans)
	}

	// Restore + query changed no project byte.
	if got := projectHashes(t, root); !reflect.DeepEqual(got, hashes) {
		t.Errorf("the restore replayed an effect:\nbefore %v\nafter  %v", hashes, got)
	}

	// The follow-up resolves against the restored context,
	// through the normal claim and gate.
	handled, out := ctxRun(t, "another one named Extra in the other file", yes, profile)
	if !handled {
		t.Fatalf("the follow-up fell through:\n%s", out)
	}
	if !strings.Contains(out, "api/one.go") {
		t.Errorf("the follow-up did not resolve to the restored other file:\n%s", out)
	}
	if got := readBack(t, root, "api/one.go"); !strings.Contains(got, "type Extra struct") {
		t.Errorf("the follow-up did not land in the restored other file:\n%s", got)
	}
}

// The stale referent: the last file deleted between
// sessions is dropped from the restored context, out loud
// — never resolved against.
func TestReceiptResumeStaleReferent(t *testing.T) {
	isolateReceiptStore(t)
	root := seedCtxProject(t)
	withChatSession(t, root, true)
	t.Chdir(root)
	profile := &UserProfile{}
	for _, line := range []string{
		"create a folder api",
		"create file api/one.go",
		"create file api/two.go",
	} {
		if handled, out := ctxRun(t, line, yes, profile); !handled {
			t.Fatalf("setup line %q fell through:\n%s", line, out)
		}
	}
	if err := os.Remove(filepath.Join(root, "api", "two.go")); err != nil {
		t.Fatal(err)
	}
	resetWorkContext()
	withChatSession(t, root, true)
	if workCtx.lastFile != "api/one.go" || workCtx.prevFile != "" {
		t.Errorf("stale context = %+v — the deleted file must drop, the surviving one stands", workCtx)
	}
	if _, ok := otherFileCandidate(); ok {
		t.Errorf("the other file resolved with only one live file in play")
	}
	line := beginMemorySession(root)
	if !strings.Contains(line, "dropped api/two.go") {
		t.Errorf("the drop was not disclosed:\n%s", line)
	}

	// Every file gone: no file referent restores — only the
	// folder receipt's context (the folder itself still
	// exists) — and the drops are still the disclosure.
	if err := os.Remove(filepath.Join(root, "api", "one.go")); err != nil {
		t.Fatal(err)
	}
	rootAbs, _ := claimRootAbs(root)
	resetWorkContext()
	restoreWorkContextFromReceipts(rootAbs)
	if workCtx.lastFile != "" || workCtx.prevFile != "" || workCtx.landed != nil {
		t.Errorf("a ghost context restored: %+v", workCtx)
	}
	if !strings.Contains(receiptResumePending.text, "dropped api/one.go, api/two.go") {
		t.Errorf("all-stale disclosure = %q", receiptResumePending.text)
	}
}

// The forget's seat in the real claim chain: "forget the
// build history" must reach the receipts claim, NOT the
// teach seat's generic "forget the <kind>" (which reads
// "build history" as a kind phrase) — the theft the live
// smoke caught. A genuine taught forget keeps its seat.
func TestReceiptForgetThroughClaimChain(t *testing.T) {
	isolateReceiptStore(t)
	root := seedCtxProject(t)
	at := time.Date(2026, time.November, 5, 9, 15, 0, 0, time.UTC)
	for _, verify := range []string{"a.go now declares the struct One", "a.go now declares the struct Two"} {
		if err := appendLandingReceipt(root, landingReceipt{At: at, Kind: "concept", PrimaryFile: "a.go", Verify: verify, Verification: receiptVerificationBuilt}); err != nil {
			t.Fatal(err)
		}
	}
	withChatSession(t, root, true)
	t.Chdir(root)
	profile := &UserProfile{}

	handled, out := ctxRun(t, "forget the build history", yes, profile)
	if !handled {
		t.Fatalf("the build-history forget fell through the chain:\n%s", out)
	}
	if strings.Contains(out, "haven't taught me") {
		t.Errorf("the teach seat answered the receipts forget:\n%s", out)
	}
	if !strings.Contains(out, "Forgotten: this project's build history — 2 landing receipts for "+canonicalMemoryRoot(root)) {
		t.Errorf("forget echo wrong:\n%s", out)
	}
	if got := mustReceipts(t, root); len(got) != 0 {
		t.Errorf("receipts survived the chained forget: %v", got)
	}

	// The teach seat keeps its own forget lines.
	handled, out = ctxRun(t, "forget the signup form", yes, profile)
	if !handled || !strings.Contains(out, "haven't taught me") {
		t.Errorf("a taught forget lost its seat:\n%s", out)
	}
}

// Forget: clears this project's receipts only, with the
// exact echo; the query after answers nothing-yet; an
// unrelated line falls through untouched.
func TestReceiptQueryAndForget(t *testing.T) {
	isolateReceiptStore(t)
	rootA, rootB := t.TempDir(), t.TempDir()
	at := time.Date(2026, time.November, 5, 9, 15, 0, 0, time.UTC)
	for _, rec := range []landingReceipt{
		{At: at, Kind: "create_file", Piece: "file", PrimaryFile: "a.go", Verify: "created a.go", Verification: receiptVerificationBuilt},
		{At: at.Add(time.Hour), Kind: "concept", Piece: "struct", Name: "Profile", PrimaryFile: "a.go", Verify: "a.go now declares the struct Profile", Verification: receiptVerificationBuilt},
	} {
		if err := appendLandingReceipt(rootA, rec); err != nil {
			t.Fatal(err)
		}
	}
	if err := appendLandingReceipt(rootB, landingReceipt{At: at, Kind: "concept", Verify: "b.go now declares the struct Other", Verification: receiptVerificationBuilt}); err != nil {
		t.Fatal(err)
	}
	ctx := TalkContext{State: systemState{Root: rootA}}

	ans, ok := talkStateAnswer(ctx, "what did you change in this project")
	if !ok || !strings.Contains(ans, "Profile") || !strings.Contains(ans, "Nov 5, 2026") {
		t.Fatalf("query answer = %q, %v", ans, ok)
	}
	if strings.Index(ans, "Profile") > strings.Index(ans, "created a.go") {
		t.Errorf("query is not newest first:\n%s", ans)
	}

	ans, ok = talkStateAnswer(ctx, "forget the build history")
	if !ok || !strings.Contains(ans, "Forgotten: this project's build history — 2 landing receipts for "+canonicalMemoryRoot(rootA)) {
		t.Fatalf("forget echo = %q, %v", ans, ok)
	}
	if got := mustReceipts(t, rootA); len(got) != 0 {
		t.Errorf("project A receipts survived the forget: %v", got)
	}
	if got := mustReceipts(t, rootB); len(got) != 1 {
		t.Errorf("project B receipts were touched by project A's forget: %v", got)
	}
	ans, ok = talkStateAnswer(ctx, "what have you built here")
	if !ok || !strings.Contains(ans, "Nothing yet") {
		t.Errorf("query after forget = %q, %v — want the honest nothing-yet", ans, ok)
	}

	// Unrelated lines keep falling through, byte for byte.
	if isTalkStateLine("what have you built this week?") {
		t.Errorf("an unrelated line matches the receipts claim")
	}
	if _, ok := talkStateAnswer(ctx, "what have you built this week?"); ok {
		t.Errorf("an unrelated line was answered by the receipts claim")
	}
}
