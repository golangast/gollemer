package chat

// Run-the-program (2026-11-01, John's screenshot): "run the
// program jake/main.go" must run THAT program. Before this
// claim existed, the sentence fell to the intent layer, which
// read it as "watch"; runWatchIt then picked the HOST project's
// own Makefile, found its "start" target, ran `make start`
// (which printed "make start opens the chat and needs a real
// terminal"), and — because that command exited 0 — narrated
// "It ran fine. What you see above is the program talking."
// The user's program never ran and the narration was invented.
//
// This claim is the deterministic fix, seated ahead of the
// beginner five (and therefore ahead of the intent layer, the
// makefile brain, and its allowlist run path):
//
//   - "run the program <path>.go" / "run <path>.go" (and the
//     "run the program in <path>.go" spelling John's second
//     screenshot used) claim the line; a missing file or a
//     non-.go path refuses by name, and "run make <target>"
//     is never this claim's shape. "start" and "launch" read
//     as the same verb (widened 2026-11-05) — the rest shapes
//     are unchanged, so "start a webserver in…" still
//     declines: this family runs programs, it does not build
//     them.
//   - Politeness is not a different ask (2026-11-09, John's
//     screenshot): "can you run jim/main.go" IS "run
//     jim/main.go" — but with "can" as the first token the
//     run shape missed, the line fell to the intent layer,
//     and the 2026-11-01 make-start failure walked back in
//     through the side door. The parser strips a closed set
//     of softeners — leading "can you" / "could you" /
//     "please" and a trailing "please" — before its shapes
//     apply, across the run, stop, log, and top shapes.
//   - The program is built first (`go build`, argv only, no
//     shell ever), so compiler errors surface verbatim as a
//     compile failure and are never narrated as a run. A file
//     whose package has sibling .go files builds as a package
//     (go run ./jake is the shape the gate shows); a lone file
//     builds as itself.
//   - The built program runs as a session-owned background
//     process. One that exits inside the startup window is
//     reported from its real exit state, with its own output.
//     One still running is reported as running — with its URL
//     when its own output (or a probe of the address its code
//     listens on) confirms a server — and the chat's prompt
//     comes back while the program keeps running.
//   - Output is captured line by line into a per-program log.
//     "show the program log" / "show visits" prints it;
//     handler visits are in it when the program prints them
//     (servers Gollemer creates print one line per visit).
//     There is no proxy and no interception.
//   - "top" / "what's running" lists what this chat itself
//     started — path, how long, its URL, lines and visits —
//     with exited programs listed as exited, never as live.
//   - "stop the program" (and friends) kills it. Bare "stop"
//     stops the program while one runs and keeps its old
//     chat-quitting meaning when none does — the loop's quit
//     check consults stopMeansQuit for exactly that fork.
//     /quit and EOF kill any running program on the way out.
//
// Every status sentence is rendered from the process's real
// state by the helpers below — never from what the command
// was supposed to do.

import (
	"context"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"
)

// ---------------------------------------------------------------------
// Parsing: the claim's shapes, kept tight on purpose.

type programShape int

const (
	shapeNone programShape = iota
	shapeRunProgram
	shapeStopProgram
	shapeStopBare // bare "stop": this claim's only while a program runs
	shapeShowLog
	shapeTop
)

// parseProgramLine classifies one chat line for this claim.
// target is the path as typed ("" when the shape carries no
// path). A run shape whose token merely looks like a path (it
// has a dot or a slash but no .go ending) still classifies as
// shapeRunProgram so the claim can refuse it by name; a token
// with neither ("run make …", "run it") never classifies.
// Politeness wrappers come off before the shapes apply
// (stripProgramSofteners) — the ask inside them is the ask.
func parseProgramLine(line string) (programShape, string) {
	fields := strings.Fields(strings.TrimSpace(line))
	if len(fields) == 0 {
		return shapeNone, ""
	}
	fold := func(s string) string {
		return strings.ToLower(strings.TrimRight(s, ".!?"))
	}
	fields = stripProgramSofteners(fields, fold)
	if len(fields) == 0 {
		return shapeNone, ""
	}
	// pathy reports (isGo, looksLikePath) for a token.
	pathy := func(tok string) (bool, bool) {
		t := fold(tok)
		if strings.HasSuffix(t, ".go") {
			return true, true
		}
		return false, strings.ContainsAny(t, "./")
	}
	// The top shapes are whole-line: "top" asks what this chat
	// itself is running (never an OS process table).
	joinedAll := strings.ToLower(strings.TrimRight(strings.Join(fields, " "), ".!?"))
	switch joinedAll {
	case "top", "/top", "what's running", "show running programs":
		return shapeTop, ""
	}
	switch fold(fields[0]) {
	case "run", "start", "launch":
		rest := fields[1:]
		if len(rest) == 1 {
			if _, p := pathy(rest[0]); p {
				return shapeRunProgram, strings.TrimRight(rest[0], ".!?")
			}
		}
		if len(rest) == 2 && fold(rest[0]) == "program" {
			if _, p := pathy(rest[1]); p {
				return shapeRunProgram, strings.TrimRight(rest[1], ".!?")
			}
		}
		if len(rest) == 3 && fold(rest[0]) == "the" && fold(rest[1]) == "program" {
			if _, p := pathy(rest[2]); p {
				return shapeRunProgram, strings.TrimRight(rest[2], ".!?")
			}
		}
		if len(rest) == 4 && fold(rest[0]) == "the" && fold(rest[1]) == "program" && fold(rest[2]) == "in" {
			if _, p := pathy(rest[3]); p {
				return shapeRunProgram, strings.TrimRight(rest[3], ".!?")
			}
		}
	case "stop":
		rest := fields[1:]
		if len(rest) == 0 {
			return shapeStopBare, ""
		}
		joined := strings.ToLower(strings.Join(fields, " "))
		joined = strings.TrimRight(joined, ".!?")
		switch joined {
		case "stop it", "stop the program", "stop the server":
			return shapeStopProgram, ""
		}
		if len(rest) == 1 {
			if g, _ := pathy(rest[0]); g {
				return shapeStopProgram, strings.TrimRight(rest[0], ".!?")
			}
		}
		if len(rest) == 3 && fold(rest[0]) == "the" && fold(rest[1]) == "program" {
			if g, _ := pathy(rest[2]); g {
				return shapeStopProgram, strings.TrimRight(rest[2], ".!?")
			}
		}
	case "show":
		joined := strings.ToLower(strings.Join(fields, " "))
		joined = strings.TrimRight(joined, ".!?")
		switch joined {
		case "show the program log", "show program log", "show the log",
			"show visits", "show the visits",
			"show me the program log", "show me the log", "show me visits":
			return shapeShowLog, ""
		}
	}
	return shapeNone, ""
}

// stripProgramSofteners removes the closed set of politeness
// wrappers this claim tolerates around its shapes: a leading
// "can you", "could you", or "please", and a trailing
// "please" — the same wrappers the phrasing layer strips
// (phrasingWrapperPrefixes / phrasingWrapperSuffixes),
// applied here so the program shapes see the bare ask.
// Stacked softeners ("can you please run …") come off one at
// a time; anything outside the closed set ("will you run …")
// is left exactly as typed, and the shapes stay as tight as
// they were — stripping can only reveal a shape that was
// already there, never invent one.
func stripProgramSofteners(fields []string, fold func(string) string) []string {
	for len(fields) > 0 && fold(fields[len(fields)-1]) == "please" {
		fields = fields[:len(fields)-1]
	}
	for len(fields) > 0 {
		switch {
		case len(fields) >= 2 && fold(fields[0]) == "can" && fold(fields[1]) == "you":
			fields = fields[2:]
		case len(fields) >= 2 && fold(fields[0]) == "could" && fold(fields[1]) == "you":
			fields = fields[2:]
		case fold(fields[0]) == "please":
			fields = fields[1:]
		default:
			return fields
		}
	}
	return fields
}

// ---------------------------------------------------------------------
// Session state: the one program this chat session is running.

const programLogCap = 400

// programLog is the running program's captured output, line by
// line: the last programLogCap lines are kept for display,
// total counts every line ever seen, and notified is the total
// the prompt-cycle notice has already reported.
type programLog struct {
	mu       sync.Mutex
	lines    []string
	total    int
	notified int
}

func (l *programLog) add(line string) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.total++
	l.lines = append(l.lines, line)
	if extra := len(l.lines) - programLogCap; extra > 0 {
		l.lines = append([]string(nil), l.lines[extra:]...)
	}
}

func (l *programLog) snapshot() (lines []string, total int) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return append([]string(nil), l.lines...), l.total
}

// lineWriter adapts the program's stdout/stderr to the log:
// bytes are split into lines as they arrive; a trailing
// partial line is held until its newline or flush.
type lineWriter struct {
	mu  sync.Mutex
	buf []byte
	log *programLog
}

func (w *lineWriter) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.buf = append(w.buf, p...)
	for {
		i := indexByte(w.buf, '\n')
		if i < 0 {
			break
		}
		w.log.add(strings.TrimRight(string(w.buf[:i]), "\r"))
		w.buf = w.buf[i+1:]
	}
	return len(p), nil
}

func (w *lineWriter) flush() {
	w.mu.Lock()
	defer w.mu.Unlock()
	if len(w.buf) > 0 {
		w.log.add(string(w.buf))
		w.buf = nil
	}
}

func indexByte(b []byte, c byte) int {
	for i, x := range b {
		if x == c {
			return i
		}
	}
	return -1
}

// programRun is one started program: the process, its log,
// and the facts the chat is allowed to narrate — all of them
// observed, none assumed.
type programRun struct {
	rel     string // the file as named, slash form
	binPath string
	cmd     *exec.Cmd
	log     *programLog
	done    chan struct{}

	startedAt time.Time

	mu       sync.Mutex
	url      string // confirmed serving URL; "" until one is
	exited   bool
	exitErr  error
	exitedAt time.Time
	exitTold bool // the exit has been surfaced to the user
}

func (p *programRun) hasExited() bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.exited
}

func (p *programRun) exitError() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.exitErr
}

func (p *programRun) confirmedURL() string {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.url
}

func (p *programRun) setURL(u string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.url = u
}

// sessionProgram is the session's one program (running or
// most recently run), like pendingGorqlite is the session's
// one held dependency ask. In memory only; never on disk.
var sessionProgram *programRun

// programRunning reports whether the session's program is
// alive right now — the fact the bare-"stop" fork turns on.
func programRunning() bool {
	p := sessionProgram
	return p != nil && !p.hasExited()
}

// stopMeansQuit reports whether this line ends the chat. Bare
// "stop" is one of the chat's quit words — except while a
// program runs, when it plainly means the program (John,
// 2026-11-01: "have it stop the program if you tell it to
// stop it"). The unified loop's quit check calls this so the
// fork lives in exactly one place.
func stopMeansQuit(line string) bool {
	if line != "stop" {
		return false
	}
	return !programRunning()
}

// ---------------------------------------------------------------------
// The claim.

// tryRunProgram is the chat-loop entry, seated ahead of the
// beginner five.
func tryRunProgram(line string, r *LineReader, profile *UserProfile) bool {
	return runRunProgram(line, func(prompt string) (string, error) {
		return r.ReadLine(prompt)
	}, profile)
}

// runRunProgram carries one line for this claim. It returns
// false for every line outside the claim's shapes, so the
// rest of the chain sees them untouched.
func runRunProgram(line string, ask editAsk, profile *UserProfile) bool {
	shape, target := parseProgramLine(line)
	switch shape {
	case shapeNone:
		return false
	case shapeShowLog:
		if sessionProgram == nil {
			return false
		}
		showProgramLog()
		return true
	case shapeTop:
		showProgramTop()
		return true
	case shapeStopBare:
		if !programRunning() {
			return false // the loop's quit check owns bare "stop" then
		}
		stopSessionProgram()
		return true
	case shapeStopProgram:
		stopProgramClaim(target)
		return true
	case shapeRunProgram:
		runProgramClaim(target, ask, profile)
		return true
	}
	return false
}

// runProgramGate is the established [y/n] run gate: the
// profile's 'do all' directive auto-yeses with the same
// announcement confirmAction prints; otherwise the answer is
// read through the session's one ask source.
func runProgramGate(ask editAsk, profile *UserProfile, prompt string) bool {
	if profile != nil && profile.ShouldAutoBuild() {
		fmt.Printf("%s [auto-yes: your 'do all' directive]\n", prompt)
		return true
	}
	ans, err := ask(prompt + " [y/n] ")
	if err != nil {
		return false
	}
	a := strings.ToLower(strings.TrimSpace(ans))
	return a == "y" || a == "yes"
}

// runProgramClaim validates the ask, gates it, builds, and
// starts the program. Every early exit answers honestly and
// by name; nothing here claims a run that did not happen.
func runProgramClaim(target string, ask editAsk, profile *UserProfile) {
	tag := cTag("gollemer [runner]>")
	rel := filepath.ToSlash(strings.TrimSpace(strings.TrimPrefix(target, "./")))
	if !strings.HasSuffix(strings.ToLower(rel), ".go") {
		fmt.Printf("%s %s isn't a Go file — I run programs from .go files. Name one, like \"run the program jake/main.go\".\n", tag, target)
		recordBrain("runner", false, "program run refused: not a .go file: "+target)
		return
	}
	// One program at a time: a live one answers for itself.
	if live := sessionProgram; live != nil && !live.hasExited() {
		if live.rel == rel {
			fmt.Printf("%s %s\n", tag, programStatusText(live))
			recordBrain("runner", true, "program already running: "+rel)
			return
		}
		fmt.Printf("%s One at a time: %s is still running. Say \"stop the program\" first, then run %s.\n", tag, live.rel, rel)
		recordBrain("runner", false, "program run refused: "+live.rel+" still running")
		return
	}
	rootAbs, ok := runProgramRoot(tag)
	if !ok {
		return
	}
	abs, refusal := runProgramResolve(rootAbs, rel)
	if refusal != "" {
		fmt.Printf("%s %s\n", tag, refusal)
		recordBrain("runner", false, "program run refused: "+refusal)
		return
	}
	if st, err := os.Stat(abs); err != nil || st.IsDir() {
		fmt.Printf("%s I can't run %s — there's no such file in this project.\n", tag, rel)
		recordBrain("runner", false, "program run refused: missing file: "+rel)
		return
	}
	src, err := os.ReadFile(abs)
	if err != nil {
		fmt.Printf("%s I couldn't read %s: %v.\n", tag, rel, err)
		recordBrain("runner", false, "program run refused: unreadable: "+rel)
		return
	}
	if refusal := checkRunnableProgram(rel, src); refusal != "" {
		fmt.Printf("%s %s\n", tag, refusal)
		recordBrain("runner", false, "program run refused: "+refusal)
		return
	}
	buildArg, cmdText, formNote := programForm(rootAbs, rel)
	prompt := "Run it? " + cmdText
	if formNote != "" {
		prompt += " (" + formNote + ")"
	}
	if !runProgramGate(ask, profile, prompt) {
		fmt.Printf("%s OK — not run.\n", tag)
		recordBrain("runner", true, "program run declined: "+rel)
		return
	}
	binPath, buildOut, buildErr := buildProgram(rootAbs, buildArg)
	if buildErr != nil {
		fmt.Printf("%s %s\n", tag, programBuildFailText(rel, buildOut))
		recordBrain("runner", false, "program build failed: "+rel)
		return
	}
	startProgram(tag, rel, rootAbs, binPath, src)
}

// runProgramRoot resolves the folder programs run in: the
// open project's root, else the folder the chat started in —
// the same rule creations land by (creationRoot), with the
// same symlink resolution runCreate performs.
func runProgramRoot(tag string) (string, bool) {
	rootRaw := creationRoot()
	if rootRaw == "" {
		fmt.Printf("%s I don't have a project folder to run in — start me inside your project and ask again.\n", tag)
		recordBrain("runner", false, "program run refused: no root")
		return "", false
	}
	rootAbs, err := filepath.EvalSymlinks(rootRaw)
	if err != nil {
		if abs, aerr := filepath.Abs(rootRaw); aerr == nil {
			rootAbs = abs
		} else {
			fmt.Printf("%s I couldn't find the folder I'd run in (%s).\n", tag, rootRaw)
			recordBrain("runner", false, "program run refused: root unreadable")
			return "", false
		}
	}
	return rootAbs, true
}

// runProgramResolve joins rel onto the root and refuses any
// path that climbs out of the project.
func runProgramResolve(rootAbs, rel string) (string, string) {
	abs := filepath.Join(rootAbs, filepath.FromSlash(rel))
	back, err := filepath.Rel(rootAbs, abs)
	if err != nil || back == ".." || strings.HasPrefix(back, ".."+string(filepath.Separator)) {
		return "", fmt.Sprintf("%s is outside this project — I only run programs that live in it.", rel)
	}
	return abs, ""
}

// checkRunnableProgram reads the file as Go and refuses, by
// name, anything that is not a runnable program: a library
// package, or a main package with no func main.
func checkRunnableProgram(rel string, src []byte) string {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, rel, src, parser.SkipObjectResolution)
	if err != nil {
		return fmt.Sprintf("I couldn't read %s as Go code: %v.", rel, err)
	}
	if f.Name.Name != "main" {
		return fmt.Sprintf("%s is package %s — a program I can run is a package main file with a func main.", rel, f.Name.Name)
	}
	for _, d := range f.Decls {
		if fd, ok := d.(*ast.FuncDecl); ok && fd.Recv == nil && fd.Name.Name == "main" {
			return ""
		}
	}
	return fmt.Sprintf("%s has no func main — there's no program to run in it.", rel)
}

// programForm decides file form vs package form. A file that
// shares its directory with other non-test .go files is part
// of a multi-file package, and running the file alone would
// drop its siblings' code — so the package runs whole
// (go build ./jake). A lone file builds as itself. The gate's
// command text is the `go run` shape a person would type for
// the same choice.
func programForm(rootAbs, rel string) (buildArg, cmdText, formNote string) {
	slashDir := filepath.ToSlash(filepath.Dir(filepath.FromSlash(rel)))
	count := 0
	if entries, err := os.ReadDir(filepath.Join(rootAbs, filepath.FromSlash(slashDir))); err == nil {
		for _, e := range entries {
			name := e.Name()
			if !e.IsDir() && strings.HasSuffix(name, ".go") && !strings.HasSuffix(name, "_test.go") {
				count++
			}
		}
	}
	if count > 1 {
		arg := "."
		if slashDir != "." {
			arg = "./" + slashDir
		}
		return arg, "go run " + arg,
			fmt.Sprintf("the whole package — %s has sibling files, and the program is all of them together", rel)
	}
	return rel, "go run " + rel, ""
}

// buildProgram compiles the target into a throwaway binary.
// Building first is what keeps the phases honest: a compile
// failure is reported as one, with the compiler's own output,
// and no process ever starts. The binary is removed when the
// run ends (the watcher goroutine owns that).
func buildProgram(rootAbs, buildArg string) (string, string, error) {
	f, err := os.CreateTemp("", "gollemer-run-*")
	if err != nil {
		return "", "", err
	}
	binPath := f.Name()
	f.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	cmd := exec.CommandContext(ctx, "go", "build", "-o", binPath, filepath.FromSlash(buildArg))
	cmd.Dir = rootAbs
	out, err := cmd.CombinedOutput()
	if err != nil {
		os.Remove(binPath)
		return "", string(out), err
	}
	return binPath, "", nil
}

// ---------------------------------------------------------------------
// Lifecycle: start, watch, report, stop.

// startupWindow is how long the claim waits at start for the
// program to declare itself: exit (a quick program — report
// its real result) or print a serving URL that answers a
// probe (a server — report it running and open the browser).
const startupWindow = 3 * time.Second

// startProgram launches the built binary as the session's
// program and narrates what actually happens inside the
// startup window.
func startProgram(tag, rel, rootAbs, binPath string, src []byte) {
	p := &programRun{
		rel:       rel,
		binPath:   binPath,
		log:       &programLog{},
		done:      make(chan struct{}),
		startedAt: time.Now(),
	}
	lw := &lineWriter{log: p.log}
	cmd := exec.Command(binPath)
	cmd.Dir = rootAbs
	cmd.Stdout = lw
	cmd.Stderr = lw
	if err := cmd.Start(); err != nil {
		os.Remove(binPath)
		fmt.Printf("%s I couldn't start %s: %v.\n", tag, rel, err)
		recordBrain("runner", false, "program start failed: "+rel)
		return
	}
	p.cmd = cmd
	sessionProgram = p
	go func() {
		err := cmd.Wait()
		lw.flush()
		p.mu.Lock()
		p.exited = true
		p.exitErr = err
		p.exitedAt = time.Now()
		p.mu.Unlock()
		os.Remove(binPath)
		close(p.done)
	}()
	fmt.Printf("%s [running: %s — its output is being kept]\n", tag, rel)

	deadline := time.Now().Add(startupWindow)
	announced := false
	for {
		if p.hasExited() {
			reportProgramExit(p)
			return
		}
		if p.confirmedURL() == "" {
			if u := logPrintedURL(p); u != "" {
				if confirmServing(p, u, deadline) {
					p.setURL(u)
					announceRunning(p)
					announceBrowser(p)
					announced = true
				}
			}
		}
		if announced || time.Now().After(deadline) {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	if announced {
		return
	}
	if p.hasExited() {
		reportProgramExit(p)
		return
	}
	// Still running at the window's end. A server that prints
	// nothing can still be confirmed: probe the address its
	// own code listens on.
	if p.confirmedURL() == "" {
		if u := listenURLFromSource(src); u != "" && probeServing(u) {
			p.setURL(u)
			announceRunning(p)
			announceBrowser(p)
			return
		}
	}
	announceRunning(p)
}

// logPrintedURL returns the first URL the program itself has
// printed so far, or "".
func logPrintedURL(p *programRun) string {
	lines, _ := p.log.snapshot()
	for _, ln := range lines {
		if u := urlFromLine(ln); u != "" {
			return u
		}
	}
	return ""
}

var programURLRe = regexp.MustCompile(`https?://[^\s"',;]+`)

// urlFromLine extracts a serving URL from one output line.
func urlFromLine(line string) string {
	m := programURLRe.FindString(line)
	return strings.TrimRight(m, ".)")
}

// confirmServing probes the printed URL until it answers or
// the startup window closes. A process that dies mid-probe
// (the classic port-already-in-use exit right after its
// "listening" line) never confirms — its exit report tells
// the real story instead.
func confirmServing(p *programRun, url string, deadline time.Time) bool {
	for {
		if p.hasExited() {
			return false
		}
		if probeServing(url) {
			return !p.hasExited()
		}
		if time.Now().After(deadline) {
			return false
		}
		time.Sleep(150 * time.Millisecond)
	}
}

// probeServing reports whether something accepts a TCP
// connection at the URL's address. 0.0.0.0 is dialed as
// 127.0.0.1.
func probeServing(url string) bool {
	hostport := strings.TrimPrefix(strings.TrimPrefix(url, "http://"), "https://")
	if i := strings.Index(hostport, "/"); i >= 0 {
		hostport = hostport[:i]
	}
	if strings.HasPrefix(hostport, "0.0.0.0") {
		hostport = "127.0.0.1" + strings.TrimPrefix(hostport, "0.0.0.0")
	}
	if !strings.Contains(hostport, ":") {
		hostport += ":80"
	}
	conn, err := net.DialTimeout("tcp", hostport, 400*time.Millisecond)
	if err != nil {
		return false
	}
	conn.Close()
	return true
}

var listenAddrRe = regexp.MustCompile(`ListenAndServe\(\s*"([^"]+)"`)

// listenURLFromSource reads the address a ListenAndServe call
// in the source listens on and renders it as a localhost URL.
// "" when the source names no literal address.
func listenURLFromSource(src []byte) string {
	m := listenAddrRe.FindSubmatch(src)
	if m == nil {
		return ""
	}
	addr := string(m[1])
	if strings.HasPrefix(addr, ":") {
		return "http://localhost" + addr
	}
	if !strings.Contains(addr, "://") {
		return "http://" + addr
	}
	return addr
}

// ---------------------------------------------------------------------
// Narration: every sentence rendered from observed state.

// programBuildFailText is the compile-failure report: the
// compiler's own output, verbatim up to a readable cap. The
// word "ran" appears nowhere in it — nothing ran.
func programBuildFailText(rel, out string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "It didn't run — %s doesn't compile yet. The compiler's own words:\n", rel)
	lines := strings.Split(strings.TrimRight(out, "\n"), "\n")
	if strings.TrimSpace(out) == "" {
		lines = nil
	}
	const maxLines = 40
	shown := lines
	if len(shown) > maxLines {
		shown = shown[:maxLines]
	}
	for _, l := range shown {
		b.WriteString(l + "\n")
	}
	if len(lines) > maxLines {
		fmt.Fprintf(&b, "[… %d more lines not shown]\n", len(lines)-maxLines)
	}
	b.WriteString("Fix those, then say \"run the program " + rel + "\" again.")
	return strings.TrimRight(b.String(), "\n")
}

// outputShowsBindFailure reports whether the program's own
// words say its listen failed because the address was taken.
// It matters because a server whose bind fails usually
// prints the error and returns from main — exit code 0 — so
// the exit state alone would read as a clean finish for a
// program that never served anything. The narration checks
// the words too; both are observed state.
func outputShowsBindFailure(lines []string) bool {
	for _, l := range lines {
		if strings.Contains(l, "address already in use") {
			return true
		}
	}
	return false
}

// reportProgramExit narrates a program that ended inside the
// startup window: its own output first, then a status line
// built from its real exit state.
func reportProgramExit(p *programRun) {
	tag := cTag("gollemer [runner]>")
	lines, _ := p.log.snapshot()
	printProgramLines(lines)
	err := p.exitError()
	p.mu.Lock()
	p.exitTold = true
	p.mu.Unlock()
	ok := err == nil
	switch {
	case err == nil && outputShowsBindFailure(lines):
		fmt.Printf("%s It stopped right away — the address it wanted was already in use, so it never got to serve. Its own words are above.\n", tag)
		ok = false
	case err == nil && len(lines) == 0:
		fmt.Printf("%s It ran and finished (exit code 0), and printed nothing.\n", tag)
	case err == nil:
		fmt.Printf("%s It ran and finished (exit code 0). The lines above are %s's own output.\n", tag, p.rel)
	case len(lines) == 0:
		fmt.Printf("%s It stopped with an error (%v) and printed nothing.\n", tag, err)
	default:
		fmt.Printf("%s It stopped with an error (%v). The lines above are its own output — paste one back and I'll explain it.\n", tag, err)
	}
	if ok {
		recordBrain("runner", true, "program ran and finished: "+p.rel)
	} else {
		recordBrain("runner", false, "program stopped with an error: "+p.rel)
	}
}

// printProgramLines prints captured output the way the watch
// narration does: raw, capped, with the hidden count named.
func printProgramLines(lines []string) {
	const maxLines = 40
	shown := lines
	if len(shown) > maxLines {
		shown = shown[:maxLines]
	}
	for _, l := range shown {
		fmt.Println(l)
	}
	if len(lines) > maxLines {
		fmt.Printf("[… %d more lines not shown]\n", len(lines)-maxLines)
	}
}

// announceRunning narrates a program that is still alive at
// the end of its startup window.
func announceRunning(p *programRun) {
	tag := cTag("gollemer [runner]>")
	if u := p.confirmedURL(); u != "" {
		fmt.Printf("%s %s is running in the background — serving at %s. Say \"show visits\" to read its log (handler visits included), or \"stop the program\" to stop it.\n", tag, p.rel, u)
		recordBrain("runner", true, "program running as a server: "+p.rel)
		return
	}
	if u := logPrintedURL(p); u != "" {
		fmt.Printf("%s %s is running in the background. It printed %s, but I couldn't reach that address yet — say \"show the program log\" to watch its output, or \"stop the program\" to stop it.\n", tag, p.rel, u)
		recordBrain("runner", true, "program running, address unconfirmed: "+p.rel)
		return
	}
	fmt.Printf("%s %s is running in the background. Say \"show the program log\" to read what it prints, or \"stop the program\" to stop it.\n", tag, p.rel)
	recordBrain("runner", true, "program running in the background: "+p.rel)
}

// programStatusText is the live answer to running a program
// that is already running: its state and URL, observed.
func programStatusText(p *programRun) string {
	_, total := p.log.snapshot()
	if u := p.confirmedURL(); u != "" {
		return fmt.Sprintf("%s is already running — serving at %s, with %d lines in its log so far. Say \"show visits\" to read them, or \"stop the program\" to stop it.", p.rel, u, total)
	}
	return fmt.Sprintf("%s is already running — %d lines in its log so far. Say \"show the program log\" to read them, or \"stop the program\" to stop it.", p.rel, total)
}

// announceBrowser performs the browser step for a confirmed
// server: announce and open, or say plainly that the open
// failed and give the address. Part of the run the user
// already approved — no second gate.
func announceBrowser(p *programRun) {
	tag := cTag("gollemer [runner]>")
	u := p.confirmedURL()
	if u == "" {
		return
	}
	if err := openBrowserURL(u); err != nil {
		fmt.Printf("%s I couldn't open your browser (%v) — the address is %s.\n", tag, err, u)
		return
	}
	fmt.Printf("%s Opening %s in your browser.\n", tag, u)
}

// openBrowserURL is the platform browser opener (see the
// runprogram_open_*.go files); a variable so tests can stand
// in for it and no test ever opens a real browser.
var openBrowserURL = platformOpenBrowser

// ---------------------------------------------------------------------
// The log and stop claims.

// showProgramLog prints the session program's captured
// output and its current state. Visits are here when the
// program prints them — the answer says so plainly, because
// for a hand-written program that is the whole contract.
func showProgramLog() {
	tag := cTag("gollemer [runner]>")
	p := sessionProgram
	if p == nil {
		return
	}
	lines, total := p.log.snapshot()
	p.log.mu.Lock()
	p.log.notified = total
	p.log.mu.Unlock()
	if total == 0 {
		fmt.Printf("%s %s hasn't printed anything yet.\n", tag, p.rel)
	} else {
		fmt.Printf("%s Here's what %s has printed (%d lines):\n", tag, p.rel, total)
		const maxShown = 50
		shown := lines
		if len(shown) > maxShown {
			fmt.Printf("[… showing the last %d of %d lines]\n", maxShown, total)
			shown = shown[len(shown)-maxShown:]
		}
		for _, l := range shown {
			fmt.Println(l)
		}
	}
	fmt.Printf("%s Visits appear in this log when the program prints them — servers Gollemer creates print one line per visit.\n", tag)
	if p.hasExited() {
		p.mu.Lock()
		p.exitTold = true
		err := p.exitErr
		p.mu.Unlock()
		if err == nil && outputShowsBindFailure(lines) {
			fmt.Printf("%s It stopped right away — the address it wanted was already in use, so it never got to serve.\n", tag)
		} else if err == nil {
			fmt.Printf("%s It has finished (exit code 0) — it's no longer running.\n", tag)
		} else {
			fmt.Printf("%s It has stopped (%v) — it's no longer running.\n", tag, err)
		}
		return
	}
	fmt.Printf("%s It's still running.\n", tag)
}

// stopProgramClaim answers an explicit stop shape ("stop the
// program", "stop it", "stop <file>.go"): with nothing
// running it says so; with a different program running it
// names the right one; otherwise it stops the program.
func stopProgramClaim(target string) {
	tag := cTag("gollemer [runner]>")
	p := sessionProgram
	if p == nil {
		fmt.Printf("%s No program is running right now — there's no program to stop.\n", tag)
		recordBrain("runner", true, "stop asked with nothing running")
		return
	}
	if target != "" {
		want := filepath.ToSlash(strings.TrimPrefix(target, "./"))
		if want != p.rel {
			if p.hasExited() {
				fmt.Printf("%s %s isn't running — it already stopped on its own.\n", tag, want)
			} else {
				fmt.Printf("%s %s isn't the program that's running — %s is. Say \"stop the program\" to stop it.\n", tag, want, p.rel)
			}
			recordBrain("runner", true, "stop asked for a different program")
			return
		}
	}
	if p.hasExited() {
		if err := p.exitError(); err == nil {
			fmt.Printf("%s %s already finished on its own (exit code 0) — nothing to stop.\n", tag, p.rel)
		} else {
			fmt.Printf("%s %s already stopped on its own (%v) — nothing to stop.\n", tag, p.rel, err)
		}
		p.mu.Lock()
		p.exitTold = true
		p.mu.Unlock()
		recordBrain("runner", true, "stop asked after the program ended: "+p.rel)
		return
	}
	stopSessionProgram()
}

// stopSessionProgram kills the session's running program and
// says so. The log stays readable afterwards.
func stopSessionProgram() {
	tag := cTag("gollemer [runner]>")
	p := sessionProgram
	if p == nil || p.hasExited() {
		return
	}
	_ = p.cmd.Process.Kill()
	select {
	case <-p.done:
	case <-time.After(3 * time.Second):
	}
	p.mu.Lock()
	p.exitTold = true
	p.mu.Unlock()
	_, total := p.log.snapshot()
	fmt.Printf("%s Stopped %s — it's no longer running. Its log kept %d lines; say \"show the program log\" to read them.\n", tag, p.rel, total)
	recordBrain("runner", true, "program stopped: "+p.rel)
}

// stopProgramAtChatEnd is the loop's exit hook (/quit and EOF
// both land here): a program the session started never
// outlives the chat.
func stopProgramAtChatEnd() {
	if !programRunning() {
		return
	}
	p := sessionProgram
	_ = p.cmd.Process.Kill()
	select {
	case <-p.done:
	case <-time.After(3 * time.Second):
	}
	fmt.Printf("[stopped %s — the chat is closing]\n", p.rel)
}

// ---------------------------------------------------------------------
// The top claim: what THIS chat is running.

// pluralCount renders "1 second" / "5 seconds" style counts.
func pluralCount(n int64, unit string) string {
	if n == 1 {
		return "1 " + unit
	}
	return fmt.Sprintf("%d %ss", n, unit)
}

// plainDuration renders a duration the way a person says it:
// "5 seconds", "1 minute 20 seconds", "2 hours 3 minutes".
func plainDuration(d time.Duration) string {
	s := int64(d.Seconds())
	if s < 0 {
		s = 0
	}
	switch {
	case s < 60:
		return pluralCount(s, "second")
	case s < 3600:
		m, r := s/60, s%60
		if r == 0 {
			return pluralCount(m, "minute")
		}
		return pluralCount(m, "minute") + " " + pluralCount(r, "second")
	default:
		h, m := s/3600, (s%3600)/60
		if m == 0 {
			return pluralCount(h, "hour")
		}
		return pluralCount(h, "hour") + " " + pluralCount(m, "minute")
	}
}

// showProgramTop answers "top" / "what's running": the
// programs this chat session itself started — honestly
// scoped in the header, because it is NOT an OS process
// table. State is reaped at render time: a program that died
// since it was last mentioned is listed as exited, with
// when, never as live.
func showProgramTop() {
	tag := cTag("gollemer [runner]>")
	p := sessionProgram
	if p == nil {
		fmt.Printf("%s No programs are running right now — say 'run the program <file>' to start one.\n", tag)
		recordBrain("runner", true, "top asked with nothing running")
		return
	}
	lines, total := p.log.snapshot()
	visits := 0
	for _, l := range lines {
		if strings.HasPrefix(l, "visit:") {
			visits++
		}
	}
	p.mu.Lock()
	exited, exitErr, exitedAt := p.exited, p.exitErr, p.exitedAt
	p.mu.Unlock()
	fmt.Printf("%s Programs this chat is running — only the ones it started, not everything on this computer:\n", tag)
	if !exited {
		entry := fmt.Sprintf("%s — running for %s", p.rel, plainDuration(time.Since(p.startedAt)))
		if u := p.confirmedURL(); u != "" {
			entry += ", serving at " + u
		}
		fmt.Printf("%s %s; %d lines logged, %s. Say \"stop the program\" to stop it.\n",
			tag, entry, total, pluralCount(int64(visits), "visit"))
		recordBrain("runner", true, "top listed a running program: "+p.rel)
		return
	}
	status := "finished (exit code 0)"
	if exitErr != nil {
		status = fmt.Sprintf("stopped (%v)", exitErr)
	}
	fmt.Printf("%s %s — %s %s ago; it ran for %s and printed %d lines (%s). It's no longer running.\n",
		tag, p.rel, status, plainDuration(time.Since(exitedAt)),
		plainDuration(exitedAt.Sub(p.startedAt)), total, pluralCount(int64(visits), "visit"))
	recordBrain("runner", true, "top listed an exited program: "+p.rel)
}

// noteProgramActivity rides the unified loop's prompt cycle:
// at most one short line, only when something actually
// changed — new output lines the user hasn't been told
// about, or an exit not yet surfaced. It never speaks twice
// about the same fact.
func noteProgramActivity() {
	p := sessionProgram
	if p == nil {
		return
	}
	_, total := p.log.snapshot()
	if p.hasExited() {
		p.mu.Lock()
		told := p.exitTold
		err := p.exitErr
		if !told {
			p.exitTold = true
		}
		p.mu.Unlock()
		if told {
			return
		}
		p.log.mu.Lock()
		p.log.notified = total
		p.log.mu.Unlock()
		if err == nil {
			fmt.Printf("[%s finished (exit code 0) — say \"show the program log\" to read its output]\n", p.rel)
		} else {
			fmt.Printf("[%s stopped (%v) — say \"show the program log\" to read its output]\n", p.rel, err)
		}
		return
	}
	p.log.mu.Lock()
	fresh := total - p.log.notified
	if fresh > 0 {
		p.log.notified = total
	}
	p.log.mu.Unlock()
	if fresh > 0 {
		fmt.Printf("[%s printed %d new lines — say \"show the program log\" to read them]\n", p.rel, fresh)
	}
}
