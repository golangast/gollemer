package chat

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// errCancelled reports a line cancelled with Ctrl-C: the chat shows a
// fresh prompt instead of processing input.
var errCancelled = errors.New("chat: line cancelled")

// LineReader reads interactive lines with shell-style editing — history
// on up/down, left/right cursor movement, tab completion — in pure Go
// with no dependencies. On a Linux terminal it drives stdin in raw
// mode; anywhere else (pipes, tests, other platforms) it falls back to
// a plain line scanner.
type LineReader struct {
	in      io.Reader
	term    *os.File // non-nil when in is an interactive terminal
	scanner *bufio.Scanner

	history  []string
	histPos  int // -1 = editing the current line, else index from the end
	saved    []rune
	histPath string
}

// NewLineReader builds a reader over in (usually os.Stdin), loading the
// persisted history when in is an interactive terminal.
func NewLineReader(in io.Reader) *LineReader {
	r := &LineReader{in: in, histPos: -1}
	if f, ok := in.(*os.File); ok && rawSupported && isCharDevice(f) {
		r.term = f
		r.histPath = gollemerHistoryPath()
		r.history = loadHistory(r.histPath, 500)
	}
	r.scanner = bufio.NewScanner(in)
	r.scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
	return r
}

func isCharDevice(f *os.File) bool {
	fi, err := f.Stat()
	return err == nil && fi.Mode()&os.ModeCharDevice != 0
}

func gollemerHistoryPath() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".gollemer_history")
}

func loadHistory(path string, max int) []string {
	if path == "" {
		return nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var out []string
	for _, l := range strings.Split(string(data), "\n") {
		if strings.TrimSpace(l) != "" {
			out = append(out, l)
		}
	}
	if len(out) > max {
		out = out[len(out)-max:]
	}
	return out
}

// ReadLine prints prompt and reads one edited line. It returns io.EOF
// when input ends (Ctrl-D on an empty line) and errCancelled when the
// line is cancelled with Ctrl-C.
func (r *LineReader) ReadLine(prompt string) (string, error) {
	if r.term == nil {
		fmt.Print(prompt)
		if !r.scanner.Scan() {
			return "", io.EOF
		}
		return r.scanner.Text(), nil
	}
	return r.readRaw(prompt)
}

// addHistory remembers a submitted line: in-memory for up/down, plus an
// append to the history file so it survives restarts. Only interactive
// sessions record history — piped input never pollutes the file.
func (r *LineReader) addHistory(line string) {
	if r.term == nil || strings.TrimSpace(line) == "" {
		return
	}
	if n := len(r.history); n > 0 && r.history[n-1] == line {
		return // no consecutive duplicates
	}
	r.history = append(r.history, line)
	if len(r.history) > 500 {
		r.history = r.history[len(r.history)-500:]
	}
	r.histPos = -1
	if r.histPath != "" {
		if f, err := os.OpenFile(r.histPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600); err == nil {
			fmt.Fprintln(f, line)
			f.Close()
		}
	}
}

// readRaw is the raw-mode editing loop: one logical key at a time,
// redrawing the line after each.
func (r *LineReader) readRaw(prompt string) (string, error) {
	restore, ok := enableRawMode(r.term)
	if !ok {
		fmt.Print(prompt)
		if !r.scanner.Scan() {
			return "", io.EOF
		}
		return r.scanner.Text(), nil
	}
	defer restore()

	var buf []rune
	pos := 0
	r.histPos = -1
	r.saved = nil
	tabAgain := false
	prevRows := r.drawRaw(prompt, buf, pos, 0)

	for {
		k, ch := readKey(r.term)
		if k != keyTab {
			tabAgain = false
		}
		switch k {
		case keyEnter:
			fmt.Print("\r\n")
			line := string(buf)
			r.addHistory(line)
			return line, nil
		case keyEOF:
			fmt.Print("\r\n")
			return "", io.EOF
		case keyCtrlD:
			if len(buf) == 0 {
				fmt.Print("\r\n")
				return "", io.EOF
			}
		case keyCtrlC:
			fmt.Print("^C\r\n")
			return "", errCancelled
		case keyBackspace:
			if pos > 0 {
				buf = append(buf[:pos-1], buf[pos:]...)
				pos--
			}
		case keyDelete:
			if pos < len(buf) {
				buf = append(buf[:pos], buf[pos+1:]...)
			}
		case keyLeft:
			if pos > 0 {
				pos--
			}
		case keyRight:
			if pos < len(buf) {
				pos++
			}
		case keyHome, keyCtrlA:
			pos = 0
		case keyEnd, keyCtrlE:
			pos = len(buf)
		case keyCtrlU:
			buf = buf[pos:]
			pos = 0
		case keyCtrlK:
			buf = buf[:pos]
		case keyCtrlW:
			start := pos
			for start > 0 && buf[start-1] == ' ' {
				start--
			}
			for start > 0 && buf[start-1] != ' ' {
				start--
			}
			buf = append(buf[:start], buf[pos:]...)
			pos = start
		case keyCtrlL:
			fmt.Print("\x1b[H\x1b[2J")
			prevRows = 0
		case keyUp:
			if r.histPos < len(r.history)-1 {
				if r.histPos == -1 {
					r.saved = append([]rune(nil), buf...)
				}
				r.histPos++
				buf = []rune(r.history[len(r.history)-1-r.histPos])
				pos = len(buf)
			}
		case keyDown:
			switch {
			case r.histPos > 0:
				r.histPos--
				buf = []rune(r.history[len(r.history)-1-r.histPos])
				pos = len(buf)
			case r.histPos == 0:
				r.histPos = -1
				buf = r.saved
				pos = len(buf)
			}
		case keyTab:
			var listed bool
			buf, pos, listed = r.doComplete(prompt, buf, pos, tabAgain)
			if listed {
				prevRows = 1 // the listing ended on a fresh line
			}
			tabAgain = true
		case keyChar:
			buf = append(buf[:pos], append([]rune{ch}, buf[pos:]...)...)
			pos++
		case keyIgnore:
			// Unrecognized escape sequences and control chars: skip.
		}
		prevRows = r.drawRaw(prompt, buf, pos, prevRows)
	}
}

// drawRaw redraws prompt+buf and places the cursor at pos, handling
// lines that wrap the terminal width. prevRows is how many rows the
// last draw occupied; it returns the rows the new draw occupies.
func (r *LineReader) drawRaw(prompt string, buf []rune, pos int, prevRows int) int {
	width := termWidth(r.term)
	if width < 10 {
		width = 80
	}
	promptCells := len([]rune(prompt))
	rows := (promptCells+len(buf))/width + 1

	var sb strings.Builder
	if prevRows > 1 {
		fmt.Fprintf(&sb, "\x1b[%dA", prevRows-1)
	}
	sb.WriteString("\r\x1b[J")
	sb.WriteString(prompt)
	sb.WriteString(string(buf))

	cursorAbs := promptCells + pos
	up := rows - 1 - cursorAbs/width
	if up > 0 {
		fmt.Fprintf(&sb, "\x1b[%dA", up)
	}
	sb.WriteString("\r")
	if col := cursorAbs % width; col > 0 {
		fmt.Fprintf(&sb, "\x1b[%dC", col)
	}
	fmt.Fprint(os.Stdout, sb.String())
	return rows
}

// doComplete handles Tab: one match completes it, several extend the
// common prefix, and a second Tab lists them. It returns the new buffer
// and cursor, plus whether it printed a listing.
func (r *LineReader) doComplete(prompt string, buf []rune, pos int, again bool) ([]rune, int, bool) {
	cands, tokStart := r.complete(buf, pos)
	if len(cands) == 0 {
		return buf, pos, false
	}
	token := string(buf[tokStart:pos])
	if len(cands) == 1 {
		return r.insertCompletion(buf, pos, tokStart, cands[0]), tokStart + len([]rune(cands[0])), false
	}
	// Several matches: extend the token to their longest common prefix
	// (replacing the token outright, so case differences resolve), and
	// list them on a second Tab.
	if common := commonPrefix(cands); len(common) > len(token) {
		return r.insertCompletion(buf, pos, tokStart, common), tokStart + len([]rune(common)), false
	}
	if again {
		fmt.Print("\r\n")
		show := cands
		more := ""
		if len(show) > 30 {
			more = fmt.Sprintf("  … and %d more\n", len(show)-30)
			show = show[:30]
		}
		for _, c := range show {
			fmt.Printf("  %s\n", c)
		}
		fmt.Print(more)
		return buf, pos, true
	}
	return buf, pos, false
}

func (r *LineReader) insertCompletion(buf []rune, pos, tokStart int, ins string) []rune {
	nb := make([]rune, 0, len(buf)+len(ins))
	nb = append(nb, buf[:tokStart]...)
	nb = append(nb, []rune(ins)...)
	nb = append(nb, buf[pos:]...)
	return nb
}

func commonPrefix(ss []string) string {
	if len(ss) == 0 {
		return ""
	}
	p := ss[0]
	for _, s := range ss[1:] {
		for !strings.HasPrefix(s, p) {
			p = p[:len(p)-1]
			if p == "" {
				return ""
			}
		}
	}
	return p
}

// Chat command starters offered at the beginning of a line.
var chatVerbs = []string{
	"analyze ", "clone ", "walk me through ", "show me ",
	"what does ", "what's in package ", "where would I add ",
	"where for ", "run make ", "explain make ", "what is ",
	"write ", "how do i ",
}

var slashCmds = []string{"/flow ", "/history", "/forget", "/thoughts", "/quit"}

// complete returns completion candidates for the token ending at pos,
// plus the buffer offset where that token starts.
func (r *LineReader) complete(buf []rune, pos int) ([]string, int) {
	before := string(buf[:pos])
	low := strings.ToLower(before)
	tokStart := strings.LastIndex(before, " ") + 1

	if strings.HasPrefix(low, "/") && tokStart == 0 {
		return matchCandidates(slashCmds, before, false), 0
	}

	head := low[:tokStart]
	token := before[tokStart:]

	var pool []string
	switch {
	case strings.HasPrefix(head, "run make ") || strings.HasPrefix(head, "explain make "):
		pool = makeTargetNames()
	case strings.HasPrefix(head, "analyze ") || strings.HasPrefix(head, "clone "):
		return completePath(token), tokStart
	case hasAnyPrefix(head, "walk me through ", "show me ", "what does ", "what's in package "):
		pool = symbolNames()
	case tokStart == 0:
		pool = append(append([]string{}, chatVerbs...), slashCmds...)
		pool = append(pool, makeTargetNames()...)
		return matchCandidates(pool, before, true), 0
	default:
		return nil, tokStart
	}
	return matchCandidates(pool, token, true), tokStart
}

func matchCandidates(pool []string, prefix string, fold bool) []string {
	var out []string
	for _, c := range pool {
		p, q := c, prefix
		if fold {
			p, q = strings.ToLower(c), strings.ToLower(prefix)
		}
		if strings.HasPrefix(p, q) {
			out = append(out, c)
		}
	}
	return out
}

func hasAnyPrefix(s string, prefixes ...string) bool {
	for _, p := range prefixes {
		if strings.HasPrefix(s, p) {
			return true
		}
	}
	return false
}

// makeTargetNames lists the Makefile targets the chat can run.
func makeTargetNames() []string {
	names := make([]string, 0, len(makeTargetsAllowlist))
	for n := range makeTargetsAllowlist {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

// symbolNames lists function and type names from the last analyzed
// project, so "walk me through <tab>" completes real symbols.
func symbolNames() []string {
	p := lastAnalyzeProject
	if p == nil {
		return nil
	}
	seen := map[string]bool{}
	var out []string
	for _, pkg := range p.Packages {
		for _, fn := range pkg.Funcs {
			if fn.Name != "" && !seen[fn.Name] {
				seen[fn.Name] = true
				out = append(out, fn.Name)
			}
		}
		for _, t := range pkg.Types {
			if t.Name != "" && !seen[t.Name] {
				seen[t.Name] = true
				out = append(out, t.Name)
			}
		}
	}
	sort.Strings(out)
	return out
}

// completePath completes a file path token.
func completePath(token string) []string {
	dir, prefix := filepath.Split(token)
	readDir := dir
	if readDir == "" {
		readDir = "."
	}
	entries, err := os.ReadDir(readDir)
	if err != nil {
		return nil
	}
	var out []string
	for _, e := range entries {
		name := e.Name()
		if !strings.HasPrefix(strings.ToLower(name), strings.ToLower(prefix)) {
			continue
		}
		if strings.HasPrefix(name, ".") && !strings.HasPrefix(prefix, ".") {
			continue
		}
		c := dir + name
		if e.IsDir() {
			c += "/"
		}
		out = append(out, c)
	}
	sort.Strings(out)
	return out
}
