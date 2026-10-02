package main

import (
	"bufio"
	"fmt"
	"os"
	"runtime"
	"strings"
	"syscall"
	"unsafe"
)

const (
	tiocgeta         = 0x40487413
	tiocseta         = 0x80487414
	tcgets           = 0x5401
	tcsets           = 0x5402
	tiocgwinsz       = 0x5413
	tiocgwinszDarwin = 0x40087468
)

type winsize struct {
	Row, Col, Xpixel, Ypixel uint16
}

type Choice struct {
	Command string
	Comment string
	Raw     string
}

func getTerminalSize(fd uintptr) (width, height int) {
	var ws winsize
	var req uintptr
	if runtime.GOOS == "darwin" {
		req = tiocgwinszDarwin
	} else {
		req = tiocgwinsz
	}

	_, _, err := syscall.Syscall(syscall.SYS_IOCTL, fd, req, uintptr(unsafe.Pointer(&ws)))
	if err != 0 || ws.Row == 0 || ws.Col == 0 {
		return 80, 25
	}
	return int(ws.Col), int(ws.Row)
}

func enableRawMode(fd uintptr) (*syscall.Termios, error) {
	var oldState syscall.Termios
	var reqGet uintptr

	if runtime.GOOS == "darwin" {
		reqGet = tiocgeta
	} else {
		reqGet = tcgets
	}

	if _, _, err := syscall.Syscall(syscall.SYS_IOCTL, fd, reqGet, uintptr(unsafe.Pointer(&oldState))); err != 0 {
		return nil, err
	}

	newState := oldState
	newState.Iflag &^= syscall.IGNBRK | syscall.BRKINT | syscall.PARMRK | syscall.ISTRIP | syscall.INLCR | syscall.IGNCR | syscall.ICRNL | syscall.IXON
	newState.Oflag &^= syscall.OPOST
	newState.Lflag &^= syscall.ECHO | syscall.ECHONL | syscall.ICANON | syscall.ISIG | syscall.IEXTEN
	newState.Cflag &^= syscall.CSIZE | syscall.PARENB
	newState.Cflag |= syscall.CS8

	var reqSet uintptr
	if runtime.GOOS == "darwin" {
		reqSet = tiocseta
	} else {
		reqSet = tcsets
	}

	if _, _, err := syscall.Syscall(syscall.SYS_IOCTL, fd, reqSet, uintptr(unsafe.Pointer(&newState))); err != 0 {
		return nil, err
	}
	return &oldState, nil
}

func disableRawMode(fd uintptr, oldState *syscall.Termios) {
	if oldState != nil {
		var reqSet uintptr
		if runtime.GOOS == "darwin" {
			reqSet = tiocseta
		} else {
			reqSet = tcsets
		}
		_, _, _ = syscall.Syscall(syscall.SYS_IOCTL, fd, reqSet, uintptr(unsafe.Pointer(oldState)))
	}
}

func fuzzyMatch(target, query string) (bool, int) {
	if query == "" {
		return true, 0
	}
	targetLower := strings.ToLower(target)
	queryLower := strings.ToLower(query)

	tIdx, score := 0, 0
	for qIdx := 0; qIdx < len(queryLower); qIdx++ {
		found := false
		for ; tIdx < len(targetLower); tIdx++ {
			if targetLower[tIdx] == queryLower[qIdx] {
				score += 10 - tIdx
				tIdx++
				found = true
				break
			}
		}
		if !found {
			return false, 0
		}
	}
	return true, score
}

func parseChoice(line string) Choice {
	parts := strings.Fields(line)
	if len(parts) == 0 {
		return Choice{Raw: line}
	}
	cmd := parts[0]
	comment := ""
	if len(parts) > 1 {
		comment = strings.Join(parts[1:], " ")
	}
	return Choice{
		Command: cmd,
		Comment: comment,
		Raw:     line,
	}
}

// picker holds the picker's interactive state.
type picker struct {
	choices  []Choice
	filtered []Choice
	query    string
	sel      int
}

// refilter rebuilds the filtered list from the current query.
func (p *picker) refilter() {
	if p.query == "" {
		p.filtered = make([]Choice, len(p.choices))
		copy(p.filtered, p.choices)
		p.sel = 0
		return
	}

	type match struct {
		choice Choice
		score  int
	}
	var matches []match

	for _, choice := range p.choices {
		if ok, score := fuzzyMatch(choice.Raw, p.query); ok {
			matches = append(matches, match{choice: choice, score: score})
		}
	}

	p.filtered = nil
	for _, m := range matches {
		p.filtered = append(p.filtered, m.choice)
	}

	p.sel = 0
}

func (p *picker) moveUp(rows int) {
	if p.sel%rows > 0 {
		p.sel--
	}
}

func (p *picker) moveDown(rows, total int) {
	if p.sel%rows < rows-1 && p.sel+1 < total {
		p.sel++
	}
}

func (p *picker) moveRight(rows, total int) {
	if p.sel+rows < total {
		p.sel += rows
	}
}

func (p *picker) moveLeft(rows int) {
	if p.sel-rows >= 0 {
		p.sel -= rows
	}
}

// Actions returned by processKeys.
const (
	keyNone = ""
	keyQuit = "quit" // Ctrl+C: leave without choosing
	keyDone = "done" // Enter: accept the current selection
)

// processKeys consumes complete key events from the front of pending and
// returns the number of bytes consumed plus an action for the caller.
// Every byte of a burst is processed; a trailing partial escape sequence
// is left unconsumed so the next read can complete it.
func (p *picker) processKeys(pending []byte, rows, total int) (int, string) {
	i := 0
	for i < len(pending) {
		b := pending[i]
		switch {
		case b == 3: // Ctrl+C
			return i + 1, keyQuit

		case b == 13: // Enter
			return i + 1, keyDone

		case b == 127 || b == 8: // Backspace
			if len(p.query) > 0 {
				p.query = p.query[:len(p.query)-1]
				p.refilter()
			}
			i++

		case b == 14 || b == 10: // Ctrl+N / Ctrl+J
			p.moveDown(rows, total)
			i++

		case b == 16 || b == 11: // Ctrl+P / Ctrl+K
			p.moveUp(rows)
			i++

		case b == 27: // Escape: CSI sequences arrive as ESC [ X
			if len(pending)-i < 3 {
				// Sequence split across reads: wait for the rest.
				return i, keyNone
			}
			if pending[i+1] != '[' {
				// Lone Escape (or Alt+key): drop the ESC, keep the rest.
				i++
				continue
			}
			skip := 3
			switch pending[i+2] {
			case 'A': // Up arrow
				p.moveUp(rows)
			case 'B': // Down arrow
				p.moveDown(rows, total)
			case 'C': // Right arrow
				p.moveRight(rows, total)
			case 'D': // Left arrow
				p.moveLeft(rows)
			case 'M': // Mouse report: ESC [ M + 3 bytes; swallow all 6
				// so click coordinates never leak into the query.
				skip = 6
			}
			if len(pending)-i < skip {
				// Sequence split across reads: wait for the rest.
				return i, keyNone
			}
			i += skip

		default:
			if b >= 32 && b <= 126 {
				p.query += string(b)
				p.refilter()
			}
			i++
		}
	}
	return i, keyNone
}

func main() {
	var choices []Choice
	stat, _ := os.Stdin.Stat()
	if (stat.Mode() & os.ModeCharDevice) == 0 {
		scanner := bufio.NewScanner(os.Stdin)
		for scanner.Scan() {
			if text := scanner.Text(); text != "" {
				choices = append(choices, parseChoice(text))
			}
		}
	}

	if len(choices) == 0 {
		fmt.Fprintln(os.Stderr, "No targets received.")
		os.Exit(1)
	}

	tty, err := os.OpenFile("/dev/tty", os.O_RDWR, 0)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to open TTY: %v\n", err)
		os.Exit(1)
	}
	defer tty.Close()

	fd := tty.Fd()
	oldState, err := enableRawMode(fd)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to set raw mode: %v\n", err)
		os.Exit(1)
	}
	defer disableRawMode(fd, oldState)

	tty.WriteString("\x1b[?1049h\x1b[?25l\x1b[?1000h")
	defer tty.WriteString("\x1b[?1049l\x1b[?25h\x1b[?1000l")

	p := &picker{choices: choices}
	p.refilter()

	draw := func() {
		width, _ := getTerminalSize(fd)

		colWidth := 24
		cols := width / colWidth
		if cols <= 0 {
			cols = 1
		}

		var sb strings.Builder
		sb.WriteString("\x1b[H\x1b[J")

		// Search Input Box
		sb.WriteString(fmt.Sprintf("\x1b[1;36m[ > %s ]\x1b[0m\r\n\r\n", p.query))

		if len(p.filtered) > 0 {
			rows := (len(p.filtered) + cols - 1) / cols

			for row := 0; row < rows; row++ {
				for col := 0; col < cols; col++ {
					i := col*rows + row
					if i >= len(p.filtered) {
						break
					}

					item := p.filtered[i]
					label := item.Command
					if len(label) > 16 {
						label = label[:16]
					}

					if i == p.sel {
						sb.WriteString(fmt.Sprintf("[ \x1b[37;45;1m%-16s\x1b[0m ] ", label))
					} else {
						sb.WriteString(fmt.Sprintf("[ \x1b[36m%-16s\x1b[0m ] ", label))
					}
				}
				sb.WriteString("\r\n")
			}

			if p.sel < len(p.filtered) && p.filtered[p.sel].Comment != "" {
				sb.WriteString(fmt.Sprintf("\r\n\x1b[90m> %s\x1b[0m", p.filtered[p.sel].Comment))
			}
		}

		tty.WriteString(sb.String())
	}

	buf := make([]byte, 64)
	var pending []byte

	for {
		draw()
		n, err := tty.Read(buf)
		if err != nil || n == 0 {
			break
		}
		pending = append(pending, buf[:n]...)

		width, _ := getTerminalSize(fd)
		cols := width / 24
		if cols <= 0 {
			cols = 1
		}

		total := len(p.filtered)
		rows := 1
		if total > 0 {
			rows = (total + cols - 1) / cols
		}

		consumed, action := p.processKeys(pending, rows, total)
		pending = pending[consumed:]

		switch action {
		case keyQuit:
			return

		case keyDone:
			var selectedResult Choice
			if len(p.filtered) > 0 && p.sel < len(p.filtered) {
				selectedResult = p.filtered[p.sel]
			}
			disableRawMode(fd, oldState)
			tty.WriteString("\x1b[?1049l\x1b[?25h\x1b[?1000l")
			if selectedResult.Command != "" {
				fmt.Println(selectedResult.Command)
			}
			return
		}
	}
}
