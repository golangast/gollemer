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

	query := ""
	selectedIndex := 0

	filtered := make([]Choice, len(choices))
	copy(filtered, choices)

	filterChoices := func() {
		if query == "" {
			filtered = make([]Choice, len(choices))
			copy(filtered, choices)
			selectedIndex = 0
			return
		}

		type match struct {
			choice Choice
			score  int
		}
		var matches []match

		for _, choice := range choices {
			if ok, score := fuzzyMatch(choice.Raw, query); ok {
				matches = append(matches, match{choice: choice, score: score})
			}
		}

		filtered = nil
		for _, m := range matches {
			filtered = append(filtered, m.choice)
		}

		selectedIndex = 0
	}

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
		sb.WriteString(fmt.Sprintf("\x1b[1;36m[ > %s ]\x1b[0m\r\n\r\n", query))

		if len(filtered) > 0 {
			rows := (len(filtered) + cols - 1) / cols

			for row := 0; row < rows; row++ {
				for col := 0; col < cols; col++ {
					i := col*rows + row
					if i >= len(filtered) {
						break
					}

					item := filtered[i]
					label := item.Command
					if len(label) > 16 {
						label = label[:16]
					}

					if i == selectedIndex {
						sb.WriteString(fmt.Sprintf("[ \x1b[37;45;1m%-16s\x1b[0m ] ", label))
					} else {
						sb.WriteString(fmt.Sprintf("[ \x1b[36m%-16s\x1b[0m ] ", label))
					}
				}
				sb.WriteString("\r\n")
			}

			if selectedIndex < len(filtered) && filtered[selectedIndex].Comment != "" {
				sb.WriteString(fmt.Sprintf("\r\n\x1b[90m> %s\x1b[0m", filtered[selectedIndex].Comment))
			}
		}

		tty.WriteString(sb.String())
	}

	buf := make([]byte, 6)
	var selectedResult Choice

	for {
		draw()
		n, err := tty.Read(buf)
		if err != nil || n == 0 {
			break
		}

		width, _ := getTerminalSize(fd)
		cols := width / 24
		if cols <= 0 {
			cols = 1
		}

		total := len(filtered)
		rows := 1
		if total > 0 {
			rows = (total + cols - 1) / cols
		}

		moveUp := func() {
			if selectedIndex%rows > 0 {
				selectedIndex--
			}
		}

		moveDown := func() {
			if selectedIndex%rows < rows-1 && selectedIndex+1 < total {
				selectedIndex++
			}
		}

		moveRight := func() {
			if selectedIndex+rows < total {
				selectedIndex += rows
			}
		}

		moveLeft := func() {
			if selectedIndex-rows >= 0 {
				selectedIndex -= rows
			}
		}

		switch {
		case buf[0] == 3: // Ctrl+C
			return

		case buf[0] == 13: // Enter
			if len(filtered) > 0 && selectedIndex < len(filtered) {
				selectedResult = filtered[selectedIndex]
			}
			disableRawMode(fd, oldState)
			tty.WriteString("\x1b[?1049l\x1b[?25h\x1b[?1000l")
			if selectedResult.Command != "" {
				fmt.Println(selectedResult.Command)
			}
			return

		case buf[0] == 127 || buf[0] == 8: // Backspace
			if len(query) > 0 {
				query = query[:len(query)-1]
				filterChoices()
			}

		case buf[0] == 14 || buf[0] == 10: // Ctrl+N / Ctrl+J
			moveDown()

		case buf[0] == 16 || buf[0] == 11: // Ctrl+P / Ctrl+K
			moveUp()

		case buf[0] == 27:
			if n >= 3 && buf[1] == '[' {
				switch buf[2] {
				case 'A': // Up Arrow (Move up in current column)
					moveUp()
				case 'B': // Down Arrow (Move down in current column)
					moveDown()
				case 'C': // Right Arrow (Jump to next column)
					moveRight()
				case 'D': // Left Arrow (Jump to previous column)
					moveLeft()
				}
			}

		default:
			if buf[0] >= 32 && buf[0] <= 126 {
				query += string(buf[0])
				filterChoices()
			}
		}
	}
}
