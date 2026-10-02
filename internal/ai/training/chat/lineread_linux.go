//go:build linux

package chat

import (
	"os"
	"syscall"
	"time"
	"unsafe"
)

const rawSupported = true

// keyCode identifies one logical key read from the terminal.
type keyCode int

const (
	keyEOF keyCode = iota
	keyChar
	keyEnter
	keyTab
	keyBackspace
	keyDelete
	keyLeft
	keyRight
	keyUp
	keyDown
	keyHome
	keyEnd
	keyCtrlA
	keyCtrlC
	keyCtrlD
	keyCtrlE
	keyCtrlK
	keyCtrlL
	keyCtrlU
	keyCtrlW
	keyIgnore
)

// termios mirrors Linux struct termios on amd64/arm64.
type termios struct {
	Iflag  uint32
	Oflag  uint32
	Cflag  uint32
	Lflag  uint32
	Line   uint8
	Cc     [32]uint8
	Ispeed uint32
	Ospeed uint32
}

const (
	tcgets     = 0x5401
	tcsets     = 0x5402
	tiocgwinsz = 0x5413

	ignbrk = 0x0001
	brkint = 0x0002
	istrip = 0x0020
	inlcr  = 0x0040
	igncr  = 0x0080
	icrnl  = 0x0100
	ixon   = 0x0400

	opost = 0x0001

	isig   = 0x0001
	icanon = 0x0002
	echo   = 0x0008
	echonl = 0x0040
	iexten = 0x8000

	csize  = 0x0030
	cs8    = 0x0030
	parenb = 0x0100

	vmin  = 6
	vtime = 5
)

func tcgetattr(f *os.File) (*termios, error) {
	var t termios
	_, _, errno := syscall.Syscall(syscall.SYS_IOCTL, f.Fd(), tcgets, uintptr(unsafe.Pointer(&t)))
	if errno != 0 {
		return nil, errno
	}
	return &t, nil
}

func tcsetattr(f *os.File, t *termios) error {
	_, _, errno := syscall.Syscall(syscall.SYS_IOCTL, f.Fd(), tcsets, uintptr(unsafe.Pointer(&t)))
	if errno != 0 {
		return errno
	}
	return nil
}

// enableRawMode switches the terminal to raw mode (no echo, no line
// buffering, no signal chars) and returns a function restoring the
// original mode. ok is false when the fd isn't a terminal.
func enableRawMode(f *os.File) (func(), bool) {
	orig, err := tcgetattr(f)
	if err != nil {
		return nil, false
	}
	raw := *orig
	raw.Iflag &^= ignbrk | brkint | istrip | inlcr | igncr | icrnl | ixon
	raw.Oflag &^= opost
	raw.Lflag &^= echo | echonl | icanon | isig | iexten
	raw.Cflag &^= csize | parenb
	raw.Cflag |= cs8
	raw.Cc[vmin] = 1
	raw.Cc[vtime] = 0
	if err := tcsetattr(f, &raw); err != nil {
		return nil, false
	}
	return func() { tcsetattr(f, orig) }, true
}

// termWidth returns the terminal width in columns, defaulting to 80.
func termWidth(f *os.File) int {
	var ws [4]uint16
	_, _, errno := syscall.Syscall(syscall.SYS_IOCTL, f.Fd(), tiocgwinsz, uintptr(unsafe.Pointer(&ws)))
	if errno != 0 || ws[1] == 0 {
		return 80
	}
	return int(ws[1])
}

// readKey reads one logical key: a rune, a control key, or an escape
// sequence (arrows, home/end, delete). Multi-byte UTF-8 is decoded.
func readKey(f *os.File) (keyCode, rune) {
	var b [1]byte
	if _, err := f.Read(b[:]); err != nil {
		return keyEOF, 0
	}
	c := b[0]
	switch c {
	case '\r', '\n':
		return keyEnter, 0
	case '\t':
		return keyTab, 0
	case 127, 8:
		return keyBackspace, 0
	case 1:
		return keyCtrlA, 0
	case 3:
		return keyCtrlC, 0
	case 4:
		return keyCtrlD, 0
	case 5:
		return keyCtrlE, 0
	case 11:
		return keyCtrlK, 0
	case 12:
		return keyCtrlL, 0
	case 21:
		return keyCtrlU, 0
	case 23:
		return keyCtrlW, 0
	case 27:
		return readEscape(f)
	case ' ':
		return keyChar, ' '
	}
	if c < 32 {
		return keyIgnore, 0
	}
	if c&0x80 == 0 {
		return keyChar, rune(c)
	}
	var n int
	var r rune
	switch {
	case c&0xE0 == 0xC0:
		n, r = 1, rune(c&0x1F)
	case c&0xF0 == 0xE0:
		n, r = 2, rune(c&0x0F)
	case c&0xF8 == 0xF0:
		n, r = 3, rune(c&0x07)
	default:
		return keyIgnore, 0
	}
	var cont [3]byte
	for i := 0; i < n; i++ {
		if _, err := f.Read(cont[i : i+1]); err != nil {
			return keyEOF, 0
		}
		if cont[i]&0xC0 != 0x80 {
			return keyIgnore, 0
		}
		r = r<<6 | rune(cont[i]&0x3F)
	}
	return keyChar, r
}

// readEscape parses an escape sequence. A lone ESC (nothing follows
// within 80ms) is ignored.
func readEscape(f *os.File) (keyCode, rune) {
	f.SetReadDeadline(time.Now().Add(80 * time.Millisecond))
	var b [1]byte
	n, err := f.Read(b[:])
	f.SetReadDeadline(time.Time{})
	if err != nil || n == 0 {
		return keyIgnore, 0
	}
	switch b[0] {
	case '[':
		return readCSI(f)
	case 'O': // application cursor keys: ESC O A/B/C/D/H/F
		var c [1]byte
		if _, err := f.Read(c[:]); err != nil {
			return keyIgnore, 0
		}
		switch c[0] {
		case 'A':
			return keyUp, 0
		case 'B':
			return keyDown, 0
		case 'C':
			return keyRight, 0
		case 'D':
			return keyLeft, 0
		case 'H':
			return keyHome, 0
		case 'F':
			return keyEnd, 0
		}
	}
	return keyIgnore, 0
}

func readCSI(f *os.File) (keyCode, rune) {
	var b [1]byte
	if _, err := f.Read(b[:]); err != nil {
		return keyIgnore, 0
	}
	switch b[0] {
	case 'A':
		return keyUp, 0
	case 'B':
		return keyDown, 0
	case 'C':
		return keyRight, 0
	case 'D':
		return keyLeft, 0
	case 'H':
		return keyHome, 0
	case 'F':
		return keyEnd, 0
	case '3': // ESC [ 3 ~ — delete
		if _, err := f.Read(b[:]); err != nil || b[0] != '~' {
			return keyIgnore, 0
		}
		return keyDelete, 0
	case '1', '7': // ESC [ 1 ~ — home
		f.Read(b[:])
		return keyHome, 0
	case '4', '8': // ESC [ 4 ~ — end
		f.Read(b[:])
		return keyEnd, 0
	}
	return keyIgnore, 0
}
