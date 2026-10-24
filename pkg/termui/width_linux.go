//go:build linux

package termui

import (
	"os"
	"syscall"
	"unsafe"
)

// terminalWidth reads the stdout terminal size via TIOCGWINSZ,
// the same approach as the chat line reader. 0 means unknown.
func terminalWidth() int {
	var ws [4]uint16
	_, _, errno := syscall.Syscall(syscall.SYS_IOCTL, os.Stdout.Fd(), syscall.TIOCGWINSZ,
		uintptr(unsafe.Pointer(&ws)))
	if errno != 0 || ws[1] == 0 {
		return 0
	}
	return int(ws[1])
}
