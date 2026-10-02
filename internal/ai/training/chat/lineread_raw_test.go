//go:build linux

package chat

import (
	"os"
	"syscall"
	"testing"
	"unsafe"
)

// openPty opens a fresh pseudoterminal pair and returns the slave side,
// which behaves like a real interactive terminal for termios ioctls.
func openPty(t *testing.T) *os.File {
	t.Helper()
	master, err := os.OpenFile("/dev/ptmx", syscall.O_RDWR|syscall.O_NOCTTY, 0)
	if err != nil {
		t.Skipf("no ptmx available: %v", err)
	}
	const (
		tiocgptn   = 0x80045430
		tiocsptlck = 0x40045431
	)
	var zero int32
	if _, _, errno := syscall.Syscall(syscall.SYS_IOCTL, master.Fd(), tiocsptlck, uintptr(unsafe.Pointer(&zero))); errno != 0 {
		master.Close()
		t.Skipf("unlockpt failed: %v", errno)
	}
	var n uint32
	if _, _, errno := syscall.Syscall(syscall.SYS_IOCTL, master.Fd(), tiocgptn, uintptr(unsafe.Pointer(&n))); errno != 0 {
		master.Close()
		t.Skipf("tiocgptn failed: %v", errno)
	}
	slave, err := os.OpenFile("/dev/pts/"+itoa(n), syscall.O_RDWR|syscall.O_NOCTTY, 0)
	if err != nil {
		master.Close()
		t.Skipf("no pts slave: %v", err)
	}
	t.Cleanup(func() { slave.Close(); master.Close() })
	return slave
}

func itoa(n uint32) string {
	if n == 0 {
		return "0"
	}
	var b [10]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	return string(b[i:])
}

// TestRawModeRoundTrip enables raw mode on a real pty and verifies the
// kernel got exactly the flags we asked for — not stack garbage. This
// caught a bug where tcsetattr passed &t (pointer-to-pointer) instead
// of t, writing random flags: the prompt came out UPPERCASE (OLCUC set
// from garbage) and reads failed instantly (baud clobbered to B0).
func TestRawModeRoundTrip(t *testing.T) {
	slave := openPty(t)

	orig, err := tcgetattr(slave)
	if err != nil {
		t.Fatalf("tcgetattr: %v", err)
	}
	if orig.Lflag&icanon == 0 {
		t.Fatalf("fresh pty should start canonical")
	}

	restore, ok := enableRawMode(slave)
	if !ok {
		t.Fatalf("enableRawMode failed on a pty")
	}
	raw, err := tcgetattr(slave)
	if err != nil {
		t.Fatalf("tcgetattr after enableRawMode: %v", err)
	}
	if raw.Lflag&icanon != 0 {
		t.Errorf("ICANON still on after enableRawMode")
	}
	if raw.Lflag&echo != 0 {
		t.Errorf("ECHO still on after enableRawMode")
	}
	if raw.Lflag&isig != 0 {
		t.Errorf("ISIG still on after enableRawMode")
	}
	if raw.Oflag&opost != 0 {
		t.Errorf("OPOST still on after enableRawMode")
	}
	const olcuc = 0x0002
	if raw.Oflag&olcuc != 0 {
		t.Errorf("OLCUC unexpectedly on (garbage flags written?)")
	}
	if raw.Cc[vmin] != 1 || raw.Cc[vtime] != 0 {
		t.Errorf("VMIN/VTIME = %d/%d, want 1/0", raw.Cc[vmin], raw.Cc[vtime])
	}

	restore()
	back, err := tcgetattr(slave)
	if err != nil {
		t.Fatalf("tcgetattr after restore: %v", err)
	}
	if back.Lflag != orig.Lflag || back.Iflag != orig.Iflag || back.Oflag != orig.Oflag {
		t.Errorf("restore did not bring back original flags: got lflag=%x iflag=%x oflag=%x, want %x %x %x",
			back.Lflag, back.Iflag, back.Oflag, orig.Lflag, orig.Iflag, orig.Oflag)
	}
}
