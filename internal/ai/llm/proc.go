package llm

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"syscall"
)

// SavePid writes the process ID to a file.
func SavePid(pid int, pidFile string) error {
	return os.WriteFile(pidFile, []byte(strconv.Itoa(pid)), 0644)
}

// ReadPid reads the process ID from a file.
func ReadPid(pidFile string) (int, error) {
	data, err := os.ReadFile(pidFile)
	if err != nil {
		return 0, err
	}
	pid, err := strconv.Atoi(string(data))
	if err != nil {
		return 0, fmt.Errorf("invalid pid in file: %w", err)
	}
	return pid, nil
}

// StartWebserver runs the go program at sourcePath and saves its PID to pidFile.
func StartWebserver(sourcePath, pidFile string) error {
	// Check if source file exists
	if _, err := os.Stat(sourcePath); os.IsNotExist(err) {
		return fmt.Errorf("webserver source not found at: %s", sourcePath)
	}

	cmd := exec.Command("go", "run", sourcePath)
	// Redirect output to stdout/stderr so you can see it in the terminal
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start webserver: %w", err)
	}

	if err := SavePid(cmd.Process.Pid, pidFile); err != nil {
		// Try to kill it if we can't save the PID
		_ = cmd.Process.Kill()
		return fmt.Errorf("failed to save PID: %w", err)
	}

	fmt.Printf("Webserver started with PID %d\n", cmd.Process.Pid)
	return nil
}

// StopWebserver reads the PID from the file and sends a SIGTERM signal.
func StopWebserver(pidFile string) error {
	pid, err := ReadPid(pidFile)
	if err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("pid file not found (is the server running?)")
		}
		return err
	}

	process, err := os.FindProcess(pid)
	if err != nil {
		return err
	}

	// Send SIGTERM (Graceful shutdown)
	if err := process.Signal(syscall.SIGTERM); err != nil {
		// If process is already dead, just remove the file
		if errors.Is(err, os.ErrProcessDone) {
			_ = os.Remove(pidFile)
			return nil
		}
		return err
	}

	// Clean up PID file
	return os.Remove(pidFile)
}
