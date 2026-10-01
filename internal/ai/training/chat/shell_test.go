package chat

import (
	"errors"
	"strings"
	"testing"
)

func TestHandleShellCommand(t *testing.T) {
	old := runShellOnce
	defer func() { runShellOnce = old }()
	runShellOnce = func(prompt string) (string, error) {
		if prompt != "build a list of squares" {
			t.Errorf("prompt = %q", prompt)
		}
		return "=== GENERATED GO SOURCE ===\npackage main\n", nil
	}
	out, ok := handleShellCommand("/shell build a list of squares")
	if !ok {
		t.Fatal("want ok=true for /shell")
	}
	if !strings.Contains(out, "package main") {
		t.Errorf("out = %q", out)
	}
}

func TestHandleShellCommandUsage(t *testing.T) {
	out, ok := handleShellCommand("/shell")
	if !ok {
		t.Fatal("want ok=true for bare /shell")
	}
	if !strings.Contains(out, "usage:") {
		t.Errorf("want usage, got %q", out)
	}
}

func TestHandleShellCommandError(t *testing.T) {
	old := runShellOnce
	defer func() { runShellOnce = old }()
	runShellOnce = func(prompt string) (string, error) {
		return "", errors.New("boom")
	}
	out, ok := handleShellCommand("/shell frobnicate")
	if !ok {
		t.Fatal("want ok=true")
	}
	if !strings.Contains(out, "shell error: boom") {
		t.Errorf("out = %q", out)
	}
}

func TestHandleShellCommandNotShell(t *testing.T) {
	if _, ok := handleShellCommand("/shelly"); ok {
		t.Error("want ok=false for /shelly")
	}
	if _, ok := handleShellCommand("hello"); ok {
		t.Error("want ok=false for plain text")
	}
}
