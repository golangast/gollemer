package main

import (
	"bufio"
	"bytes"
	"fmt"
	"testing"
)

func TestReadMessage(t *testing.T) {
	jsonPayload := `{"jsonrpc":"2.0","id":1,"method":"initialize"}`
	framedMessage := fmt.Sprintf("Content-Length: %d\r\n\r\n%s", len(jsonPayload), jsonPayload)

	reader := bufio.NewReader(bytes.NewBufferString(framedMessage))
	req, err := readMessage(reader)
	if err != nil {
		t.Fatalf("readMessage failed: %v", err)
	}

	if req.Method != "initialize" {
		t.Errorf("got method %q, want %q", req.Method, "initialize")
	}
}
