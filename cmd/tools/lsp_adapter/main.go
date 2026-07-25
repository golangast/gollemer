package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"strconv"
	"strings"
)

type Request struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      interface{}     `json:"id,omitempty"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

type Response struct {
	JSONRPC string      `json:"jsonrpc"`
	ID      interface{} `json:"id,omitempty"`
	Result  interface{} `json:"result,omitempty"`
	Error   interface{} `json:"error,omitempty"`
}

type Notification struct {
	JSONRPC string      `json:"jsonrpc"`
	Method  string      `json:"method"`
	Params  interface{} `json:"params"`
}

type TextDocumentItem struct {
	URI  string `json:"uri"`
	Text string `json:"text"`
}

type TextDocumentIdentifier struct {
	URI string `json:"uri"`
}

type DidOpenTextDocumentParams struct {
	TextDocument TextDocumentItem `json:"textDocument"`
}

type DidSaveTextDocumentParams struct {
	TextDocument TextDocumentIdentifier `json:"textDocument"`
}

type CodeActionParams struct {
	TextDocument TextDocumentIdentifier `json:"textDocument"`
}

type ExecuteCommandParams struct {
	Command   string        `json:"command"`
	Arguments []interface{} `json:"arguments,omitempty"`
}

func main() {
	reader := bufio.NewReader(os.Stdin)
	for {
		req, err := readMessage(reader)
		if err != nil {
			if err == io.EOF {
				break
			}
			log.Printf("LSP Reader error: %v", err)
			continue
		}

		handleMessage(req)
	}
}

func readMessage(r *bufio.Reader) (*Request, error) {
	var contentLength int
	for {
		line, err := r.ReadString('\n')
		if err != nil {
			return nil, err
		}
		line = strings.TrimSpace(line)
		if line == "" {
			break
		}
		if strings.HasPrefix(line, "Content-Length:") {
			lenStr := strings.TrimSpace(strings.TrimPrefix(line, "Content-Length:"))
			contentLength, _ = strconv.Atoi(lenStr)
		}
	}

	if contentLength <= 0 {
		return nil, fmt.Errorf("invalid content length")
	}

	body := make([]byte, contentLength)
	if _, err := io.ReadFull(r, body); err != nil {
		return nil, err
	}

	var req Request
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, err
	}

	return &req, nil
}

func writeMessage(msg interface{}) {
	body, err := json.Marshal(msg)
	if err != nil {
		return
	}
	header := fmt.Sprintf("Content-Length: %d\r\n\r\n", len(body))
	os.Stdout.WriteString(header)
	os.Stdout.Write(body)
}

func handleMessage(req *Request) {
	switch req.Method {
	case "initialize":
		res := map[string]interface{}{
			"capabilities": map[string]interface{}{
				"textDocumentSync": 1, // Full sync
				"codeActionProvider": map[string]interface{}{
					"codeActionKinds": []string{"quickfix", "refactor"},
				},
				"executeCommandProvider": map[string]interface{}{
					"commands": []string{"gollemer.patch", "gollemer.test"},
				},
			},
		}
		writeMessage(Response{JSONRPC: "2.0", ID: req.ID, Result: res})

	case "initialized":
		// No-op notification

	case "textDocument/didOpen":
		var params DidOpenTextDocumentParams
		if err := json.Unmarshal(req.Params, &params); err == nil {
			publishDiagnostics(params.TextDocument.URI)
		}

	case "textDocument/didSave":
		var params DidSaveTextDocumentParams
		if err := json.Unmarshal(req.Params, &params); err == nil {
			publishDiagnostics(params.TextDocument.URI)
		}

	case "textDocument/codeAction":
		var params CodeActionParams
		_ = json.Unmarshal(req.Params, &params)

		actions := []map[string]interface{}{
			{
				"title": "⚡ Gollemer: Apply AI Patch & Self-Heal",
				"kind":  "quickfix",
				"command": map[string]interface{}{
					"title":     "Apply AI Patch",
					"command":   "gollemer.patch",
					"arguments": []interface{}{params.TextDocument.URI, "fix code"},
				},
			},
			{
				"title": "🧪 Gollemer: Scaffold Unit Tests",
				"kind":  "refactor",
				"command": map[string]interface{}{
					"title":     "Scaffold Unit Tests",
					"command":   "gollemer.test",
					"arguments": []interface{}{params.TextDocument.URI, "add unit test"},
				},
			},
		}
		writeMessage(Response{JSONRPC: "2.0", ID: req.ID, Result: actions})

	case "workspace/executeCommand":
		var params ExecuteCommandParams
		if err := json.Unmarshal(req.Params, &params); err == nil {
			targetURI := ""
			promptStr := "fix code"
			if len(params.Arguments) > 0 {
				if uri, ok := params.Arguments[0].(string); ok {
					targetURI = uri
				}
			}
			if len(params.Arguments) > 1 {
				if p, ok := params.Arguments[1].(string); ok {
					promptStr = p
				}
			}

			targetFile := strings.TrimPrefix(targetURI, "file://")
			if targetFile != "" {
				subcmd := "patch"
				if params.Command == "gollemer.test" {
					subcmd = "test"
				}
				_ = exec.Command("./bin/gollemer", subcmd, promptStr, "-target="+targetFile).Run()
				publishDiagnostics(targetURI)
			}
		}
		writeMessage(Response{JSONRPC: "2.0", ID: req.ID, Result: "OK"})

	case "shutdown":
		writeMessage(Response{JSONRPC: "2.0", ID: req.ID, Result: nil})

	case "exit":
		os.Exit(0)

	default:
		if req.ID != nil {
			writeMessage(Response{JSONRPC: "2.0", ID: req.ID, Result: nil})
		}
	}
}

func publishDiagnostics(uri string) {
	diags := []map[string]interface{}{}
	notif := Notification{
		JSONRPC: "2.0",
		Method:  "textDocument/publishDiagnostics",
		Params: map[string]interface{}{
			"uri":         uri,
			"diagnostics": diags,
		},
	}
	writeMessage(notif)
}
