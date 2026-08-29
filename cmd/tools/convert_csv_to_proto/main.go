// Command convert_csv_to_proto converts the CSV training data files
// (command_examples.csv and conversations.csv) into protobuf format.
//
// Usage:
//
//	go run ./cmd/tools/convert_csv_to_proto \
//	  -commands=data/training/command_examples.csv \
//	  -conversations=data/training/trainingdata/conversations.csv \
//	  -out-commands=data/training/command_examples.pb \
//	  -out-conversations=data/training/trainingdata/conversations.pb
package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	trainingpb "github.com/golangast/gollemer/internal/ai/training/proto"
)

// systemPrompt is the standard system message used across all command examples.
const systemPrompt = "You are Gollemer, an expert Go development assistant. Explain your code modifications clearly before providing code snippets."

// generateAssistantResponse creates a natural-language explanation of the
// code modification for a given prompt and code snippet. This makes the
// training data multi-conversational by providing an assistant explanation
// alongside the code before/after states.
func generateAssistantResponse(prompt, codeAfter string) string {
	lower := strings.ToLower(strings.TrimSpace(prompt))

	// HTTP-related patterns.
	switch {
	case strings.Contains(lower, "http handlefunc"):
		return "I'll register a basic HTTP handler route for the root endpoint using `http.HandleFunc`."
	case strings.Contains(lower, "http get endpoint"):
		return "I'll add an HTTP GET endpoint handler that responds with a 200 OK status."
	case strings.Contains(lower, "http post json decoder"):
		return "I'll add an HTTP POST handler that decodes the JSON request body into a struct."
	case strings.Contains(lower, "http middleware chain"):
		return "I'll add an HTTP middleware chain that logs each request before passing it to the next handler."
	case strings.Contains(lower, "http error response"):
		return "I'll add an HTTP error response that returns the appropriate status code and message."
	case strings.Contains(lower, "http status ok"):
		return "I'll write an HTTP 200 OK status using `w.WriteHeader(http.StatusOK)`."
	case strings.Contains(lower, "http status not found"):
		return "I'll write an HTTP 404 Not Found status using `w.WriteHeader(http.StatusNotFound)`."
	case strings.Contains(lower, "http json response content type"):
		return "I'll set the `Content-Type` header to `application/json` for the HTTP response."
	}

	// Import patterns.
	if strings.HasPrefix(lower, "add import ") {
		importName := strings.TrimPrefix(lower, "add import ")
		importName = strings.TrimSuffix(importName, " statement")
		importName = strings.TrimSuffix(importName, " clause")
		return fmt.Sprintf("I'll add the `%s` import to enable %s.", importName, importDescription(importName))
	}

	// Package patterns.
	if strings.HasPrefix(lower, "add package ") {
		pkgName := strings.TrimPrefix(lower, "add package ")
		pkgName = strings.TrimSuffix(pkgName, " clause")
		pkgName = strings.TrimSuffix(pkgName, " declaration")
		return fmt.Sprintf("I'll add the `package %s` declaration.", pkgName)
	}

	// Return patterns.
	if strings.HasPrefix(lower, "add return ") {
		returnExpr := strings.TrimPrefix(lower, "add return ")
		returnExpr = strings.TrimSuffix(returnExpr, " statement")
		returnExpr = strings.TrimSuffix(returnExpr, " header")
		returnExpr = strings.TrimSuffix(returnExpr, " pattern")
		return fmt.Sprintf("I'll add a `return %s` statement.", returnExpr)
	}

	// Struct/interface patterns.
	switch {
	case strings.Contains(lower, "struct definition for user"):
		return "I'll add a `User` struct definition with the specified fields."
	case strings.Contains(lower, "struct definition for request"):
		return "I'll add a `Request` struct definition for the payload."
	case strings.Contains(lower, "interface definition for service"):
		return "I'll add a `Service` interface definition."
	case strings.Contains(lower, "interface definition for reader"):
		return "I'll add a `DataReader` interface definition."
	case strings.Contains(lower, "interface declaration for userrepository"):
		return "I'll add a `UserRepository` interface declaration."
	case strings.Contains(lower, "interface declaration for cacheservice"):
		return "I'll add a `Cache` interface declaration."
	case strings.Contains(lower, "struct fields for db connection"):
		return "I'll add struct fields for DB connection settings."
	case strings.Contains(lower, "struct fields for auth token"):
		return "I'll add struct fields for an auth token payload."
	case strings.Contains(lower, "generic struct definition"):
		return "I'll add a generic struct definition."
	case strings.Contains(lower, "generic constraint interface"):
		return "I'll add a generic constraint interface."
	case strings.Contains(lower, "opening brace for struct"):
		return "I'll add the opening brace for the struct definition."
	case strings.Contains(lower, "opening brace for interface"):
		return "I'll add the opening brace for the interface definition."
	case strings.Contains(lower, "struct tag for json json omit empty"):
		return "I'll add a JSON struct tag with `omitempty`."
	case strings.Contains(lower, "struct tag for json id"):
		return "I'll add a JSON struct tag for the ID field."
	}

	// Function/method patterns.
	switch {
	case strings.Contains(lower, "main function declaration"):
		return "I'll add a `main` function declaration as the program entry point."
	case strings.Contains(lower, "main function header"):
		return "I'll add a `main` function header as the program entry point."
	case strings.Contains(lower, "init function declaration"):
		return "I'll add an `init` function declaration."
	case strings.Contains(lower, "method receiver declaration"):
		return "I'll add a method receiver declaration."
	case strings.Contains(lower, "pointer receiver method"):
		return "I'll add a pointer receiver method pattern."
	case strings.Contains(lower, "value receiver method"):
		return "I'll add a value receiver method pattern."
	case strings.Contains(lower, "unit test function signature"):
		return "I'll add a unit test function signature."
	case strings.Contains(lower, "benchmark test function signature"):
		return "I'll add a benchmark test function signature."
	case strings.Contains(lower, "example test function signature"):
		return "I'll add an example test function signature."
	case strings.Contains(lower, "generic function signature"):
		return "I'll add a generic function signature."
	case strings.Contains(lower, "method with context parameter"):
		return "I'll add a method with a `context.Context` parameter."
	case strings.Contains(lower, "constructor pattern"):
		return "I'll add a constructor pattern for the struct."
	}

	// Loop/control flow patterns.
	switch {
	case strings.Contains(lower, "opening brace to else if"):
		return "I'll add the opening brace to the `else if` condition to complete the block."
	case strings.Contains(lower, "opening brace to else"):
		return "I'll add the opening brace to the `else` statement to complete the block."
	case strings.Contains(lower, "opening brace to case"):
		return "I'll add the opening brace to the `case` statement to complete the block."
	case strings.Contains(lower, "opening brace to default"):
		return "I'll add the opening brace to the `default` case to complete the block."
	case strings.Contains(lower, "closing brace"):
		return "I'll add the closing brace to complete the block."
	case strings.Contains(lower, "closing parenthesis"):
		return "I'll add the closing parenthesis to complete the expression."
	case strings.Contains(lower, "opening parenthesis"):
		return "I'll add the opening parenthesis to begin the expression."
	case strings.Contains(lower, "range loop over slice"):
		return "I'll add a `range` loop to iterate over the slice."
	case strings.Contains(lower, "infinite for loop"):
		return "I'll add an infinite `for` loop."
	case strings.Contains(lower, "standard for loop header"):
		return "I'll add a standard `for` loop header."
	case strings.Contains(lower, "while-style for loop"):
		return "I'll add a while-style `for` loop."
	case strings.Contains(lower, "for range map loop"):
		return "I'll add a `for range` loop over the map."
	case strings.Contains(lower, "for range slice with index"):
		return "I'll add a `for range` loop over the slice with index."
	case strings.Contains(lower, "break statement"):
		return "I'll add a `break` statement to exit the loop."
	case strings.Contains(lower, "continue statement"):
		return "I'll add a `continue` statement to skip to the next iteration."
	case strings.Contains(lower, "fallthrough statement"):
		return "I'll add a `fallthrough` statement to continue to the next case."
	case strings.Contains(lower, "select block start"):
		return "I'll add a `select` block to handle multiple channel operations."
	case strings.Contains(lower, "select default case"):
		return "I'll add a `default` case to the `select` block."
	case strings.Contains(lower, "select context done"):
		return "I'll add a `ctx.Done()` case to the `select` block."
	case strings.Contains(lower, "type switch header"):
		return "I'll add a type switch header."
	case strings.Contains(lower, "type switch string case"):
		return "I'll add a string case to the type switch."
	case strings.Contains(lower, "type switch int case"):
		return "I'll add an int case to the type switch."
	case strings.Contains(lower, "table driven test loop"):
		return "I'll add a table-driven test loop pattern."
	case strings.Contains(lower, "table driven test setup"):
		return "I'll add a table-driven test setup."
	case strings.Contains(lower, "worker pool loop"):
		return "I'll add a worker pool loop pattern."
	case strings.Contains(lower, "channel range loop"):
		return "I'll add a channel range loop pattern."
	}

	// Channel/concurrency patterns.
	switch {
	case strings.Contains(lower, "channel receive statement"):
		return "I'll add a channel receive statement to read from the channel."
	case strings.Contains(lower, "channel receive assignment"):
		return "I'll add a channel receive assignment."
	case strings.Contains(lower, "channel send operation"):
		return "I'll add a channel send operation."
	case strings.Contains(lower, "channel close statement"):
		return "I'll add a `close(ch)` statement."
	case strings.Contains(lower, "channel creation"):
		return "I'll add a channel creation using `make(chan int)`."
	case strings.Contains(lower, "buffered channel creation"):
		return "I'll add a buffered channel creation."
	case strings.Contains(lower, "go routine invocation"):
		return "I'll add a goroutine invocation to run the function concurrently."
	case strings.Contains(lower, "sync mutex lock"):
		return "I'll add a `mu.Lock()` call to acquire the mutex."
	case strings.Contains(lower, "sync mutex unlock"):
		return "I'll add a `mu.Unlock()` call to release the mutex."
	case strings.Contains(lower, "waitgroup add"):
		return "I'll add a `wg.Add(1)` call to increment the WaitGroup counter."
	case strings.Contains(lower, "waitgroup done"):
		return "I'll add a `wg.Done()` call to decrement the WaitGroup counter."
	case strings.Contains(lower, "waitgroup wait"):
		return "I'll add a `wg.Wait()` call to block until the WaitGroup counter reaches zero."
	case strings.Contains(lower, "sync map load"):
		return "I'll add a `sync.Map` load call."
	case strings.Contains(lower, "sync map store"):
		return "I'll add a `sync.Map` store call."
	case strings.Contains(lower, "sync pool get"):
		return "I'll add a `sync.Pool` get-and-cast pattern."
	case strings.Contains(lower, "sync pool put"):
		return "I'll add a `sync.Pool` put call."
	case strings.Contains(lower, "sync once do"):
		return "I'll add a `sync.Once` execution block."
	case strings.Contains(lower, "sync.once value initialization"):
		return "I'll add a `sync.Once` value initialization pattern."
	case strings.Contains(lower, "atomic add int64"):
		return "I'll add an `atomic.AddInt64` operation."
	case strings.Contains(lower, "atomic load pointer"):
		return "I'll add an `atomic.LoadPointer` call."
	case strings.Contains(lower, "defer unlock"):
		return "I'll add a deferred `Unlock()` call to ensure the mutex is released."
	case strings.Contains(lower, "deferred close body"):
		return "I'll add a deferred `Close()` call to ensure the body is properly closed."
	case strings.Contains(lower, "defer rollback transaction"):
		return "I'll add a deferred `tx.Rollback()` call."
	case strings.Contains(lower, "defer file close"):
		return "I'll add a deferred `file.Close()` call."
	case strings.Contains(lower, "defer body close"):
		return "I'll add a deferred `resp.Body.Close()` call."
	case strings.Contains(lower, "cancel deferred"):
		return "I'll add a deferred `cancel()` invocation to release the context."
	}

	// Error handling patterns.
	switch {
	case strings.Contains(lower, "error check against nil"):
		return "I'll add an error check against `nil` to handle failures."
	case strings.Contains(lower, "nil check for pointer"):
		return "I'll add a `nil` check for the pointer."
	case strings.Contains(lower, "check for empty string"):
		return "I'll add a check for an empty string."
	case strings.Contains(lower, "check for zero length"):
		return "I'll add a check for zero length."
	case strings.Contains(lower, "check for positive number"):
		return "I'll add a check for a positive number."
	case strings.Contains(lower, "return custom error"):
		return "I'll add a `return errors.New` call to return a custom error."
	case strings.Contains(lower, "fmt error wrapping"):
		return "I'll add `fmt.Errorf` to wrap the error with context."
	case strings.Contains(lower, "error checking with fmt error"):
		return "I'll add error checking with `fmt.Errorf` return."
	case strings.Contains(lower, "panic call"):
		return "I'll add a `panic` call to handle the unexpected state."
	case strings.Contains(lower, "panic recover"):
		return "I'll add a panic-recover block pattern."
	case strings.Contains(lower, "t fatal error"):
		return "I'll add a `t.Fatalf` error check."
	case strings.Contains(lower, "t error logging"):
		return "I'll add a `t.Errorf` call to log a test error."
	}

	// Variable/type patterns.
	switch {
	case strings.Contains(lower, "short variable declaration integer"):
		return "I'll add a short variable declaration for an integer."
	case strings.Contains(lower, "short variable declaration string"):
		return "I'll add a short variable declaration for a string."
	case strings.Contains(lower, "short variable declaration slice"):
		return "I'll add a short variable declaration for a slice."
	case strings.Contains(lower, "short variable declaration map"):
		return "I'll add a short variable declaration for a map."
	case strings.Contains(lower, "type assertion check"):
		return "I'll add a type assertion check to safely extract the value."
	case strings.Contains(lower, "const declaration block"):
		return "I'll add a `const` declaration block."
	case strings.Contains(lower, "var declaration block"):
		return "I'll add a `var` declaration block."
	case strings.Contains(lower, "iota enum"):
		return "I'll add an `iota` enum definition."
	case strings.Contains(lower, "pointer string conversion"):
		return "I'll add a pointer-to-string conversion helper."
	case strings.Contains(lower, "pointer int conversion"):
		return "I'll add a pointer-to-int conversion helper."
	case strings.Contains(lower, "pointer bool conversion"):
		return "I'll add a pointer-to-bool conversion helper."
	case strings.Contains(lower, "interface implementation check"):
		return "I'll add an interface implementation check assertion."
	case strings.Contains(lower, "context value key type"):
		return "I'll add a context value key type pattern."
	}

	// String/slice/map patterns.
	switch {
	case strings.Contains(lower, "string join function"):
		return "I'll add a `strings.Join` call to concatenate the slice elements."
	case strings.Contains(lower, "string split function"):
		return "I'll add a `strings.Split` call to split the string."
	case strings.Contains(lower, "string contains check"):
		return "I'll add a `strings.Contains` check."
	case strings.Contains(lower, "string has prefix"):
		return "I'll add a `strings.HasPrefix` check."
	case strings.Contains(lower, "string has suffix"):
		return "I'll add a `strings.HasSuffix` check."
	case strings.Contains(lower, "string trim space"):
		return "I'll add a `strings.TrimSpace` call to remove surrounding whitespace."
	case strings.Contains(lower, "append to slice"):
		return "I'll add an `append` call to add an item to the slice."
	case strings.Contains(lower, "make byte slice"):
		return "I'll add a `make` call to create a byte slice."
	case strings.Contains(lower, "slice re-slicing"):
		return "I'll add a slice re-slicing operation."
	case strings.Contains(lower, "slice clearing"):
		return "I'll add a slice clearing operation."
	case strings.Contains(lower, "copy slice"):
		return "I'll add a `copy` call to copy slice elements."
	case strings.Contains(lower, "sort slice call"):
		return "I'll add a `sort.Strings` call to sort the slice."
	case strings.Contains(lower, "custom sort slice"):
		return "I'll add a `sort.SliceStable` call for custom sorting."
	case strings.Contains(lower, "map deletion key"):
		return "I'll add a `delete` call to remove a key from the map."
	}

	// JSON patterns.
	switch {
	case strings.Contains(lower, "json new encoder"):
		return "I'll add a `json.NewEncoder(w).Encode(resp)` call to encode the response as JSON."
	case strings.Contains(lower, "json marshal"):
		return "I'll add a `json.Marshal` call to serialize the value to JSON."
	case strings.Contains(lower, "json unmarshal"):
		return "I'll add a `json.Unmarshal` call to deserialize JSON data."
	}

	// Time patterns.
	switch {
	case strings.Contains(lower, "time now assignment"):
		return "I'll add a `time.Now()` assignment."
	case strings.Contains(lower, "time since calculation"):
		return "I'll add a `time.Since` calculation."
	case strings.Contains(lower, "time sleep"):
		return "I'll add a `time.Sleep` call."
	case strings.Contains(lower, "time parse"):
		return "I'll add a `time.Parse` call."
	case strings.Contains(lower, "time format"):
		return "I'll add a `time.Format` call."
	case strings.Contains(lower, "time ticker setup"):
		return "I'll add a `time.NewTicker` setup."
	case strings.Contains(lower, "time ticker stop"):
		return "I'll add a `ticker.Stop()` call."
	case strings.Contains(lower, "time timer setup"):
		return "I'll add a `time.NewTimer` setup."
	}

	// OS/random/crypto patterns.
	switch {
	case strings.Contains(lower, "os exit non zero"):
		return "I'll add an `os.Exit(1)` call to exit with a non-zero status."
	case strings.Contains(lower, "os getenv"):
		return "I'll add an `os.Getenv` call to read an environment variable."
	case strings.Contains(lower, "os lookupenv"):
		return "I'll add an `os.LookupEnv` call to check for an environment variable."
	case strings.Contains(lower, "rand intn"):
		return "I'll add a `rand.Intn` call to generate a random number."
	case strings.Contains(lower, "crypto rand read"):
		return "I'll add a `crypto/rand` read call for secure random bytes."
	case strings.Contains(lower, "sha256 sum"):
		return "I'll add a `sha256.Sum256` call to compute a hash."
	case strings.Contains(lower, "base64 std encoding encode"):
		return "I'll add a base64 standard encoding call to encode a string."
	case strings.Contains(lower, "base64 std encoding decode"):
		return "I'll add a base64 standard encoding call to decode a string."
	}

	// Math patterns.
	switch {
	case strings.Contains(lower, "math max"):
		return "I'll add a `math.Max` calculation."
	case strings.Contains(lower, "math min"):
		return "I'll add a `math.Min` calculation."
	case strings.Contains(lower, "math abs"):
		return "I'll add a `math.Abs` calculation."
	}

	// Comment/directive patterns.
	switch {
	case strings.Contains(lower, "comment for todo"):
		return "I'll add a `TODO` comment to mark the implementation as pending."
	case strings.Contains(lower, "comment for fixme"):
		return "I'll add a `FIXME` comment to flag a potential issue."
	case strings.Contains(lower, "nolint directive"):
		return "I'll add a `//nolint` directive to suppress the linter warning."
	case strings.Contains(lower, "go embed directive"):
		return "I'll add a `//go:embed` directive to embed static files."
	case strings.Contains(lower, "go build tag"):
		return "I'll add a `//go:build` tag to conditionally compile the file."
	}

	// Print/log patterns.
	switch {
	case strings.Contains(lower, "print statement with println"):
		return "I'll add a `fmt.Println` statement to print output."
	case strings.Contains(lower, "formatted print statement"):
		return "I'll add a formatted print statement using `fmt.Printf`."
	case strings.Contains(lower, "log error statement"):
		return "I'll add a `log.Printf` statement to log the error."
	case strings.Contains(lower, "log fatal statement"):
		return "I'll add a `log.Fatalf` statement to log a fatal error."
	}

	// Context patterns.
	switch {
	case strings.Contains(lower, "context background"):
		return "I'll add a `context.Background()` call to create a base context."
	case strings.Contains(lower, "context with timeout"):
		return "I'll add a `context.WithTimeout` call to set a timeout."
	}

	// Test patterns.
	switch {
	case strings.Contains(lower, "t run subtest"):
		return "I'll add a `t.Run` call to run a subtest."
	case strings.Contains(lower, "t parallel execution"):
		return "I'll add a `t.Parallel()` call to run the test in parallel."
	case strings.Contains(lower, "test cleanup hook"):
		return "I'll add a `t.Cleanup` hook to clean up test resources."
	case strings.Contains(lower, "unit test assertion"):
		return "I'll add a unit test assertion check."
	}

	// SQL patterns.
	switch {
	case strings.Contains(lower, "sql transaction setup"):
		return "I'll add a SQL transaction setup pattern."
	case strings.Contains(lower, "sql rows scan"):
		return "I'll add a SQL rows scan loop pattern."
	}

	// Flag patterns.
	if strings.Contains(lower, "flag parse setup") {
		return "I'll add a `flag.Parse` setup pattern."
	}

	// Create file/folder patterns.
	if strings.HasPrefix(lower, "create ") {
		return fmt.Sprintf("I'll create %s.", strings.TrimPrefix(lower, "create "))
	}

	// Delete patterns.
	if strings.HasPrefix(lower, "delete ") {
		return fmt.Sprintf("I'll delete %s.", strings.TrimPrefix(lower, "delete "))
	}

	// Remove patterns.
	if strings.HasPrefix(lower, "remove ") {
		return fmt.Sprintf("I'll remove %s.", strings.TrimPrefix(lower, "remove "))
	}

	// Edit/change/update/refactor patterns.
	for _, prefix := range []string{"rename ", "change ", "refactor ", "wrap ", "update ", "replace ", "convert "} {
		if strings.HasPrefix(lower, prefix) {
			return fmt.Sprintf("I'll %s.", strings.TrimPrefix(lower, prefix))
		}
	}

	// Fallback: use a template based on the prompt.
	return fmt.Sprintf("I'll add the requested Go code snippet: %s", codeAfter)
}

// importDescription returns a natural-language description of what an import
// enables in Go code.
func importDescription(importName string) string {
	descriptions := map[string]string{
		"fmt":           "formatted I/O operations",
		"time":          "time-related operations",
		"context":       "context propagation",
		"os":            "operating system interactions",
		"net/http":      "HTTP server and client functionality",
		"sync":          "synchronization primitives",
		"strings":       "string manipulation functions",
		"encoding/json": "JSON encoding and decoding",
		"math":          "mathematical operations",
		"io":            "I/O operations",
		"errors":        "error handling",
		"log":           "logging",
		"testing":       "test functions",
		"database/sql":  "SQL database operations",
		"crypto/rand":   "secure random number generation",
		"flag":          "command-line flag parsing",
	}
	if desc, ok := descriptions[importName]; ok {
		return desc
	}
	return "the requested functionality"
}

// loadCommandExamplesCSV reads the command examples CSV using the standard
// encoding/csv package, which correctly handles double-quoted multi-line fields.
func loadCommandExamplesCSV(path string) ([]*trainingpb.CommandExample, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open csv: %w", err)
	}
	defer f.Close()

	reader := csv.NewReader(f)
	reader.FieldsPerRecord = -1 // allow variable field counts
	reader.LazyQuotes = true

	var examples []*trainingpb.CommandExample
	lineNum := 0
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("read csv line %d: %w", lineNum+1, err)
		}
		lineNum++

		// Skip header row.
		if lineNum == 1 {
			continue
		}

		if len(record) < 4 {
			log.Printf("⚠️  Skipping record %d: %d fields, want 4: %v", lineNum, len(record), record)
			continue
		}

		typ := strings.TrimSpace(record[0])
		prompt := strings.TrimSpace(record[1])
		response := strings.TrimSpace(record[2])
		codeAfter := strings.TrimSpace(record[3])

		// For code_update examples, generate a natural-language assistant
		// response that explains the code modification. This makes the
		// training data multi-conversational.
		if response == "" && typ == "code_update" {
			response = generateAssistantResponse(prompt, codeAfter)
		}

		examples = append(examples, &trainingpb.CommandExample{
			Type:              typ,
			SystemPrompt:      systemPrompt,
			UserPrompt:        prompt,
			AssistantResponse: response,
			CodeBefore:        "",
			CodeAfter:         codeAfter,
		})
	}

	return examples, nil
}

// loadConversationsCSV reads the conversations CSV and groups turns by conversation ID.
func loadConversationsCSV(path string) ([]*trainingpb.Conversation, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open csv: %w", err)
	}
	defer f.Close()

	reader := csv.NewReader(f)
	reader.FieldsPerRecord = -1 // allow variable field counts
	reader.LazyQuotes = true

	convMap := make(map[string]*trainingpb.Conversation)
	convOrder := []string{}
	lineNum := 0
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("read csv line %d: %w", lineNum+1, err)
		}
		lineNum++

		// Skip header row.
		if lineNum == 1 {
			continue
		}

		if len(record) < 4 {
			log.Printf("⚠️  Skipping record %d: %d fields, want 4: %v", lineNum, len(record), record)
			continue
		}

		id := strings.TrimSpace(record[0])
		seq, _ := strconv.Atoi(strings.TrimSpace(record[1]))
		role := strings.TrimSpace(record[2])
		content := strings.TrimSpace(record[3])

		conv, ok := convMap[id]
		if !ok {
			conv = &trainingpb.Conversation{Id: id}
			convMap[id] = conv
			convOrder = append(convOrder, id)
		}
		conv.Turns = append(conv.Turns, &trainingpb.ConversationTurn{
			ConversationId: id,
			TurnSequence:   int32(seq),
			Role:           role,
			Content:        content,
		})
	}

	conversations := make([]*trainingpb.Conversation, 0, len(convOrder))
	for _, id := range convOrder {
		conversations = append(conversations, convMap[id])
	}
	return conversations, nil
}

func main() {
	commandsCSV := flag.String("commands", "data/training/command_examples.csv", "path to command examples CSV (type,prompt,response,code_after)")
	conversationsCSV := flag.String("conversations", "data/training/trainingdata/conversations.csv", "path to conversations CSV (conversation_id,turn_sequence,role,content)")
	outCommands := flag.String("out-commands", "data/training/command_examples.pb", "output path for command examples protobuf file")
	outConversations := flag.String("out-conversations", "data/training/trainingdata/conversations.pb", "output path for conversations protobuf file")
	flag.Parse()

	// Convert command examples.
	examples, err := loadCommandExamplesCSV(*commandsCSV)
	if err != nil {
		log.Fatalf("load command examples CSV: %v", err)
	}
	if err := os.MkdirAll(filepath.Dir(*outCommands), 0755); err != nil {
		log.Fatalf("create output dir: %v", err)
	}
	if err := trainingpb.SaveCommandExamplesToProto(*outCommands, examples); err != nil {
		log.Fatalf("save command examples protobuf: %v", err)
	}
	fmt.Printf("✅ Converted %d command examples → %s\n", len(examples), *outCommands)

	// Convert conversations.
	conversations, err := loadConversationsCSV(*conversationsCSV)
	if err != nil {
		log.Fatalf("load conversations CSV: %v", err)
	}
	if err := os.MkdirAll(filepath.Dir(*outConversations), 0755); err != nil {
		log.Fatalf("create output dir: %v", err)
	}
	if err := trainingpb.SaveConversationsToProto(*outConversations, conversations); err != nil {
		log.Fatalf("save conversations protobuf: %v", err)
	}
	fmt.Printf("✅ Converted %d conversations → %s\n", len(conversations), *outConversations)
}
