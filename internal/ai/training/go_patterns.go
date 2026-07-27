// Package training provides specialized auxiliary objectives for Go patterns.
// These include error handling completion, interface implementation generation,
// and concurrency primitive training tasks.
package training

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"strings"
)

// GoPatternExample represents a training example for a specific Go pattern.
type GoPatternExample struct {
	PatternType string `json:"pattern_type"` // "error_handling", "interface_impl", "concurrency", "struct_tag"
	Instruction string `json:"instruction"`
	Input       string `json:"input"`
	Output      string `json:"output"`
	Explanation string `json:"explanation,omitempty"`
}

// GenerateErrorHandlingExamples generates training examples for idiomatic Go error handling.
// These cover patterns like:
//   - fmt.Errorf("...: %w", err) for error wrapping
//   - errors.Is() / errors.As() for error inspection
//   - if err != nil { return ... } for error checking
func GenerateErrorHandlingExamples() []GoPatternExample {
	return []GoPatternExample{
		{
			PatternType: "error_handling",
			Instruction: "Wrap the error with additional context using fmt.Errorf",
			Input: `func readConfig(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, REPLACE_ME
	}
	// ... parse config
	return &cfg, nil
}`,
			Output: `func readConfig(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading config %s: %w", path, err)
	}
	// ... parse config
	return &cfg, nil
}`,
			Explanation: "Use fmt.Errorf with %w to wrap errors with context while preserving the original error for errors.Is/errors.As.",
		},
		{
			PatternType: "error_handling",
			Instruction: "Check if the error is a specific sentinel error using errors.Is",
			Input: `func processFile(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		if REPLACE_ME {
			// File doesn't exist, use defaults
			return nil
		}
		return err
	}
	// ... process data
	return nil
}`,
			Output: `func processFile(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			// File doesn't exist, use defaults
			return nil
		}
		return err
	}
	// ... process data
	return nil
}`,
			Explanation: "Use errors.Is to check for specific sentinel errors in the error chain.",
		},
		{
			PatternType: "error_handling",
			Instruction: "Type-assert the error to access custom error fields using errors.As",
			Input: `func handleRequest() error {
	err := processRequest()
	if err != nil {
		var netErr *net.DNSError
		if REPLACE_ME {
			return fmt.Errorf("DNS resolution failed: %v", netErr)
		}
		return err
	}
	return nil
}`,
			Output: `func handleRequest() error {
	err := processRequest()
	if err != nil {
		var netErr *net.DNSError
		if errors.As(err, &netErr) {
			return fmt.Errorf("DNS resolution failed: %v", netErr)
		}
		return err
	}
	return nil
}`,
			Explanation: "Use errors.As to type-assert errors and access custom error fields.",
		},
		{
			PatternType: "error_handling",
			Instruction: "Create a new sentinel error with errors.New",
			Input: `package config

var (
	ErrNotFound = REPLACE_ME
	ErrInvalid  = errors.New("invalid configuration")
)`,
			Output: `package config

var (
	ErrNotFound = errors.New("config not found")
	ErrInvalid  = errors.New("invalid configuration")
)`,
			Explanation: "Define sentinel errors at the package level using errors.New for callers to check with errors.Is.",
		},
		{
			PatternType: "error_handling",
			Instruction: "Join multiple errors using errors.Join",
			Input: `func validate(input *Input) error {
	var errs []error
	if input.Name == "" {
		errs = append(errs, errors.New("name is required"))
	}
	if input.Age < 0 {
		errs = append(errs, errors.New("age must be non-negative"))
	}
	if len(errs) > 0 {
		return REPLACE_ME
	}
	return nil
}`,
			Output: `func validate(input *Input) error {
	var errs []error
	if input.Name == "" {
		errs = append(errs, errors.New("name is required"))
	}
	if input.Age < 0 {
		errs = append(errs, errors.New("age must be non-negative"))
	}
	if len(errs) > 0 {
		return errors.Join(errs...)
	}
	return nil
}`,
			Explanation: "Use errors.Join to combine multiple errors into a single error value.",
		},
	}
}

// GenerateInterfaceImplExamples generates training examples for interface implementation.
// These cover:
//   - var _ Interface = (*Type)(nil) compile-time check
//   - Implementing io.Reader, io.Writer, etc.
//   - Implementing custom interfaces
func GenerateInterfaceImplExamples() []GoPatternExample {
	return []GoPatternExample{
		{
			PatternType: "interface_impl",
			Instruction: "Add compile-time interface satisfaction check for a custom type",
			Input: `package store

type UserStore struct {
	db *sql.DB
}

// REPLACE_ME: Add compile-time check that UserStore implements UserRepository
`,
			Output: `package store

type UserStore struct {
	db *sql.DB
}

var _ UserRepository = (*UserStore)(nil)
`,
			Explanation: "Use 'var _ Interface = (*Type)(nil)' to get a compile-time error if the type doesn't implement the interface.",
		},
		{
			PatternType: "interface_impl",
			Instruction: "Implement the io.Reader interface for a custom type",
			Input: `type LimitedReader struct {
	reader io.Reader
	limit  int64
	read   int64
}

func (r *LimitedReader) REPLACE_ME {
	if r.read >= r.limit {
		return 0, io.EOF
	}
	n, err := r.reader.Read(p)
	if n > 0 {
		r.read += int64(n)
		if r.read > r.limit {
			n -= int(r.read - r.limit)
			r.read = r.limit
		}
	}
	return n, err
}`,
			Output: `type LimitedReader struct {
	reader io.Reader
	limit  int64
	read   int64
}

func (r *LimitedReader) Read(p []byte) (int, error) {
	if r.read >= r.limit {
		return 0, io.EOF
	}
	n, err := r.reader.Read(p)
	if n > 0 {
		r.read += int64(n)
		if r.read > r.limit {
			n -= int(r.read - r.limit)
			r.read = r.limit
		}
	}
	return n, err
}`,
			Explanation: "Implement Read(p []byte) (int, error) to satisfy io.Reader. Track bytes read and return io.EOF when limit is reached.",
		},
		{
			PatternType: "interface_impl",
			Instruction: "Generate a full struct stub that implements a given interface",
			Input: `// Given this interface:
type Cache interface {
	Get(key string) (interface{}, bool)
	Set(key string, value interface{})
	Delete(key string)
	Clear() error
}

// Generate a MemoryCache struct that implements Cache
type MemoryCache struct {
	REPLACE_ME
}

func (c *MemoryCache) Get(key string) (interface{}, bool) {
	REPLACE_ME
}

func (c *MemoryCache) Set(key string, value interface{}) {
	REPLACE_ME
}

func (c *MemoryCache) Delete(key string) {
	REPLACE_ME
}

func (c *MemoryCache) Clear() error {
	REPLACE_ME
}`,
			Output: `type MemoryCache struct {
	data map[string]interface{}
	mu   sync.RWMutex
}

func (c *MemoryCache) Get(key string) (interface{}, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	val, ok := c.data[key]
	return val, ok
}

func (c *MemoryCache) Set(key string, value interface{}) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.data[key] = value
}

func (c *MemoryCache) Delete(key string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.data, key)
}

func (c *MemoryCache) Clear() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.data = make(map[string]interface{})
	return nil
}`,
			Explanation: "Implement the interface with a map-backed store protected by sync.RWMutex for concurrent access.",
		},
	}
}

// GenerateConcurrencyExamples generates training examples for Go concurrency patterns.
// These cover:
//   - sync.WaitGroup for goroutine synchronization
//   - select for channel multiplexing
//   - chan struct{} for signaling
//   - sync.Once for one-time initialization
func GenerateConcurrencyExamples() []GoPatternExample {
	return []GoPatternExample{
		{
			PatternType: "concurrency",
			Instruction: "Use sync.WaitGroup to wait for multiple goroutines to complete",
			Input: `func processItems(items []Item) {
	var wg sync.WaitGroup
	for _, item := range items {
		REPLACE_ME
		go func(it Item) {
			REPLACE_ME
			processItem(it)
			REPLACE_ME
		}(item)
	}
	REPLACE_ME
}`,
			Output: `func processItems(items []Item) {
	var wg sync.WaitGroup
	for _, item := range items {
		wg.Add(1)
		go func(it Item) {
			defer wg.Done()
			processItem(it)
		}(item)
	}
	wg.Wait()
}`,
			Explanation: "Use wg.Add(1) before each goroutine, defer wg.Done() inside, and wg.Wait() to block until all complete.",
		},
		{
			PatternType: "concurrency",
			Instruction: "Use select with a timeout channel to implement a deadline",
			Input: `func fetchWithTimeout(url string, timeout time.Duration) (*Response, error) {
	result := make(chan *Response, 1)
	errCh := make(chan error, 1)

	go func() {
		resp, err := http.Get(url)
		if err != nil {
			errCh <- err
			return
		}
		result <- resp
	}()

	REPLACE_ME
}`,
			Output: `func fetchWithTimeout(url string, timeout time.Duration) (*Response, error) {
	result := make(chan *Response, 1)
	errCh := make(chan error, 1)

	go func() {
		resp, err := http.Get(url)
		if err != nil {
			errCh <- err
			return
		}
		result <- resp
	}()

	select {
	case resp := <-result:
		return resp, nil
	case err := <-errCh:
		return nil, err
	case <-time.After(timeout):
		return nil, fmt.Errorf("request timed out after %v", timeout)
	}
}`,
			Explanation: "Use select to multiplex multiple channels: the result, error, and a timeout channel created with time.After.",
		},
		{
			PatternType: "concurrency",
			Instruction: "Use chan struct{} for goroutine signaling (close to broadcast)",
			Input: `type WorkerPool struct {
	workers int
	tasks   chan Task
	REPLACE_ME // done channel for signaling
}

func (wp *WorkerPool) Start() {
	wp.done = make(REPLACE_ME)
	for i := 0; i < wp.workers; i++ {
		go wp.worker(i)
	}
}

func (wp *WorkerPool) Stop() {
	REPLACE_ME // Signal all workers to stop
}

func (wp *WorkerPool) worker(id int) {
	for {
		select {
		case task, ok := <-wp.tasks:
			if !ok {
				return
			}
			processTask(task)
		case REPLACE_ME:
			return
		}
	}
}`,
			Output: `type WorkerPool struct {
	workers int
	tasks   chan Task
	done    chan struct{} // done channel for signaling
}

func (wp *WorkerPool) Start() {
	wp.done = make(chan struct{})
	for i := 0; i < wp.workers; i++ {
		go wp.worker(i)
	}
}

func (wp *WorkerPool) Stop() {
	close(wp.done) // Signal all workers to stop
}

func (wp *WorkerPool) worker(id int) {
	for {
		select {
		case task, ok := <-wp.tasks:
			if !ok {
				return
			}
			processTask(task)
		case <-wp.done:
			return
		}
	}
}`,
			Explanation: "Use chan struct{} for signaling. Closing the channel broadcasts to all goroutines via the select statement.",
		},
		{
			PatternType: "concurrency",
			Instruction: "Use sync.Once for thread-safe one-time initialization",
			Input: `type Singleton struct {
	config *Config
	once   REPLACE_ME
}

func (s *Singleton) GetConfig() *Config {
	REPLACE_ME
	return s.config
}

func (s *Singleton) initConfig() {
	// Expensive initialization
	s.config = loadConfig()
}`,
			Output: `type Singleton struct {
	config *Config
	once   sync.Once
}

func (s *Singleton) GetConfig() *Config {
	s.once.Do(s.initConfig)
	return s.config
}

func (s *Singleton) initConfig() {
	// Expensive initialization
	s.config = loadConfig()
}`,
			Explanation: "Use sync.Once.Do to ensure a function is called exactly once, even from multiple goroutines.",
		},
	}
}

// GenerateAllGoPatternExamples returns all Go pattern training examples.
func GenerateAllGoPatternExamples() []GoPatternExample {
	var examples []GoPatternExample
	examples = append(examples, GenerateErrorHandlingExamples()...)
	examples = append(examples, GenerateInterfaceImplExamples()...)
	examples = append(examples, GenerateConcurrencyExamples()...)
	return examples
}

// ExtractInterfaceMethods extracts method signatures from a Go interface definition.
func ExtractInterfaceMethods(code string) ([]string, error) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "", code, parser.ParseComments)
	if err != nil {
		return nil, fmt.Errorf("parse error: %w", err)
	}

	var methods []string
	ast.Inspect(f, func(n ast.Node) bool {
		if ts, ok := n.(*ast.TypeSpec); ok {
			if iface, ok := ts.Type.(*ast.InterfaceType); ok {
				for _, m := range iface.Methods.List {
					if len(m.Names) > 0 {
						methodName := m.Names[0].Name
						methodType := types.ExprString(m.Type)
						methods = append(methods, fmt.Sprintf("%s %s", methodName, methodType))
					}
				}
			}
		}
		return true
	})

	return methods, nil
}

// GenerateInterfaceStub generates a struct stub that implements the given interface.
func GenerateInterfaceStub(structName string, code string) (string, error) {
	methods, err := ExtractInterfaceMethods(code)
	if err != nil {
		return "", err
	}

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("type %s struct {\n", structName))
	sb.WriteString("\t// TODO: add fields\n")
	sb.WriteString("}\n\n")

	// Compile-time check
	sb.WriteString(fmt.Sprintf("var _ InterfaceName = (*%s)(nil)\n\n", structName))

	for _, m := range methods {
		// Parse the method signature
		parts := strings.SplitN(m, " ", 2)
		if len(parts) == 2 {
			methodName := parts[0]
			methodSig := parts[1]
			sb.WriteString(fmt.Sprintf("func (s *%s) %s %s {\n", structName, methodName, methodSig))
			sb.WriteString("\t// TODO: implement\n")
			// Add return statement if there are return values
			if strings.Contains(methodSig, ")") {
				sb.WriteString("\treturn\n")
			}
			sb.WriteString("}\n\n")
		}
	}

	return sb.String(), nil
}
