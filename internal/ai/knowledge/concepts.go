// Package knowledge provides a structured registry of Go idioms, design patterns,
// and textbook terminology mapped directly to required language constructs and AST
// mutation rules. This enables concept-guided code generation where high-level
// patterns (e.g. "Worker Pool") are decomposed into their primitive Go building
// blocks (e.g. chan, sync.WaitGroup, go fn()).
package knowledge

import "encoding/json"

// ConceptTemplate maps a high-level Go idiom or pattern to the concrete language
// constructs and structural mutation rules needed to implement it.
type ConceptTemplate struct {
	Term               string         `json:"term"`                // "Worker Pool"
	Synonyms           []string       `json:"synonyms"`            // ["task queue", "goroutine pool"]
	RequiredConstructs []string       `json:"required_constructs"` // ["chan", "go fn()", "sync.WaitGroup"]
	ASTMutations       []MutationRule `json:"ast_mutations"`       // Structural rules to apply
}

// MutationRule describes a single structural transformation to apply to the
// target Go source code. Each rule specifies what kind of mutation to perform,
// which struct or type to target, and the Go code template to inject.
type MutationRule struct {
	Type         string `json:"type"`          // "add_field", "wrap_body", "add_defer", "wrap_loop", "add_import"
	TargetStruct string `json:"target_struct"` // The struct or type name to modify (empty for file-level mutations)
	CodeTemplate string `json:"code_template"` // Go code template to inject
}

// Registry holds all known concept templates indexed by lowercase key.
type Registry struct {
	concepts map[string]ConceptTemplate
}

// NewRegistry creates a registry pre-populated with the standard Go idiom catalog.
func NewRegistry() *Registry {
	r := &Registry{
		concepts: make(map[string]ConceptTemplate),
	}
	r.registerDefaults()
	return r
}

// Get returns the concept template for the given key, or false if not found.
func (r *Registry) Get(key string) (ConceptTemplate, bool) {
	c, ok := r.concepts[key]
	return c, ok
}

// All returns all registered concept templates.
func (r *Registry) All() []ConceptTemplate {
	all := make([]ConceptTemplate, 0, len(r.concepts))
	for _, c := range r.concepts {
		all = append(all, c)
	}
	return all
}

// registerDefaults populates the registry with standard Go idioms and patterns.
// This is the core catalog that maps textbook terminology to AST-level mutations.
func (r *Registry) registerDefaults() {
	r.concepts["worker_pool"] = ConceptTemplate{
		Term:               "Worker Pool",
		Synonyms:           []string{"goroutine pool", "task pool", "parallel workers", "worker goroutines", "concurrent workers"},
		RequiredConstructs: []string{"sync.WaitGroup", "chan", "go fn()", "goroutine"},
		ASTMutations: []MutationRule{
			{
				Type:         "add_field",
				TargetStruct: "Planner",
				CodeTemplate: "tasks chan Task\nwg sync.WaitGroup",
			},
			{
				Type:         "wrap_loop",
				TargetStruct: "",
				CodeTemplate: "for i := 0; i < numWorkers; i++ {\n    p.wg.Add(1)\n    go p.worker()\n}",
			},
			{
				Type:         "add_defer",
				TargetStruct: "",
				CodeTemplate: "defer p.wg.Wait()",
			},
		},
	}

	r.concepts["caching"] = ConceptTemplate{
		Term:               "Caching",
		Synonyms:           []string{"cache layer", "memoization", "cache", "in-memory cache", "result cache"},
		RequiredConstructs: []string{"sync.RWMutex", "map", "sync.Map"},
		ASTMutations: []MutationRule{
			{
				Type:         "add_field",
				TargetStruct: "",
				CodeTemplate: "mu      sync.RWMutex\ncache   map[string]interface{}",
			},
			{
				Type:         "add_field",
				TargetStruct: "",
				CodeTemplate: "cacheHit  int64\ncacheMiss int64",
			},
		},
	}

	r.concepts["circuit_breaker"] = ConceptTemplate{
		Term:               "Circuit Breaker",
		Synonyms:           []string{"circuit breaker pattern", "resilience", "fault tolerance", "breaker"},
		RequiredConstructs: []string{"sync.Mutex", "time.Time", "atomic"},
		ASTMutations: []MutationRule{
			{
				Type:         "add_field",
				TargetStruct: "",
				CodeTemplate: "state       int32\nfailureCount int32\nlastFailure  time.Time\nmu          sync.Mutex",
			},
		},
	}

	r.concepts["rate_limiter"] = ConceptTemplate{
		Term:               "Rate Limiter",
		Synonyms:           []string{"rate limit", "throttle", "token bucket", "rate limiting"},
		RequiredConstructs: []string{"time.Ticker", "chan struct{}", "sync.Mutex"},
		ASTMutations: []MutationRule{
			{
				Type:         "add_field",
				TargetStruct: "",
				CodeTemplate: "ticker  *time.Ticker\nlimit   int\ntokens  chan struct{}\nmu      sync.Mutex",
			},
		},
	}

	r.concepts["observer_pattern"] = ConceptTemplate{
		Term:               "Observer Pattern",
		Synonyms:           []string{"publish-subscribe", "pubsub", "event bus", "event listener", "event emitter", "pub/sub"},
		RequiredConstructs: []string{"chan", "sync.Mutex", "interface{}", "slice"},
		ASTMutations: []MutationRule{
			{
				Type:         "add_field",
				TargetStruct: "",
				CodeTemplate: "subscribers map[string][]chan interface{}\nmu          sync.RWMutex",
			},
		},
	}

	r.concepts["singleton"] = ConceptTemplate{
		Term:               "Singleton",
		Synonyms:           []string{"single instance", "global instance", "once"},
		RequiredConstructs: []string{"sync.Once", "sync.Mutex"},
		ASTMutations: []MutationRule{
			{
				Type:         "add_field",
				TargetStruct: "",
				CodeTemplate: "once   sync.Once\ninstance *Singleton",
			},
		},
	}

	r.concepts["context_propagation"] = ConceptTemplate{
		Term:               "Context Propagation",
		Synonyms:           []string{"context", "context.Context", "context propagation", "deadline", "cancelation"},
		RequiredConstructs: []string{"context.Context", "context.WithCancel", "context.WithDeadline"},
		ASTMutations: []MutationRule{
			{
				Type:         "add_field",
				TargetStruct: "",
				CodeTemplate: "ctx    context.Context\ncancel context.CancelFunc",
			},
		},
	}

	r.concepts["fan_out"] = ConceptTemplate{
		Term:               "Fan-Out",
		Synonyms:           []string{"fan out", "parallel dispatch", "broadcast", "scatter"},
		RequiredConstructs: []string{"go fn()", "sync.WaitGroup", "chan"},
		ASTMutations: []MutationRule{
			{
				Type:         "wrap_loop",
				TargetStruct: "",
				CodeTemplate: "for _, item := range items {\n    wg.Add(1)\n    go func(val Item) {\n        defer wg.Done()\n        process(val)\n    }(item)\n}\nwg.Wait()",
			},
		},
	}

	r.concepts["fan_in"] = ConceptTemplate{
		Term:               "Fan-In",
		Synonyms:           []string{"fan in", "merge", "multiplex", "channel merge", "gather"},
		RequiredConstructs: []string{"chan", "go fn()", "sync.WaitGroup"},
		ASTMutations: []MutationRule{
			{
				Type:         "add_field",
				TargetStruct: "",
				CodeTemplate: "merged chan Result",
			},
			{
				Type:         "wrap_loop",
				TargetStruct: "",
				CodeTemplate: "for _, ch := range channels {\n    wg.Add(1)\n    go func(c <-chan Result) {\n        defer wg.Done()\n        for v := range c {\n            merged <- v\n        }\n    }(ch)\n}\ngo func() {\n    wg.Wait()\n    close(merged)\n}()",
			},
		},
	}

	r.concepts["pipeline"] = ConceptTemplate{
		Term:               "Pipeline",
		Synonyms:           []string{"data pipeline", "processing pipeline", "stage", "pipe", "stream processing"},
		RequiredConstructs: []string{"chan", "go fn()", "stage"},
		ASTMutations: []MutationRule{
			{
				Type:         "add_field",
				TargetStruct: "",
				CodeTemplate: "input  chan Input\noutput chan Output",
			},
			{
				Type:         "wrap_loop",
				TargetStruct: "",
				CodeTemplate: "go func() {\n    defer close(p.output)\n    for item := range p.input {\n        result := process(item)\n        p.output <- result\n    }\n}()",
			},
		},
	}

	r.concepts["graceful_shutdown"] = ConceptTemplate{
		Term:               "Graceful Shutdown",
		Synonyms:           []string{"shutdown", "clean shutdown", "graceful", "signal handling", "os.Signal"},
		RequiredConstructs: []string{"os.Signal", "os/signal", "context.WithCancel", "sync.WaitGroup"},
		ASTMutations: []MutationRule{
			{
				Type:         "add_field",
				TargetStruct: "",
				CodeTemplate: "sigCh  chan os.Signal\nstopCh chan struct{}\nwg     sync.WaitGroup",
			},
			{
				Type:         "add_import",
				TargetStruct: "",
				CodeTemplate: "\"os/signal\"\n\"os\"",
			},
		},
	}

	r.concepts["connection_pool"] = ConceptTemplate{
		Term:               "Connection Pool",
		Synonyms:           []string{"conn pool", "pool", "connection pooling", "resource pool"},
		RequiredConstructs: []string{"chan", "sync.Mutex", "sync.WaitGroup"},
		ASTMutations: []MutationRule{
			{
				Type:         "add_field",
				TargetStruct: "",
				CodeTemplate: "pool   chan *Conn\nmaxSize int\nactive int32\nmu     sync.Mutex",
			},
		},
	}

	r.concepts["retry_pattern"] = ConceptTemplate{
		Term:               "Retry Pattern",
		Synonyms:           []string{"retry", "retry with backoff", "exponential backoff", "retry logic", "backoff"},
		RequiredConstructs: []string{"time.Duration", "time.Sleep", "math"},
		ASTMutations: []MutationRule{
			{
				Type:         "add_field",
				TargetStruct: "",
				CodeTemplate: "maxRetries   int\nbaseDelay    time.Duration\nmaxDelay     time.Duration",
			},
		},
	}

	r.concepts["dependency_injection"] = ConceptTemplate{
		Term:               "Dependency Injection",
		Synonyms:           []string{"DI", "injection", "constructor injection", "inversion of control", "ioc"},
		RequiredConstructs: []string{"interface{}", "struct embedding", "constructor function"},
		ASTMutations: []MutationRule{
			{
				Type:         "add_field",
				TargetStruct: "",
				CodeTemplate: "store     Store\ncache     Cache\nlogger    Logger\nmetrics   MetricsReporter",
			},
		},
	}
}

// MarshalJSON implements json.Marshaler for serializing the full registry.
func (r *Registry) MarshalJSON() ([]byte, error) {
	return json.MarshalIndent(r.concepts, "", "  ")
}

// UnmarshalJSON implements json.Unmarshaler for deserializing a registry.
func (r *Registry) UnmarshalJSON(data []byte) error {
	return json.Unmarshal(data, &r.concepts)
}

// DefaultConceptKeys returns the keys of all built-in concepts.
func DefaultConceptKeys() []string {
	return []string{
		"worker_pool",
		"caching",
		"circuit_breaker",
		"rate_limiter",
		"observer_pattern",
		"singleton",
		"context_propagation",
		"fan_out",
		"fan_in",
		"pipeline",
		"graceful_shutdown",
		"connection_pool",
		"retry_pattern",
		"dependency_injection",
	}
}
