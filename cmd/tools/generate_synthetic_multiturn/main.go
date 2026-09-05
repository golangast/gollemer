package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

type Conversation struct {
	ID      string   `json:"conversation_id"`
	Turns   []Turn   `json:"turns"`
	Problem string   `json:"problem,omitempty"`
	Tags    []string `json:"tags,omitempty"`
}

type Turn struct {
	Sequence       int    `json:"turn_sequence"`
	Role           string `json:"role"`
	Content        string `json:"content"`
	ReasoningTrace string `json:"reasoning_trace,omitempty"`
}

func main() {
	scenarios := generateScenarios()

	outputPath := "data/training/trainingdata/conversing.yaml"
	if len(os.Args) > 1 {
		outputPath = os.Args[1]
	}

	var sb strings.Builder
	sb.WriteString("\n# Synthetic Multi-Turn Bootstrapping - Auto-generated\n")
	sb.WriteString("# Source: cmd/tools/generate_synthetic_multiturn\n\n")

	for _, conv := range scenarios {
		sb.WriteString(fmt.Sprintf("- conversation_id: \"%s\"\n", conv.ID))
		sb.WriteString("  turns:\n")
		for _, turn := range conv.Turns {
			sb.WriteString(fmt.Sprintf("  - turn_sequence: %d\n", turn.Sequence))
			sb.WriteString(fmt.Sprintf("    role: \"%s\"\n", turn.Role))
			content := strings.ReplaceAll(turn.Content, "\n", "\\n")
			sb.WriteString(fmt.Sprintf("    content: \"%s\"\n", content))
			if turn.Role == "assistant" && turn.ReasoningTrace != "" {
				reasoning := strings.ReplaceAll(turn.ReasoningTrace, "\n", "\\n")
				sb.WriteString(fmt.Sprintf("    reasoning_trace: \"%s\"\n", reasoning))
			}
		}
		sb.WriteString("\n")
	}

	f, err := os.OpenFile(outputPath, os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		fmt.Printf("Error opening file: %v\n", err)
		os.Exit(1)
	}
	defer f.Close()

	if _, err := f.WriteString(sb.String()); err != nil {
		fmt.Printf("Error writing: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("✅ Appended %d synthetic multi-turn conversations to %s\n", len(scenarios), outputPath)
	fmt.Printf("   Total turns: %d\n", countTurns(scenarios))
}

func generateScenarios() []Conversation {
	var scenarios []Conversation

	scenarios = append(scenarios, generateRaceConditionScenario())
	scenarios = append(scenarios, generateGCTuningScenario())
	scenarios = append(scenarios, generateChannelLifecycleScenario())
	scenarios = append(scenarios, generateContextCancellationScenario())
	scenarios = append(scenarios, generateMutexRecoveryScenario())
	scenarios = append(scenarios, generatePipelineBackpressureScenario())
	scenarios = append(scenarios, generateAtomicCountersScenario())
	scenarios = append(scenarios, generateSelectStarvationScenario())

	return scenarios
}

func generateRaceConditionScenario() Conversation {
	return Conversation{
		ID:      "conv_synth_race_01",
		Problem: "Race condition in concurrent map writes",
		Tags:    []string{"race", "sync.Map", "mutex", "concurrency"},
		Turns: []Turn{
			{Sequence: 1, Role: "user", Content: "I have a race condition in my Go code. Two goroutines are writing to the same map without synchronization."},
			{Sequence: 2, Role: "assistant", Content: "[TRIPLETS] Subject: Goroutine -> Action: Write -> Object: Map | Subject: Map -> Action: Corrupt -> Object: State\n</think> 1. Identify core task: Fix concurrent map write. 2. Evaluate options: sync.Mutex, sync.Map, channel serialization. 3. Decide heuristic: Use mutex for write-heavy access or sync.Map for read-heavy.\n</think> Go maps panic on concurrent writes. Wrap access in a sync.Mutex or use sync.Map. For write-heavy workloads, a mutex is usually faster. Use `go run -race` to confirm."},
			{Sequence: 3, Role: "user", Content: "The race detector confirms it. But sync.Map feels heavy for my use case. Is there a lighter alternative?"},
			{Sequence: 4, Role: "assistant", Content: "[TRIPLETS] Subject: Developer -> Action: Seek -> Object: Alternative | Subject: sync.Mutex -> Action: Protect -> Object: Map\n</think> 1. Identify core task: Lightweight concurrent map. 2. Evaluate options: Mutex, channel owner, copy-on-write. 3. Decide heuristic: Channel-based owner goroutine serializes access without explicit locking.\n</think> Use a channel-based owner goroutine pattern. All reads/writes go through a single goroutine via a request channel. This avoids mutex overhead and is easy to reason about. Example: `type MapOwner struct { updates chan func(map[string]int); data map[string]int }`"},
			{Sequence: 5, Role: "user", Content: "What about garbage collection? Will many short-lived goroutines cause GC pressure?"},
			{Sequence: 6, Role: "assistant", Content: "[TRIPLETS] Subject: Goroutine -> Action: Allocate -> Object: Memory | Subject: GC -> Action: Reclaim -> Object: Memory\n</think> 1. Identify core task: GC pressure from goroutines. 2. Evaluate options: Object pooling, sync.Pool, reducing allocations. 3. Decide heuristic: Go's GC handles short-lived objects well; only optimize if profiling shows it.\n</think> Go's concurrent GC is designed for high allocation rates. Short-lived goroutines and their stack allocations are cheap. Only worry about GC pressure if `pprof` shows it. If needed, use `sync.Pool` to reuse objects and reduce allocation churn."},
		},
	}
}

func generateGCTuningScenario() Conversation {
	return Conversation{
		ID:      "conv_synth_gc_01",
		Problem: "Garbage collection tuning for high-throughput service",
		Tags:    []string{"gc", "GOGC", "memory", "performance"},
		Turns: []Turn{
			{Sequence: 1, Role: "user", Content: "My Go service has high throughput but GC pauses are causing latency spikes. How do I tune the garbage collector?"},
			{Sequence: 2, Role: "assistant", Content: "[TRIPLETS] Subject: GC -> Action: Tune -> Object: Latency | Subject: Service -> Action: Experience -> Object: Pause\n</think> 1. Identify core task: Reduce GC pause latency. 2. Evaluate options: GOGC tuning, runtime.SetGCPercent, arena allocation. 3. Decide heuristic: Lower GOGC means more frequent but smaller pauses; use profiling to find the sweet spot.\n</think> Start by profiling with `go tool pprof` to see GC impact. Tune `GOGC` (default 100) — lower values trigger GC more frequently with smaller pauses. For latency-sensitive services, try `GOGC=50` or use `runtime/debug.SetGCPercent` at runtime. Avoid manual memory management unless absolutely necessary."},
			{Sequence: 3, Role: "user", Content: "I set GOGC=50 but CPU usage went up 20%. Is there a better approach?"},
			{Sequence: 4, Role: "assistant", Content: "[TRIPLETS] Subject: CPU -> Action: Increase -> Object: Usage | Subject: GOGC -> Action: Trigger -> Object: GC\n</think> 1. Identify core task: Balance GC frequency vs CPU. 2. Evaluate options: GOGC tuning, sync.Pool, value reuse, arena. 3. Decide heuristic: Use sync.Pool for frequently allocated objects; tune GOGC incrementally.\n</think> Lower GOGC increases CPU because GC runs more often. Instead, reduce allocation rate first: reuse objects with `sync.Pool`, preallocate slices with `make`, and avoid unnecessary boxing. Then tune GOGC in small steps (100 → 80 → 60). Measure both latency p99 and CPU to find the optimal point."},
			{Sequence: 5, Role: "user", Content: "Can you show me a sync.Pool example for reducing allocations in a hot path?"},
			{Sequence: 6, Role: "assistant", Content: "[TRIPLETS] Subject: sync.Pool -> Action: Reuse -> Object: Object | Subject: Goroutine -> Action: Allocate -> Object: Memory\n</think> 1. Identify core task: Reduce allocations in hot path. 2. Evaluate options: sync.Pool, preallocation, stack allocation. 3. Decide heuristic: Use sync.Pool for objects that are frequently created and discarded.\n</think> Here's a sync.Pool pattern for a hot path:\n\n```go\nvar bufPool = sync.Pool{\n    New: func() interface{} { return new(bytes.Buffer) },\n}\n\nfunc handleRequest() {\n    buf := bufPool.Get().(*bytes.Buffer)\n    buf.Reset()\n    defer bufPool.Put(buf)\n    // use buf...\n}\n```\n\nThis avoids allocating a new buffer on every request. The pool grows to the peak concurrency level and reuses objects across calls."},
		},
	}
}

func generateChannelLifecycleScenario() Conversation {
	return Conversation{
		ID:      "conv_synth_channel_01",
		Problem: "Channel lifecycle management in pipelines",
		Tags:    []string{"channel", "pipeline", "close", "goroutine"},
		Turns: []Turn{
			{Sequence: 1, Role: "user", Content: "I'm building a pipeline with channels. My worker goroutines seem to leak when the pipeline is cancelled."},
			{Sequence: 2, Role: "assistant", Content: "[TRIPLETS] Subject: Channel -> Action: Block -> Object: Goroutine | Subject: Context -> Action: Cancel -> Object: Pipeline\n</think> 1. Identify core task: Fix goroutine leaks in channel pipeline. 2. Evaluate options: Context cancellation, done channels, buffered channels. 3. Decide heuristic: Always drain or close channels when exiting; use select with ctx.Done().\n</think> Worker goroutines leak when they block forever on channel operations after cancellation. Use `context.Context` to broadcast cancellation, and wrap blocking ops in `select`:\n\n```go\nselect {\ncase ch <- val:\ncase <-ctx.Done():\n    return\n}\n```\n\nEnsure the sender closes the channel so receivers can exit cleanly."},
			{Sequence: 3, Role: "user", Content: "How do I handle a case where the receiver might still be blocked when the sender finishes?"},
			{Sequence: 4, Role: "assistant", Content: "[TRIPLETS] Subject: Receiver -> Action: Block -> Object: Channel | Subject: Sender -> Action: Close -> Object: Channel\n</think> 1. Identify core task: Unblock receiver after sender finishes. 2. Evaluate options: Close channel, done channel, select with default. 3. Decide heuristic: Close the channel when all sends are done; receivers exit when drained.\n</think> Use `sync.WaitGroup` to track active senders. When all senders finish, close the channel. Receivers use `for val := range ch` to drain until closed:\n\n```go\nvar wg sync.WaitGroup\ngo func() {\n    defer close(ch)\n    wg.Wait()\n}()\nfor val := range ch {\n    // process val\n}\n```\n\nThis guarantees receivers never block indefinitely."},
			{Sequence: 5, Role: "user", Content: "What if I need to send multiple results from one goroutine to many receivers?"},
			{Sequence: 6, Role: "assistant", Content: "[TRIPLETS] Subject: Goroutine -> Action: Send -> Object: Channel | Subject: Receiver -> Action: Receive -> Object: Value\n</think> 1. Identify core task: Fan-out pattern. 2. Evaluate options: Multiple channels, single channel with multiple receivers, broadcast. 3. Decide heuristic: Use a single channel with multiple receivers for work distribution; use context for cancellation.\n</think> For one-to-many distribution, use a single channel with multiple receivers (work stealing pattern):\n\n```go\nresults := make(chan Result, workers)\nfor i := 0; i < workers; i++ {\n    go func(id int) {\n        for res := range results {\n            process(id, res)\n        }\n    }(i)\n}\nfor _, work := range workItems {\n    results <- process(work)\n}\nclose(results)\n```\n\nEach receiver processes items independently. Closing the channel signals all receivers to exit."},
		},
	}
}

func generateContextCancellationScenario() Conversation {
	return Conversation{
		ID:      "conv_synth_ctx_01",
		Problem: "Context cancellation propagation across API boundaries",
		Tags:    []string{"context", "cancellation", "timeout", "http"},
		Turns: []Turn{
			{Sequence: 1, Role: "user", Content: "I'm confused about when to use context.Context. My HTTP handler takes one, but I'm not sure what to pass down."},
			{Sequence: 2, Role: "assistant", Content: "[TRIPLETS] Subject: Context -> Action: Carry -> Object: Deadline | Subject: Handler -> Action: Pass -> Object: Function\n</think> 1. Identify core task: Context propagation. 2. Evaluate options: Background, WithTimeout, WithCancel. 3. Decide heuristic: Pass request context to all downstream I/O functions.\n</think> Pass the `context.Context` from your HTTP handler to every function that performs I/O, database queries, or spans multiple goroutines. Use `context.WithTimeout` or `context.WithCancel` to derive child contexts. Never store context in a struct; always pass it as the first parameter."},
			{Sequence: 3, Role: "user", Content: "What happens if I call a blocking function without passing context?"},
			{Sequence: 4, Role: "assistant", Content: "[TRIPLETS] Subject: Function -> Action: Block -> Object: Goroutine | Subject: Request -> Action: Timeout -> Object: Handler\n</think> 1. Identify core task: Consequences of ignoring context. 2. Evaluate options: Leaked goroutines, hung connections, resource exhaustion. 3. Decide heuristic: Always respect context cancellation.\n</think> Without context, a blocking function can hang forever if the client disconnects or the operation stalls. This leaks goroutines and file descriptors. Always pass context so your function can abort early when the parent operation is cancelled. Wrap top-level handlers with `recover()` to return 500 instead of crashing."},
			{Sequence: 5, Role: "user", Content: "Can I use the same context for multiple parallel requests?"},
			{Sequence: 6, Role: "assistant", Content: "[TRIPLETS] Subject: Context -> Action: Cancel -> Object: Child | Subject: Goroutine -> Action: Receive -> Object: Signal\n</think> 1. Identify core task: Context sharing across parallel requests. 2. Evaluate options: Single context vs derived contexts. 3. Decide heuristic: Derive separate child contexts for each parallel request.\n</think> Derive separate child contexts for each parallel request using `context.WithCancel` or `context.WithTimeout`. If one request fails and you cancel its child context, the other parallel requests continue unaffected. Cancellation cascades from parent to child, so be careful with shared parent contexts."},
		},
	}
}

func generateMutexRecoveryScenario() Conversation {
	return Conversation{
		ID:      "conv_synth_mutex_01",
		Problem: "Mutex recovery and deadlock prevention patterns",
		Tags:    []string{"mutex", "deadlock", "defer", "recovery"},
		Turns: []Turn{
			{Sequence: 1, Role: "user", Content: "I keep getting deadlocks when using mutexes. How can I make my locking safer?"},
			{Sequence: 2, Role: "assistant", Content: "[TRIPLETS] Subject: Goroutine -> Action: Lock -> Object: Mutex | Subject: Runtime -> Action: Detect -> Object: Deadlock\n</think> 1. Identify core task: Make mutex locking safer. 2. Evaluate options: defer unlock, consistent ordering, lock timeouts. 3. Decide heuristic: Use defer immediately after lock; document lock ordering.\n</think> Use `defer mu.Unlock()` immediately after `mu.Lock()` so the mutex is always released, even on early returns or panics. For complex code, document the lock acquisition order to prevent circular dependencies. Go's `sync.Mutex` is not reentrant — the same goroutine locking twice will deadlock."},
			{Sequence: 3, Role: "user", Content: "That threw a runtime deadlock error. How do I fix it?"},
			{Sequence: 4, Role: "assistant", Content: "[TRIPLETS] Subject: Runtime -> Action: Detect -> Object: Deadlock | Subject: Goroutine -> Action: Wait -> Object: Mutex\n</think> 1. Identify core task: Fix runtime deadlock. 2. Evaluate options: Restructure, channel-based coordination, context timeouts. 3. Decide heuristic: Redesign to avoid nested locking; use channels if possible.\n</think> Go's `sync.Mutex` is not recursive — if the same goroutine tries to lock it twice, it deadlocks. Refactor so each goroutine only holds one lock at a time. If you need reentrancy, switch to a channel-based state machine or use `sync.RWMutex` for read-heavy workloads."},
			{Sequence: 5, Role: "user", Content: "Can you show me a safe pattern for a concurrent counter?"},
			{Sequence: 6, Role: "assistant", Content: "[TRIPLETS] Subject: Counter -> Action: Increment -> Object: Atomic | Subject: Goroutine -> Action: Update -> Object: State\n</think> 1. Identify core task: Safe concurrent counter. 2. Evaluate options: Mutex vs atomic vs channel. 3. Decide heuristic: sync/atomic is fastest for simple counters; mutex is clearer for complex state.\n</think> For a simple counter, use `sync/atomic`:\n\n```go\nvar count int64\nfunc increment() {\n    atomic.AddInt64(&count, 1)\n}\n```\n\nFor complex state, use a mutex with a single goroutine owner or a struct with `defer mu.Unlock()` after every `mu.Lock()`."},
		},
	}
}

func generatePipelineBackpressureScenario() Conversation {
	return Conversation{
		ID:      "conv_synth_pipeline_01",
		Problem: "Backpressure handling in streaming pipelines",
		Tags:    []string{"pipeline", "backpressure", "channel", "streaming"},
		Turns: []Turn{
			{Sequence: 1, Role: "user", Content: "My Go pipeline slows to a crawl when the downstream consumer is slow. How do I handle backpressure?"},
			{Sequence: 2, Role: "assistant", Content: "[TRIPLETS] Subject: Pipeline -> Action: Slow -> Object: Throughput | Subject: Channel -> Action: Block -> Object: Sender\n</think> 1. Identify core task: Handle backpressure in pipeline. 2. Evaluate options: Buffered channels, bounded channels, drop policy. 3. Decide heuristic: Use buffered channels for burst absorption; limit buffer size to control memory.\n</think> Backpressure is normal in Go pipelines — unbuffered channels naturally apply backpressure by blocking the sender until the receiver is ready. For bursty workloads, use a small buffered channel (e.g., `make(chan T, 16)`). If memory is a concern, implement a drop-or-batch policy in the producer."},
			{Sequence: 3, Role: "user", Content: "What if I want to drop old data instead of blocking when the buffer is full?"},
			{Sequence: 4, Role: "assistant", Content: "[TRIPLETS] Subject: Channel -> Action: Drop -> Object: Value | Subject: Producer -> Action: Skip -> Object: Item\n</think> 1. Identify core task: Drop-oldest strategy. 2. Evaluate options: Select with default, ring buffer, skip-on-full. 3. Decide heuristic: Use select with default for non-blocking send; or implement a ring buffer for controlled dropping.\n</think> Use `select` with a `default` case for non-blocking sends:\n\n```go\nselect {\ncase ch <- val:\n    // sent\ndefault:\n    // drop oldest or skip\n    <-ch // drain oldest\n    ch <- val\n}\n```\n\nFor more control, implement a ring buffer (circular buffer) that overwrites the oldest element when full. This gives you bounded memory with no blocking."},
		},
	}
}

func generateAtomicCountersScenario() Conversation {
	return Conversation{
		ID:      "conv_synth_atomic_01",
		Problem: "Lock-free programming with atomic operations",
		Tags:    []string{"atomic", "lock-free", "counter", "sync/atomic"},
		Turns: []Turn{
			{Sequence: 1, Role: "user", Content: "When should I use sync/atomic instead of a mutex for counters?"},
			{Sequence: 2, Role: "assistant", Content: "[TRIPLETS] Subject: sync/atomic -> Action: Increment -> Object: Counter | Subject: Mutex -> Action: Protect -> Object: State\n</think> 1. Identify core task: Choose atomic vs mutex. 2. Evaluate options: atomic for simple ops, mutex for complex state. 3. Decide heuristic: Use atomic for single-variable counters; mutex for multi-variable invariants.\n</think> Use `sync/atomic` for simple counters and flags where a single operation is sufficient. Use `sync.Mutex` when you need to read-modify-write multiple variables atomically. Atomic operations are faster but limited to single-word values."},
			{Sequence: 3, Role: "user", Content: "Can I use atomics for more complex state transitions?"},
			{Sequence: 4, Role: "assistant", Content: "[TRIPLETS] Subject: Atomic -> Action: Transition -> Object: State | Subject: CAS -> Action: Compare -> Object: Value\n</think> 1. Identify core task: Complex state with atomics. 2. Evaluate options: CAS loops, state machine, mutex. 3. Decide heuristic: CAS works for simple state machines; use mutex for complex transitions.\n</think> For complex state transitions, use Compare-And-Swap (CAS) loops:\n\n```go\nfor {\n    old := atomic.LoadInt32(&state)\n    new := transition(old)\n    if atomic.CompareAndSwapInt32(&state, old, new) {\n        break\n    }\n}\n```\n\nBut CAS loops can suffer from livelock under high contention. If your state machine has more than 2-3 states or multiple fields, use a mutex instead."},
		},
	}
}

func generateSelectStarvationScenario() Conversation {
	return Conversation{
		ID:      "conv_synth_select_01",
		Problem: "Select statement starvation and fairness",
		Tags:    []string{"select", "channel", "starvation", "fairness"},
		Turns: []Turn{
			{Sequence: 1, Role: "user", Content: "My select statement seems to starve one of the cases. How does Go choose which case to execute?"},
			{Sequence: 2, Role: "assistant", Content: "[TRIPLETS] Subject: Select -> Action: Choose -> Object: Case | Subject: Runtime -> Action: Randomize -> Object: Order\n</think> 1. Identify core task: Understand select fairness. 2. Evaluate options: Random selection, pseudo-random, deterministic. 3. Decide heuristic: Go randomizes ready cases; if one case is always ready, it may starve others.\n</think> Go's `select` pseudo-randomly chooses among ready cases. If one case is always ready (e.g., a default case or a channel with buffered data), it may starve others. To ensure fairness, make sure all cases have similar readiness probabilities, or use a `default` case sparingly."},
			{Sequence: 3, Role: "user", Content: "How can I implement fair round-robin across multiple channels?"},
			{Sequence: 4, Role: "assistant", Content: "[TRIPLETS] Subject: Select -> Action: Round-robin -> Object: Channel | Subject: Runtime -> Action: Serve -> Object: Request\n</think> 1. Identify core task: Fair channel serving. 2. Evaluate options: Manual round-robin, timer-based, weighted fair queueing. 3. Decide heuristic: Use a ticker or manual index to rotate through channels.\n</think> Implement manual round-robin by tracking the last served channel:\n\n```go\nvar last int\nfor {\n    select {\n    case <-channels[last]:\n        // handle\n    default:\n    }\n    last = (last + 1) % len(channels)\n}\n```\n\nFor production, use a `time.Ticker` to periodically rotate and prevent starvation of slow channels."},
		},
	}
}

func countTurns(scenarios []Conversation) int {
	total := 0
	for _, s := range scenarios {
		total += len(s.Turns)
	}
	return total
}

func callOllamaForScenarios() ([]Conversation, error) {
	prompt := `Generate 3 Go debugging multi-turn conversation scenarios. Each scenario should have 4-6 turns (user/assistant alternating).

Format as JSON array:
[{
  "conversation_id": "conv_synth_xxx",
  "problem": "short description",
  "tags": ["tag1", "tag2"],
  "turns": [
    {"turn_sequence": 1, "role": "user", "content": "..."},
    {"turn_sequence": 2, "role": "assistant", "content": "..."}
  ]
}]

Topics: race conditions, GC tuning, channel lifecycle, context cancellation, mutex recovery, pipeline backpressure, atomic operations, select starvation.

Assistant answers MUST include [TRIPLETS], </think>, and </think> blocks.`

	payload := map[string]interface{}{
		"model":  "qwen2.5:3b",
		"prompt": prompt,
		"stream": false,
	}

	payloadBytes, _ := json.Marshal(payload)

	client := &http.Client{Timeout: 120 * time.Second}
	resp, err := client.Post("http://127.0.0.1:11434/api/generate", "application/json", strings.NewReader(string(payloadBytes)))
	if err != nil {
		return nil, fmt.Errorf("ollama connection failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("ollama read error: %w", err)
	}

	var ollamaResp struct {
		Response string `json:"response"`
	}
	if err := json.Unmarshal(body, &ollamaResp); err != nil {
		return nil, fmt.Errorf("ollama JSON parse error: %w", err)
	}

	var scenarios []Conversation
	if err := json.Unmarshal([]byte(ollamaResp.Response), &scenarios); err != nil {
		return nil, fmt.Errorf("failed to parse scenarios JSON: %w", err)
	}

	return scenarios, nil
}
