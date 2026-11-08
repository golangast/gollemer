package chat

// The cookbook expansion of the Go knowledge base: entries taught
// from the Go Cookbook (go-cookbook.com/snippets) and the official
// testing package documentation, plus per-release version
// summaries verified against Anton Zhiyanov's Go features index
// (antonz.org/go-features), rewritten in this book's own
// beginner words — what a topic is, the standard pattern, when you
// would use it, and the pitfall that bites. The authorship rule is
// the book's: topics and key points are the source's; the words are
// ours. Version-specific topics carry their version in the body
// ("New in Go 1.27: ...") so the chat never teaches a future API as
// today's default, and topics that need a third-party package say
// so plainly. Retrieval, the gated claim, and the body discipline
// (short prose, no code blocks) are exactly the base book's — see
// goknowledge.go. init() appends these to goKnowledge so every
// consumer of the book sees one list; this file's order only breaks
// exact score ties, and the reachability tests pin every entry
// against the full merged book.

var goKnowledgeCookbook = []goKnowledgeEntry{
	{
		title:    "base64 and hex encoding",
		keywords: []string{"base64 encoding", "hex encoding", "encoding/base64", "encoding/hex", "encode binary data", "base64"},
		body:     "Encoding turns binary data into text so it can travel through text-only channels. Use encoding/base64: base64.StdEncoding.EncodeToString(data) encodes and DecodeString reverses it, always checking the decode error. For values inside URLs or filenames pick base64.URLEncoding instead. encoding/hex does the same job with hex.EncodeToString, giving longer but human-readable output that suits digests, hashes and debugging.",
	},
	{
		title:    "built-in functions",
		keywords: []string{"built-in functions", "builtin functions", "built in functions", "len and cap", "make and new", "predeclared functions"},
		body:     "Built-in functions come with the language itself, so they need no import and work across many types. The everyday ones are len and cap for sizes, append and copy for slices, make for creating slices, maps and channels, new for allocating a zeroed value, delete for removing a map key, and close for channels. Use them before reaching for a package.",
	},
	{
		title:    "environment variables",
		keywords: []string{"environment variables", "environment variable", "os.Getenv", "os.LookupEnv", "os.Setenv", "read environment variables"},
		body:     "Environment variables pass configuration like ports, hosts and secrets into a program from outside the code. Read one with os.Getenv(key), which returns an empty string when it is missing. Use os.LookupEnv(key) when you must tell unset apart from set-but-empty: it also returns a boolean. os.Setenv changes a value for the current process only, never for the parent shell.",
	},
	{
		title:    "generic methods",
		keywords: []string{"generic methods", "generic method", "method type parameters", "methods with type parameters", "generic methods go 1.27"},
		body:     "New in Go 1.27: a method can declare its own type parameters, separate from its receiver, like func (s Stream[T]) Map[U any](convert func(T) U) Stream[U]. Before 1.27 that operation had to be a package-level generic function. Generic methods cannot satisfy an interface, and your module must require Go 1.27 or later, so libraries supporting older releases should keep the function form.",
	},
	{
		title:    "input validation",
		keywords: []string{"input validation", "validate input", "validating input", "data validation", "validator package", "validate user input"},
		body:     "Input validation checks data arriving from users or other systems before your program trusts it: required fields present, numbers in range, text matching an expected shape. In the standard library, regexp patterns and manual checks with clear error returns cover simple cases, always validated server-side, never only in the browser. For struct-heavy rules, the external go-playground/validator package adds validate tags; the standard library has no equivalent.",
	},
	{
		title:    "reading user input",
		keywords: []string{"reading user input", "read user input", "bufio.Reader", "fmt.Scanln", "console input", "stdin input"},
		body:     "For a full line of console input, wrap os.Stdin in a buffered reader made with bufio.NewReader(os.Stdin), read with ReadString, then clean the result with strings.TrimSpace and check the error. For one or two simple whitespace-separated values, fmt.Scanln into variables is shorter, but it stops at spaces. Convert typed numbers afterwards with strconv.Atoi or strconv.ParseFloat, handling their errors instead of trusting the text.",
	},
	{
		title:    "reflect iterators",
		keywords: []string{"reflect iterators", "reflect iterator", "reflect.Type Fields", "reflect.Value Fields", "iterate struct fields", "reflection metadata", "fields with reflect", "reflect"},
		body:     "In Go 1.26, reflect gained iterator methods that replace counting and indexing loops over type metadata. Range over t.Fields() on a reflect.Type or reflect.Value to visit struct fields in declaration order, and use Methods, Ins and Outs for method and function parameter lists. They panic on the wrong kind just like the indexed forms, so check Kind first, and guard unexported fields.",
	},
	{
		title:    "reflect package",
		keywords: []string{"reflect package", "reflection", "reflect.TypeOf", "reflect.ValueOf", "inspect types at runtime", "runtime reflection", "runtime type inspection", "type at runtime", "runtime"},
		body:     "The reflect package lets a program inspect and work with types it did not name at compile time, using reflect.TypeOf for the type and reflect.ValueOf for the value. You can list struct fields, read tags and call methods dynamically. To change a value you must pass a pointer and check CanSet first. Prefer generics or interfaces when they fit; reflection is slower and panics on misuse.",
	},
	{
		title:    "struct tags",
		keywords: []string{"struct tags", "struct field tags", "json tags", "struct annotations", "field tags", "omitempty"},
		body:     "Struct tags are the quoted strings after a field, like a json key name together with omitempty, that tell libraries how to handle that field. encoding/json uses them to rename keys and skip empty values; other packages read their own keys, such as db or validate. Only exported fields are seen by these libraries, a misspelled tag key is silently ignored, and backtick-quoted raw strings are the usual way to write tags.",
	},
	{
		title:    "new in go 1.25",
		keywords: []string{"new in go 1.25", "go 1.25", "go 1.25 features", "what is new in go 1.25", "go 1.25 release"},
		body:     "Go 1.25 is a past release whose features are part of today's defaults. Its highlights include container-aware GOMAXPROCS, which sizes the runtime to CPU limits inside containers, http.CrossOriginProtection for guarding requests against cross-site forgery, and the runtime trace FlightRecorder, which keeps the last seconds of execution trace ready for debugging production problems. It also made testing/synctest's fake clock stable and added WaitGroup.Go, which launches a tracked goroutine without manual Add and Done bookkeeping.",
	},
	{
		title:    "new in go 1.26",
		keywords: []string{"new in go 1.26", "go 1.26", "go 1.26 features", "what is new in go 1.26", "go 1.26 release"},
		body:     "Go 1.26 is the release Gollemer builds with, so its features are today's defaults. Highlights are errors.AsType for type-safe error matching without a target pointer, new(expression) returning a pointer to any value for optional fields, reflect iterator methods that replace index loops, and go fix for applying safe modernizations. Its simd/archsimd, runtime/secret, and goroutine leak profile are experimental, not settled APIs.",
	},
	{
		title:    "new in go 1.27",
		keywords: []string{"new in go 1.27", "go 1.27", "go 1.27 features", "what is new in go 1.27", "go 1.27 release"},
		body:     "New in Go 1.27: this is the next release after the Go 1.26 toolchain Gollemer builds with, so treat its features as coming, not today's defaults. The headline items are generic methods, which let a method declare its own type parameters, encoding/json/v2, a standard uuid package, and crypto/mldsa for post-quantum signatures, plus a stable goroutine leak profile, strings.CutLast, maphash hashers, and an experimental portable simd package. Code using them needs a Go 1.27 module.",
	},
	{
		title:    "character escaping",
		keywords: []string{"character escaping", "escape sequences", "escape a quote", "raw strings", "raw string literals", "backslash escapes", "escape", "escaping", "quote", "go strings"},
		body:     "Double-quoted strings can hold special characters through escape sequences: \\\" for a quote inside, \\\\ for a backslash, \\n for a newline, and \\t for a tab. Raw strings written with backticks take every character literally, with no escapes processed, which suits multiline text and text full of backslashes or quotes. A raw string cannot itself contain a backtick.",
	},
	{
		title:    "cutting strings with CutLast",
		keywords: []string{"strings.CutLast", "bytes.CutLast", "CutLast", "split on last separator", "last separator", "cut last"},
		body:     "New in Go 1.27: strings.CutLast splits a string around the last occurrence of a separator, returning the part before it, the part after it, and a boolean saying whether the separator was found. bytes.CutLast does the same for byte slices. Use it for image tags, file extensions, and host ports where earlier separators belong to the value. Always check the boolean.",
	},
	{
		title:    "string building",
		keywords: []string{"string building", "strings.Builder", "build a string", "concatenate efficiently", "concatenate strings"},
		body:     "Strings are immutable, so adding pieces with + in a loop creates a new string every time and wastes memory. Use strings.Builder instead: declare a Builder, add pieces with WriteString, pre-size it with Grow when you know roughly how long the result will be, and finish with String. For a ready-made slice of pieces, strings.Join glues them together with one separator.",
	},
	{
		title:    "string comparison",
		keywords: []string{"string comparison", "compare strings", "strings.Compare", "strings.EqualFold", "case-insensitive comparison", "equal strings"},
		body:     "Compare strings with == and != for exact equality, which checks the bytes in order. When letter case should not matter, strings.EqualFold compares two strings ignoring case without allocating lowercase copies. When you need ordering rather than equality, strings.Compare returns 0 for equal strings, a negative number when the first sorts earlier, and a positive number when it sorts later.",
	},
	{
		title:    "stringer interface",
		keywords: []string{"Stringer interface", "fmt.Stringer", "String method", "stringer", "custom string representation", "prints", "type prints", "customize how"},
		body:     "fmt.Stringer is a tiny interface holding one method, String() string. When your type defines that method, the fmt package calls it automatically whenever the value is printed with verbs like %v or %s, so you control how the value reads. Keep String simple and cheap, because logging and debugging may call it often, and never let it panic on a zero value.",
	},
	{
		title:    "uuid package",
		keywords: []string{"uuid package", "uuid", "generate uuid", "uuid.NewV4", "uuid.NewV7", "uuid.Parse", "unique identifier"},
		body:     "New in Go 1.27: the standard library gains a uuid package for these 16-byte unique identifiers, so you no longer need an external package. uuid.NewV4 makes a random identifier, uuid.NewV7 makes one that sorts roughly by creation time, uuid.New returns a version 4 value, and uuid.Parse validates text form. The zero value is all zero bytes, so decide whether that counts as a real identifier.",
	},
	{
		title:    "bloom filters",
		keywords: []string{"bloom filter", "bloom filters", "false positives", "membership test", "probabilistic data structure", "bits-and-blooms"},
		body:     "A bloom filter is a small, fast membership tester built on a bit array plus several hash functions. Adding a value sets its bits, and asking about a value checks them: a no answer is certain, but a yes answer can occasionally be a false positive, and values cannot be removed. It needs the external bits-and-blooms bloom package; the standard library has no bloom filter.",
	},
	{
		title:    "container package",
		keywords: []string{"container package", "container/heap", "container/list", "container/ring", "container ring", "data structures package"},
		body:     "The container package groups three ready-made data structures in subpackages. container/heap turns a slice into a heap when your type implements its interface, container/list provides a doubly linked list, and container/ring provides a circular list of fixed size. Reach for them when a plain slice does not fit, but remember slices stay simpler and usually faster for ordinary ordered data.",
	},
	{
		title:    "custom hashing with maphash",
		keywords: []string{"maphash.Hasher", "custom hashing", "custom equality", "hashing keys"},
		body:     "New in Go 1.27: maphash.Hasher lets you define custom hashing together with custom equality for keys a normal map cannot treat your way, such as keys that should match after lowercasing, or types that are not comparable. Equal values must always hash equally under the same seed, and buckets still check equality because different values can collide. Never save these hashes outside the process.",
	},
	{
		title:    "double-ended queues",
		keywords: []string{"deque", "deques", "double-ended queue", "double ended queue", "queue both ends", "front and back queue", "a deque"},
		body:     "A deque, or double-ended queue, lets you add and remove items at both the front and the back, unlike a queue which only takes from the front. Go has no built-in deque, so you build one from a slice or with container/list, or use a ring buffer for heavy use. Adding at the front of a plain slice copies every item, so it gets slow as the deque grows.",
	},
	{
		title:    "graphs",
		keywords: []string{"graph", "graphs", "graph data structure", "adjacency list", "adjacency matrix", "vertices and edges", "a graph", "represent a graph"},
		body:     "A graph is a set of vertices connected by edges, used to model things like road networks, friendships, and dependencies. Go has no built-in graph type, so you build one yourself: an adjacency list stores each vertex with a slice of its neighbours in a map, while an adjacency matrix stores a grid of connections. Lists suit sparse graphs; matrices suit dense ones but waste memory on empty connections.",
	},
	{
		title:    "heaps",
		keywords: []string{"heap", "heaps", "container/heap", "heap.Interface", "min-heap", "heap.Push"},
		body:     "A heap keeps the smallest (or largest) value at the top, so reading the next priority item is cheap while adding and removing cost a little reordering. Go provides container/heap, but it works on your own slice type: you implement heap.Interface with Len, Less, Swap, Push, and Pop, call heap.Init once, then use heap.Push and heap.Pop. Forgetting heap.Init leaves the order wrong.",
	},
	{
		title:    "linked lists",
		keywords: []string{"linked list", "linked lists", "container/list", "doubly linked list", "list.PushBack"},
		body:     "A linked list is a chain of nodes where each node holds a value and a pointer to the next node, so inserting or removing in the middle does not shift other items. container/list gives you a doubly linked list with PushBack, PushFront, and Remove; walk it from Front following Next. Slices are still faster for most work because list nodes are scattered in memory.",
	},
	{
		title:    "maps package",
		keywords: []string{"maps package", "maps utilities", "clone a map", "merge maps", "copy a map", "compare maps", "map keys", "cloning", "clone", "maps"},
		body:     "The maps package holds ready-made helpers for working with maps. maps.Clone copies a map, maps.Copy merges one map into another overwriting matching keys, and maps.Equal reports whether two maps hold the same keys and values. maps.Keys and maps.Values hand back sequences you can loop over or collect into a slice. Cloning is shallow: nested maps and slices inside are still shared.",
	},
	{
		title:    "priority queues",
		keywords: []string{"priority queue", "priority queues", "container/heap", "heap.Fix", "highest priority first", "priority queue heap"},
		body:     "A priority queue hands back the most important item first instead of the oldest, which suits task schedulers and event handling. Go has no built-in one; you build it on container/heap by storing items with a priority field, ranking them in Less, and letting heap.Push and heap.Pop maintain the order. When an item's priority changes in place, call heap.Fix so the queue reorders it.",
	},
	{
		title:    "queues",
		keywords: []string{"queue", "queues", "fifo queue", "first in first out", "enqueue dequeue", "implement a queue", "a queue"},
		body:     "A queue is a first-in, first-out line: you add items at the back with enqueue and take them from the front with dequeue, like people waiting at a counter. Go has no built-in queue, so a common version wraps a slice, appending at the back and slicing off the front. Buffered channels and container/list also work as queues, especially when goroutines pass work between them.",
	},
	{
		title:    "sets",
		keywords: []string{"set", "sets", "set data structure", "implement a set", "unique values", "map set", "a set", "make a set"},
		body:     "A set stores unique values with no duplicates and answers membership questions quickly. Go has no built-in set type, so the usual trick is a map with empty struct values: map[string]struct{}. Adding sets the key, delete removes it, and a two-value lookup tests membership, all in constant time on average. Slice-based sets work too, but every membership check scans the whole slice.",
	},
	{
		title:    "skip lists",
		keywords: []string{"skip list", "skip lists", "skip-list", "layered linked list", "fast search list"},
		body:     "A skip list is a linked list with extra express lanes: some nodes also point further ahead on higher levels, so searching can jump over many items, giving fast search, insertion, and deletion without balancing a tree. Go has no built-in skip list and the standard library offers none, so you implement one yourself with nodes holding slices of next pointers and randomly chosen levels.",
	},
	{
		title:    "stacks",
		keywords: []string{"stack", "stacks", "lifo", "last in first out", "push and pop", "implement a stack", "a stack", "stack data structure"},
		body:     "A stack is a last-in, first-out pile: you push items onto the top and pop them off the top, like a stack of plates, which suits undo features, parsing, and backtracking. Go has no built-in stack, so wrap a slice: push with append, and pop by reading the last item and reslicing, after checking the stack is not empty so an empty pop cannot panic.",
	},
	{
		title:    "tries",
		keywords: []string{"trie", "tries", "prefix tree", "prefix search", "autocomplete", "prefix trie", "a trie"},
		body:     "A trie, also called a prefix tree, stores strings character by character in shared branches, so words with a common beginning share nodes. That makes prefix questions fast, like listing every word starting with what a user has typed for autocomplete. Go has no built-in trie, so you build nodes holding a map from rune to child node plus a marker that a complete word ends there.",
	},
	{
		title:    "atomic types",
		keywords: []string{"atomic types", "sync/atomic", "atomic", "atomic.Int64", "atomic.Bool", "atomic.Value", "compare and swap", "lock-free"},
		body:     "Atomic types from sync/atomic give lock-free operations on one shared value that many goroutines can use safely at the same time. Since Go 1.19 the package offers typed values like atomic.Int64, atomic.Bool, and atomic.Pointer, plus atomic.Value for swapping a whole value, each with Load, Store, Add, and CompareAndSwap methods. Prefer them for simple counters and flags; use a mutex when several fields must change together, and never mix atomic and plain access to the same value.",
	},
	{
		title:    "sync.Map",
		keywords: []string{"sync.Map", "concurrent map", "concurrent maps", "concurrent map access", "thread-safe map", "use sync.Map", "map safely", "safely"},
		body:     "Plain maps are not safe when several goroutines read and write them at once: the runtime can panic and data races corrupt results. Guard a shared map with sync.Mutex, or sync.RWMutex when reads dominate, or use sync.Map, a map type built for concurrent use with Store, Load, Delete, and Range methods. sync.Map suits read-heavy or disjoint-key workloads, but it stores values as any, so you lose compile-time type safety.",
	},
	{
		title:    "go memory model",
		keywords: []string{"go memory model", "memory model", "happens before", "happens-before", "memory visibility", "visibility", "synchronization"},
		body:     "The Go memory model is the set of rules that says when a write by one goroutine is guaranteed to be visible to another. A channel send and receive, a mutex lock and unlock, and other sync operations create happens-before edges that carry that guarantee. Without such an edge there is no promise, even if timing makes code look correct, so always synchronize shared data and let go test -race check your work.",
	},
	{
		title:    "graceful shutdown",
		keywords: []string{"graceful shutdown", "shutdown", "signal.NotifyContext", "server shutdown", "shut down gracefully", "http.Server.Shutdown", "drain connections", "shut down"},
		body:     "Graceful shutdown means stopping cleanly: stop accepting new work, let in-flight work finish, close resources, then exit. Run every long-lived component — server and workers — under one errgroup and a context canceled by a termination signal (signal.NotifyContext), so one failure cancels the rest. Derive the bounded shutdown context from context.Background(), not the canceled one; call http.Server.Shutdown with it to drain connections, and treat http.ErrServerClosed as a clean stop, not an error.",
	},
	{
		title:    "handling os signals",
		keywords: []string{"os signals", "handling signals", "signal handling", "os/signal", "signal.Notify", "SIGINT", "SIGTERM", "signal.NotifyContext"},
		body:     "Programs receive operating system signals such as SIGINT from Ctrl+C and SIGTERM from the system asking them to stop. The os/signal package delivers them on a channel with signal.Notify, or as a canceled context with signal.NotifyContext, so your program can clean up instead of dying instantly. Use a buffered channel of size one, and call the returned stop function when you finish so default behavior is restored.",
	},
	{
		title:    "race conditions",
		keywords: []string{"race condition", "race conditions", "race detector", "detect race conditions", "concurrent access", "unsynchronized", "race", "condition", "race condition in go"},
		body:     "A race condition happens when two or more goroutines touch the same data at the same time, at least one of them writes, and nothing synchronizes them, so the result depends on timing and can be wrong or corrupt. Go ships a race detector: run go test -race or go run -race to print both conflicting accesses when a race actually happens. Fix races by guarding data with a mutex, using atomic types, or passing values over channels.",
	},
	{
		title:    "semaphores",
		keywords: []string{"semaphore", "semaphores", "semaphore.Weighted", "limit concurrency", "concurrency limit", "limiting goroutines", "acquire and release"},
		body:     "A semaphore limits how many goroutines may use a resource at once, like allowing only five downloads at a time. Go has no semaphore in the standard library: the simple pattern is a buffered channel filled with tokens, where sending acquires a slot and receiving releases it. For cancellation-aware or weighted slots, use semaphore.Weighted from the external golang.org/x/sync/semaphore package. Always release every slot you acquire, or the program eventually stalls.",
	},
	{
		title:    "errgroup",
		keywords: []string{"errgroup", "structured concurrency", "first error", "cancel on error", "run goroutines with errors", "run goroutines", "error"},
		body:     "errgroup runs a group of related goroutines and reports failure properly, which a plain WaitGroup cannot do. It needs the external golang.org/x/sync/errgroup package. You launch tasks with Group.Go, and Wait returns the first error any task produced. With errgroup.WithContext, that first error also cancels a context the other tasks can stop on, and SetLimit caps how many tasks run at once. Tasks that ignore the context keep running anyway.",
	},
	{
		title:    "custom error types",
		keywords: []string{"custom error types", "custom errors", "error type", "structured errors", "Error() method", "errors.As"},
		body:     "A custom error type is your own struct (or other named type) that carries extra facts about a failure, like a code or the field that failed. Give it an Error() string method and it satisfies the error interface, so functions can return it like any error. Callers pull the details back out with errors.As. Use one when a plain message is not enough.",
	},
	{
		title:    "joining multiple errors",
		keywords: []string{"errors.Join", "join errors", "multiple errors", "joining errors", "combine errors", "multi-error"},
		body:     "errors.Join combines several errors into one, for jobs that can fail in several independent ways, like checking many form fields at once. Collect failures in a slice and return errors.Join(errs...): nil values drop out, and if every error is nil the result is nil. Callers can still find each cause inside with errors.Is and errors.As. Since Go 1.20.",
	},
	{
		title:    "matching error types with errors.AsType",
		keywords: []string{"errors.AsType", "AsType", "error matching", "typed error matching", "extract error type", "typed error", "extract a typed error"},
		body:     "New in Go 1.26: errors.AsType is the generic version of errors.As for pulling a specific error type out of a wrapped chain. Write pathErr, ok := errors.AsType[*fs.PathError](err) and use the boolean: on a match you get the typed error directly, with no separate target variable to declare first. Keep using errors.Is when you only need to compare against one sentinel value.",
	},
	{
		title:    "cloning urls and query values",
		keywords: []string{"URL.Clone", "Values.Clone", "clone url", "cloning urls", "query values", "copy a url"},
		body:     "New in Go 1.27: net/url gains URL.Clone and Values.Clone, which make a deep copy of a parsed URL or its query values. Clone the base URL, then change the copy with Query() and Set without touching the original — handy for building the next page link from the current request. On older Go versions, copy the struct and its User and query values by hand.",
	},
	{
		title:    "http connection reuse",
		keywords: []string{"connection reuse", "reuse connections", "http.Client", "http.Transport", "response body", "close the body", "keep-alive"},
		body:     "An http.Client reuses its underlying connections, so create one client (and one http.Transport) and share it instead of building a new one per request. Reuse only works when you finish with each response properly: read the body to the end when you need it, and always close resp.Body, even on errors. Go 1.27 drains a little unread body on close, but closing stays your job.",
	},
	{
		title:    "gob encoding",
		keywords: []string{"gob", "encoding/gob", "gob encoding", "binary serialization", "serialize", "serialization"},
		body:     "gob is the standard library's binary format for turning Go values into bytes and back, made for Go programs talking to other Go programs. Create a gob.NewEncoder(w) over any writer and call Encode(v); a gob.NewDecoder(r) with Decode(&v) reverses it. Only exported fields travel, and values stored in interface fields need gob.Register first. For other languages or humans, pick JSON instead.",
	},
	{
		title:    "http header limits",
		keywords: []string{"header limits", "MaxHeaderValueCount", "MaxHeaderBytes", "too many headers", "http/2 priorities", "DisableClientPriority", "limit http headers", "http headers", "headers"},
		body:     "New in Go 1.27: http.Server gains MaxHeaderValueCount, a cap on how many header values a request may carry, alongside the older MaxHeaderBytes cap on total header size. Zero does not mean unlimited — it selects a built-in default. The same release lets HTTP/2 servers honor client priority hints unless you set DisableClientPriority; priorities only order work, they never replace limits or rate limiting.",
	},

	{
		title:    "encoding/json v2",
		keywords: []string{"json v2", "encoding/json/v2", "jsonv2", "GOEXPERIMENT", "json options", "stricter json"},
		body:     "New in Go 1.27: encoding/json/v2 ships as a standard package (in Go 1.25 and 1.26 it was experimental, switched on with GOEXPERIMENT=jsonv2). Its functions take options and are stricter by default: duplicate object names and invalid UTF-8 are rejected instead of silently accepted. The classic encoding/json keeps working and is backed by v2, so migrate one boundary at a time and test edge cases.",
	},
	{
		title:    "http client requests",
		keywords: []string{"http client", "http.Get", "http.Post", "make http request", "call an api", "go http client", "http get"},
		body:     "For quick calls, http.Get(url) and http.Post(url, contentType, body) fetch a page or send data. When you need headers or a method like PUT, build the request with http.NewRequest, set req.Header values, and send it with client.Do(req). Always defer resp.Body.Close() right after a successful call, then read the answer with io.ReadAll(resp.Body) and check resp.StatusCode.",
	},
	{
		title:    "production http servers",
		keywords: []string{"production server", "http.Server", "server timeouts", "MaxBytesReader", "ReadHeaderTimeout", "production", "production ready"},
		body:     "http.ListenAndServe hides every policy choice, so a production service builds its own http.Server with an Addr, its own http.NewServeMux, and timeouts — ReadHeaderTimeout at minimum. Cap each request body with http.MaxBytesReader so one huge upload cannot eat memory. On a shutdown signal, call server.Shutdown(ctx): it stops new connections and waits for in-flight requests to finish instead of cutting them off.",
	},
	{
		title:    "protocol buffers",
		keywords: []string{"protocol buffers", "protobuf", "protoc", "proto.Marshal", "proto file", "service serialization", "use protobuf"},
		body:     "Protocol buffers (protobuf) serialize structured data to compact binary using a schema you write in a .proto file. The protoc compiler and its Go plugin generate structs from that schema; the external google.golang.org/protobuf package then provides proto.Marshal and proto.Unmarshal. Field numbers are the real contract — never delete or reuse one. Protobuf beats JSON between services, but needs that extra tooling and is not human-readable.",
	},
	{
		title:    "rate limiting",
		keywords: []string{"rate limiting", "rate limiter", "rate.NewLimiter", "token bucket", "limit requests", "429", "Retry-After", "rate limit", "limit http requests"},
		body:     "Rate limiting caps how many requests a client may make, protecting a server from overload. The golang.org/x/time/rate package implements a token bucket: rate.NewLimiter(rate.Limit(10), 4) refills ten tokens a second and holds bursts of four. Middleware calls limiter.Allow() and answers 429 with a Retry-After header when the bucket is empty; give each client its own limiter for fairness, and use Wait to pace outgoing calls.",
	},
	{
		title:    "http routing",
		keywords: []string{"http routing", "routing requests", "http.NewServeMux", "ServeMux", "PathValue", "path parameters", "method patterns", "http.ServeMux"},
		body:     "Since Go 1.22, http.NewServeMux routes with patterns that include the method and capture path pieces: mux.HandleFunc(\"GET /users/{id}\", h) matches only GET requests, and the handler reads the captured piece with r.PathValue(\"id\"). A {path...} pattern swallows the rest of the path. Middleware is just a function wrapping an http.Handler, and a second mux mounted with http.StripPrefix groups routes under one prefix.",
	},
	{
		title:    "serving static files",
		keywords: []string{"serve static files", "static files", "http.FileServer", "file server", "http.StripPrefix", "http.FileServerFS", "static content"},
		body:     "http.FileServer(http.Dir(\"./static\")) serves the files in a folder — HTML, CSS, JavaScript, images — straight from disk. Mount it under a prefix with http.StripPrefix, as in mux.Handle(\"/static/\", http.StripPrefix(\"/static/\", fs)), so URL paths and folder paths line up. To ship the assets inside your binary instead, use embed with http.FileServerFS. Only place files you mean to publish in the served folder.",
	},
	{
		title:    "http request timeouts",
		keywords: []string{"request timeouts", "http timeout", "client.Timeout", "set timeout", "ResponseHeaderTimeout", "http.NewRequestWithContext", "timeout"},
		body:     "The default HTTP client waits forever, so a stuck server can hang your program. Give your client a whole-request budget with client := &http.Client{Timeout: 10 * time.Second}. For a deadline on just one call, attach context.WithTimeout through http.NewRequestWithContext instead. Finer phases live on http.Transport — ResponseHeaderTimeout, for example, caps how long you wait for the first response headers.",
	},
	{
		title:    "streaming json with jsontext",
		keywords: []string{"jsontext", "streaming json", "stream json", "large json", "incremental decoding", "json tokens", "encoding/json/jsontext", "stream"},
		body:     "New in Go 1.27: encoding/json/jsontext reads JSON as a stream of tokens instead of one whole value. Paired with encoding/json/v2, you can walk a huge array and decode one element at a time, so the full document never sits in memory. Streaming does not bound the input by itself — still cap request bodies, and copy any token or value you keep, because the decoder reuses its storage.",
	},
	{
		title:    "text and html templates",
		keywords: []string{"text templates", "text/template", "template.New", "generate text", "FuncMap", "plain text", "html templates", "html/template", "template.ParseFiles", "render html", "web page template", "template.FuncMap"},
		body:     "text/template generates plain text \u2014 emails, reports, config files \u2014 by filling {{.Name}}-style placeholders from a struct or map: template.New(\"report\").Parse(text), then Execute(w, data). Register helpers in a template.FuncMap before parsing, and parse once, not per use. Nothing is escaped for you. For web pages, html/template has the same API but auto-escapes output, so user content can't break the page or inject markup \u2014 prefer it for anything rendered in a browser.",
	},
	{
		title:    "dns lookups",
		keywords: []string{"dns", "dns lookup", "what is dns", "net.LookupIP", "net.LookupAddr", "resolve a domain", "reverse lookup"},
		body:     "DNS turns domain names into IP addresses. In the net package, net.LookupIP(name) returns all addresses for a host, and net.LookupAddr(ip) does the reverse lookup. For timeouts, use a Resolver method like LookupIPAddr with a context. Always check the error, and expect several addresses back, not just one.",
	},
	{
		title:    "ftp file transfer",
		keywords: []string{"ftp", "ftp server", "file transfer protocol", "goftp", "ftp client", "upload to ftp", "download from ftp", "transfer"},
		body:     "FTP moves files between your program and a remote server. Go has no FTP support in its standard library, so this needs the external github.com/goftp/client package. You dial with client.DialConfig and a Config holding the user and password, list with List, upload with Stor, and download with Retr. Prefer secure FTPS for sensitive files, and always close the connection.",
	},
	{
		title:    "grpc clients",
		keywords: []string{"grpc", "grpc client", "grpc.NewClient", "grpc stub", "protobuf"},
		body:     "gRPC clients call remote methods like local functions, using generated stubs. This needs the external google.golang.org/grpc package plus code generated from your .proto file. Create one shared connection with grpc.NewClient and reuse it, build a stub from it, and call methods with a context that has a timeout. Use TLS credentials in production; insecure credentials are only for local testing.",
	},
	{
		title:    "grpc servers",
		keywords: []string{"grpc", "grpc server", "grpc.NewServer", "protobuf", "remote procedure call"},
		body:     "A gRPC server provides remote methods that clients in any language can call. This needs the external google.golang.org/grpc package: define the service in a .proto file, generate Go code with protoc, and implement the methods on a struct embedding the generated Unimplemented server type. Register it with grpc.NewServer, listen on a TCP port with net.Listen, and Serve. Add TLS before exposing it publicly.",
	},
	{
		title:    "ntp time synchronization",
		keywords: []string{"ntp", "what is ntp", "ntp server", "network time protocol", "ntp.Time", "clock synchronization", "beevik"},
		body:     "NTP keeps computer clocks agreeing by asking a time server for the correct time. Go has no NTP client in its standard library, so this needs the external github.com/beevik/ntp package. ntp.Time(pool.ntp.org) returns the server time, and ntp.QueryWithOptions gives details like clock offset and round-trip delay. Compare the offset with your local clock instead of trusting it blindly, and set a timeout.",
	},
	{
		title:    "tls and ssl",
		keywords: []string{"tls", "ssl", "what is tls", "crypto/tls", "tls.Config", "tls server", "certificates"},
		body:     "TLS encrypts a network connection and proves who you are talking to; SSL is its old, deprecated name. The crypto/tls package does the work: servers load a certificate and key with tls.LoadX509KeyPair and serve through tls.Listen, clients connect with tls.Dial. Never set InsecureSkipVerify in production, because it turns off certificate checking and allows man-in-the-middle attacks.",
	},
	{
		title:    "tcp networking",
		keywords: []string{"tcp", "tcp server", "tcp client", "net.Listen", "net.Dial", "transmission control protocol"},
		body:     "TCP gives you a reliable, ordered stream of bytes between two programs. A server calls net.Listen, accepts connections with Accept, and handles each one in its own goroutine; a client connects with net.Dial. Build addresses with net.JoinHostPort, set deadlines with SetDeadline, and close connections when done. Remember reads can return fewer bytes than you asked for.",
	},
	{
		title:    "udp networking",
		keywords: []string{"udp", "udp packets", "udp server", "net.ListenUDP", "ReadFromUDP", "user datagram protocol"},
		body:     "UDP sends small independent packets called datagrams, with no connection, no delivery guarantee, and no ordering. Send with net.Dial on a udp network and Write; receive with net.ListenUDP and ReadFromUDP into a buffer, which also gives the sender address. Choose UDP when speed matters more than perfection, like games or DNS, and add your own checks if loss matters.",
	},
	{
		title:    "websockets",
		keywords: []string{"websocket", "websockets", "websocket server", "websocket client", "gorilla", "websocket.Upgrader"},
		body:     "WebSockets keep one connection open so browser and server can send messages to each other in real time, unlike plain HTTP request and response. Go needs the external gorilla/websocket package for this. An http handler upgrades the request with websocket.Upgrader, then both sides exchange messages with WriteMessage and ReadMessage. Always set CheckOrigin to only allow your own sites, and close connections when finished.",
	},
	{
		title:    "extracting tar and zip archives",
		keywords: []string{"archive/tar", "archive/zip", "tar archive", "zip archive", "zip file", "extract archive", "archive extraction"},
		body:     "archive/tar and archive/zip read tar and zip archives. Archive entry names are untrusted: a crafted name with .. or a symlink can write outside your target folder, the classic Zip Slip attack. Extract safely by opening the destination with os.OpenRoot (Since Go 1.24) and doing every write through that root, rejecting links and non-regular files, and capping file count and total unpacked size against zip bombs.",
	},
	{
		title:    "creating directories",
		keywords: []string{"create a directory", "creating directories", "os.Mkdir", "os.MkdirAll", "mkdir", "nested directories"},
		body:     "Use os.Mkdir to create one directory: os.Mkdir(name, 0755). It fails if the parent folder is missing or the directory already exists. Use os.MkdirAll to create a whole nested path at once; it creates every missing parent and succeeds silently if the path already exists. The 0755 permission means the owner can do everything and everyone else can read and enter the directory.",
	},
	{
		title:    "io/fs package",
		keywords: []string{"fs package", "fs.FS", "os.DirFS", "fs.Sub", "fs.WalkDir", "filesystem", "io/fs"},
		body:     "The io/fs package defines one small interface, fs.FS, for read-only filesystems, plus helpers that work on any of them: fs.ReadFile, fs.ReadDir, and fs.WalkDir. os.DirFS turns a real folder into an fs.FS, and fs.Sub narrows one to a subfolder. Writing code against fs.FS instead of os directly lets the same code read embedded files, test filesystems, or sandboxed roots unchanged.",
	},
	{
		title:    "listing directory files",
		keywords: []string{"os.ReadDir", "list files in a directory", "directory listing", "list directory", "list files"},
		body:     "os.ReadDir(folder) lists one directory and returns entries sorted by filename. Each entry gives Name and IsDir cheaply; call entry.Info only when you need size or modification time, because it costs an extra system call. To list a whole tree including subfolders, walk it with filepath.WalkDir instead. Always handle the error: unreadable folders and missing paths are common.",
	},
	{
		title:    "memory-mapped files",
		keywords: []string{"memory-mapped files", "memory mapped files", "mmap", "unix.Mmap", "map a file", "memory-mapped"},
		body:     "Memory mapping makes a file behave like a byte slice in memory: the operating system loads pages on demand instead of you calling Read. Go has no mmap in its standard library; use unix.Mmap from the external golang.org/x/sys/unix package, or the experimental read-only golang.org/x/exp/mmap wrapper. Always unmap when done, and never touch the slice afterwards, because that crashes the program.",
	},
	{
		title:    "os.Root sandboxed file access",
		keywords: []string{"os.Root", "os.OpenRoot", "path traversal", "traversal", "sandboxed file access", "untrusted paths"},
		body:     "Since Go 1.24, os.OpenRoot(dir) returns an os.Root that confines every file operation to that one directory tree. Its Open, Create, and MkdirAll methods reject paths that escape, whether through .. segments or symlinks pointing outside, and paths are always relative to the root. Use a root whenever a filename comes from a user, an archive, or the network, instead of checking path strings yourself.",
	},
	{
		title:    "system calls and external commands",
		keywords: []string{"system calls", "system call"},
		body:     "A system call is your program asking the operating system kernel to do something. Go's old syscall package is frozen, so low-level work belongs in the external golang.org/x/sys packages, such as x/sys/unix. Most programs never need that level: the os/exec package runs external programs instead, with exec.Command(name, args) and CombinedOutput to capture output and errors together. Never build commands from unsanitized user input.",
	},
	{
		title:    "bufio buffered io",
		keywords: []string{"bufio", "bufio package", "bufio.Scanner", "bufio.Reader", "bufio.Writer", "buffered io", "line by line"},
		body:     "The bufio package adds a memory buffer between your code and a file or connection, so many small reads and writes become few system calls. bufio.Scanner reads line by line, which suits logs and text files, but its default token limit is about 64KB, so long lines need scanner.Buffer. bufio.Writer collects output until you call Flush; forgetting Flush silently loses the last chunk.",
	},
	{
		title:    "csv files",
		keywords: []string{"csv", "csv file", "a csv file", "csv.Reader", "csv.Writer", "encoding/csv", "comma-separated values", "file"},
		body:     "The encoding/csv package reads and writes comma-separated values, where each record is a row of string fields. csv.NewReader reads one record at a time with Read, or everything with ReadAll. csv.NewWriter writes records with Write; its output is buffered, so always call Flush and check writer.Error afterwards. Set reader.Comma for other delimiters like semicolons, and treat the header row as an ordinary first record.",
	},
	{
		title:    "xml files",
		keywords: []string{"xml", "xml file", "an xml file", "encoding/xml", "xml.Marshal", "xml.Unmarshal", "xml.Decoder", "file"},
		body:     "The encoding/xml package turns Go structs into XML and back, much like encoding/json. Field tags like xml:\"name\" choose element names, the special XMLName field sets the root element, and tags with ,attr mark attributes instead of elements. xml.Marshal and xml.Unmarshal handle whole documents; for large files, stream tokens with an xml.Decoder so you never load everything into memory at once.",
	},
	{
		title:    "yaml files",
		keywords: []string{"yaml", "yaml file", "yaml config", "yaml.Marshal", "yaml.Unmarshal", "yaml.v3", "go-yaml"},
		body:     "YAML is a human-friendly format used mostly for configuration files. Go has no YAML support in its standard library, so this needs the external gopkg.in/yaml.v3 package. Its Marshal and Unmarshal mirror the JSON functions, using yaml tags on struct fields. When writing with a yaml.Encoder, always Close it: the encoder buffers output, and skipping Close can silently leave your file truncated.",
	},
	{
		title:    "database connection pooling",
		keywords: []string{"connection pooling", "database connection pool", "SetMaxOpenConns", "SetMaxIdleConns", "db.Stats", "tune connection pool"},
		body:     "A sql.DB is a pool of database connections, not one connection: it opens connections lazily, reuses them across queries, and is safe to share. Tune it with db.SetMaxOpenConns to cap load, db.SetMaxIdleConns for ready connections, and db.SetConnMaxLifetime so old connections retire. Watch db.Stats: a rising WaitCount means requests are stuck waiting for a free connection.",
	},
	{
		title:    "database/sql package",
		keywords: []string{"database/sql", "sql package", "sql.Open", "sql.DB", "query a database", "database queries", "sql queries"},
		body:     "database/sql is the standard interface for SQL databases: you blank-import a driver for your database, open a handle with sql.Open, and check it with db.Ping. Run reads with db.Query and loop rows.Next with rows.Scan, writes with db.Exec, and reuse work with db.Prepare. Always pass values as arguments instead of building SQL strings, which prevents injection.",
	},
	{
		title:    "sqlite databases",
		keywords: []string{"sqlite", "sqlite database", "sqlite3", "go-sqlite3", "embedded database", "local database storage", "use sqlite"},
		body:     "SQLite is a serverless SQL database kept in a single file, handy for local storage, small apps, and tests. Go reaches it through database/sql with an external driver, usually github.com/mattn/go-sqlite3, imported blank and opened with sql.Open(\"sqlite3\", \"app.db\"). Queries, Exec, and transactions then work like any other SQL database, with question-mark placeholders.",
	},
	{
		title:    "database transactions",
		keywords: []string{"database transactions", "BeginTx", "tx.Commit", "rollback transaction", "atomic", "transaction boundaries", "database transaction", "transaction"},
		body:     "A transaction groups several database statements so they all succeed or none do. Start one with db.BeginTx, run every statement on the returned tx, and finish with tx.Commit; success is only real once Commit returns nil. Defer a rollback guarded so it runs on error or panic, keep transactions short, and never mix plain db queries inside one.",
	},
	{
		title:    "comparing and adding dates",
		keywords: []string{"compare dates", "comparing dates", "add dates", "AddDate", "date arithmetic", "time.Time", "days between dates"},
		body:     "Compare time.Time values with the Before, After, and Equal methods, not with == or <, because methods handle locations correctly. Shift a date with t.Add(duration) for hours and minutes, or t.AddDate(years, months, days) for calendar steps. Get the gap between two times with end.Sub(start), which returns a time.Duration you can read as hours or days.",
	},
	{
		title:    "measuring execution time",
		keywords: []string{"measuring execution time", "measure execution time", "time.Since", "time code", "how long code takes", "benchmark timing"},
		body:     "To time a piece of code, save start := time.Now() before it runs and print time.Since(start) afterwards, which gives the elapsed time.Duration. A common pattern is defer func() { log.Println(time.Since(start)) }() at the top of a function. For serious comparisons use Go benchmarks instead, since one timing run is noisy and includes scheduling delays.",
	},
	{
		title:    "time zones",
		keywords: []string{"time zones", "time zone", "convert time zones", "time.Location", "UTC storage", "timezone"},
		body:     "A time.Time carries a location, and time.LoadLocation(\"America/New_York\") loads a named zone with its daylight saving rules. Convert with t.In(loc), which shows the same moment in that zone; never use a fixed hour offset, because offsets change. Store times in UTC with t.UTC() and convert to the reader's zone only when displaying them.",
	},
	{
		title:    "character encoding",
		keywords: []string{"character encoding", "character encodings", "convert encoding", "charmap", "transform.NewReader", "ISO-8859-1", "non-utf-8 text", "encoding"},
		body:     "Go strings are UTF-8 by default, but outside data may arrive in another encoding such as ISO-8859-1 or Windows-1252. The standard library alone does not convert these; you need the external golang.org/x/text packages. Wrap bytes with transform.NewReader and a decoder like charmap.ISO8859_1.NewDecoder to read them as UTF-8, and convert before parsing formats like JSON that assume UTF-8.",
	},
	{
		title:    "date and number formatting",
		keywords: []string{"date and number formatting", "format numbers for a locale", "locale formatting", "localized numbers", "currency formatting", "number formatting", "format dates"},
		body:     "Go formats dates with time.Format layouts, which always use English names and never adapt to a locale by themselves. Plain fmt verbs print numbers without locale grouping, decimal separators, or currency symbols. For real locale-aware dates and numbers you need golang.org/x/text or an external package, which supply the per-language rules the standard library leaves out.",
	},
	{
		title:    "localizing strings with go-i18n",
		keywords: []string{"localizing strings", "localize strings", "go-i18n", "translate strings", "translation files", "language packs", "what is go-i18n", "localize"},
		body:     "Localization means showing text in the user's language instead of one hardcoded language. Go's standard library has no translator, so you use the external go-i18n package: keep wording in per-language message files, load them into an i18n.Bundle, and ask an i18n.Localizer for each message ID in the reader's language. Templates inside messages fill in values like a user name at runtime.",
	},
	{
		title:    "pluralization",
		keywords: []string{"pluralization", "plural forms", "plural rules", "PluralCount", "plurals in translations", "singular and plural"},
		body:     "Pluralization is picking the right word form for a count, and languages disagree: English has one and other, while others need more forms. A simple if count == 1 check in code breaks for those languages. With go-i18n you give each message its plural forms in the translation file and pass PluralCount, and the library picks the form using that language's rules.",
	},
	{
		title:    "logging levels",
		keywords: []string{"logging levels", "log levels", "set the log level", "slog.HandlerOptions", "slog.LevelVar", "minimum level"},
		body:     "The plain log package has no levels, so programs fake them with separate log.Logger values carrying INFO or ERROR prefixes. log/slog builds levels in: Debug for detail, Info for normal events, Warn for problems, Error for failures. Set the minimum with slog.HandlerOptions Level, or use a slog.LevelVar to change verbosity while the program runs. Keep production at Info or Warn so debug noise never hides real errors.",
	},
	{
		title:    "logging to files",
		keywords: []string{"log to a file", "logging to a file", "log file", "logfile", "log", "log.SetOutput", "write logs to a file", "log rotation"},
		body:     "To log to a file, open it with os.OpenFile using O_CREATE, O_APPEND, and O_WRONLY, then point the logger at it with log.SetOutput(file) or log.New(file, prefix, flags). Remember defer file.Close(), or logs vanish when the program exits badly. Files grow forever without rotation; the standard library has no rotator, so use the external lumberjack package or an outside tool to cap size and age.",
	},
	{
		title:    "structured logging with slog",
		keywords: []string{"structured logging", "log/slog", "slog", "structured logs", "logging with context", "log with context", "slog.With"},
		body:     "log/slog, in the standard library since Go 1.21, logs structured records as key-value fields instead of plain text. Build one with slog.New(slog.NewTextHandler(os.Stdout, nil)) or NewJSONHandler for JSON, then call slog.Info(msg, key, value) style methods; slog.With attaches fields every record carries. Pass request values like IDs through context and log with slog.InfoContext so records stay traceable. Since Go 1.26, slog.NewMultiHandler sends records to several handlers at once.",
	},
	{
		title:    "checksums",
		keywords: []string{"checksum", "checksums", "crc32", "hash/crc32", "compute a checksum", "data integrity"},
		body:     "A checksum is a short fingerprint of data that changes when the data changes, so you can spot corruption. For fast non-secret checks use hash/crc32 Checksum or hash/fnv; both stream through hash.Hash like bigger hashes. Checksums stop accidents, not attackers, because anyone can recompute one. When tampering matters use a cryptographic hash like SHA-256 instead, and never use broken MD5 except where a legacy protocol demands it.",
	},
	{
		title:    "cross-site scripting xss prevention",
		keywords: []string{"xss", "xss prevention", "cross-site scripting", "prevent xss", "html/template", "html.EscapeString", "escape html"},
		body:     "Cross-site scripting (XSS) happens when attacker text is rendered as HTML or script in another user's browser. Render pages with html/template, never text/template: it escapes values for their exact context automatically. For one-off strings use html.EscapeString from the html package. Escape at output time, not on input, and add a Content-Security-Policy header as a second layer so injected scripts still cannot run.",
	},
	{
		title:    "csrf protection",
		keywords: []string{"csrf", "csrf protection", "cross-site request forgery", "CrossOriginProtection", "http.NewCrossOriginProtection", "protect against csrf"},
		body:     "Since Go 1.25, net/http ships http.NewCrossOriginProtection for cross-site request forgery (CSRF) defense: wrap your mux with protection.Handler(mux) and it rejects state-changing cross-origin requests by checking Sec-Fetch-Site, falling back to Origin against Host. AddTrustedOrigin allows a known separate frontend, and AddInsecureBypassPattern exempts webhook-style routes. It costs almost nothing per request, but it is not CORS: CORS controls reading responses, this stops forged requests from running.",
	},
	{
		title:    "digital signatures",
		keywords: []string{"digital signature", "digital signatures", "signing data", "crypto/ecdsa", "ecdsa.SignASN1", "verify a signature"},
		body:     "A digital signature proves who signed data and that it was not altered: sign a hash of the message with a private key, and anyone verifies with the public key. In crypto/ecdsa use ecdsa.SignASN1 and ecdsa.VerifyASN1 with a curve like elliptic.P256; crypto/rsa signs with rsa.SignPSS. Hash with SHA-256 first, generate keys with crypto/rand, and never reuse a signing key for encryption.",
	},
	{
		title:    "encryption and decryption",
		keywords: []string{"encryption", "decryption", "encrypt data", "decrypt data", "AES-GCM", "cipher.NewGCM", "authenticated encryption", "encrypt"},
		body:     "Encryption scrambles data so only a key holder can read it. For application data use authenticated encryption: build an AES key into cipher.NewGCM via crypto/aes, seal plaintext with a fresh random nonce from crypto/rand for every message, and open it the same way; tampering makes Open fail. Extra context can be bound as associated data. ChaCha20-Poly1305 from golang.org/x/crypto is the alternative when AES hardware is slow.",
	},
	{
		title:    "go 1.26 cryptography migrations",
		keywords: []string{"go 1.26 cryptography", "new in go 1.26", "cryptography migrations", "cryptography", "PKCS1v15", "EncryptOAEP", "deprecated crypto", "rsa", "encryption", "new in go 1.26 cryptography"},
		body:     "Go 1.26 deprecates fragile cryptography habits. Leave tls.Config CipherSuites and CurvePreferences unset so upgrades, including post-quantum hybrids, keep flowing; pinning curves disables them. rsa.EncryptPKCS1v15 and rsa.DecryptPKCS1v15 are deprecated: move legacy RSA encryption to rsa.EncryptOAEP with SHA-256. Direct access to ecdsa key big.Int fields is also deprecated: serialize keys through crypto/x509 and sign with ecdsa.SignASN1 and ecdsa.VerifyASN1 instead.",
	},
	{
		title:    "hashing algorithms",
		keywords: []string{"hashing", "hash function", "sha256", "crypto/sha256", "sha256 hash", "hash algorithms"},
		body:     "Hashing turns data into a fixed-size digest with a one-way hash function: same input, same digest, and no way back. crypto/sha256 is the everyday default: h := sha256.New(), h.Write(data), then h.Sum(nil); crypto/sha512 suits 64-bit-heavy work. MD5 and SHA-1 are broken and survive only for legacy interop. Never hash passwords with these fast hashes, and use crypto/hmac when a secret key must authenticate the digest.",
	},
	{
		title:    "https and tls configuration",
		keywords: []string{"https", "tls configuration", "crypto/tls", "ListenAndServeTLS", "https server", "https and tls", "configure tls"},
		body:     "HTTPS encrypts HTTP with TLS so eavesdroppers cannot read or alter traffic. Serve it with http.Server ListenAndServeTLS or a crypto/tls Config with MinVersion tls.VersionTLS12, and let Go pick cipher suites and curves instead of copying old lists. Never set InsecureSkipVerify on clients; add a private CA to an x509 CertPool instead. For mutual TLS, require client certificates with tls.RequireAndVerifyClientCert, and set server timeouts alongside the TLS settings.",
	},
	{
		title:    "hybrid public key encryption hpke",
		keywords: []string{"hpke", "hybrid public key encryption", "crypto/hpke", "hpke encryption", "encrypt to a public key", "public key encryption"},
		body:     "New in Go 1.26, crypto/hpke implements hybrid public key encryption: a sender encrypts to the recipient's public key and only the matching private key can open it. Pick a suite such as X25519 with HKDF-SHA256 and AES-GCM, seal the message, and bind context with the info value and per-message associated data; changed associated data fails to open. Use it for envelope-style payloads instead of hand-assembling ECDH, HKDF, and AEAD yourself.",
	},
	{
		title:    "json web tokens jwt",
		keywords: []string{"jwt", "json web token", "json web tokens", "jwt in go", "validate jwt", "golang-jwt", "web token", "token"},
		body:     "JSON Web Tokens (JWTs) are signed tokens a server issues so clients can prove identity on later requests without a session lookup. Go has no standard-library JWT support: use the external golang-jwt/jwt package, or go-jose when you need encrypted tokens. Sign with a strong secret of at least 256 bits, keep expiry short, store tokens in HttpOnly Secure cookies, and always restrict accepted signing algorithms when validating, or attackers can substitute their own.",
	},
	{
		title:    "key derivation with hkdf and pbkdf2",
		keywords: []string{"key derivation", "hkdf", "pbkdf2", "crypto/hkdf", "crypto/pbkdf2", "derive a key", "derive keys from passwords"},
		body:     "Key derivation turns one secret into strong keys. Since Go 1.24, crypto/hkdf expands a high-entropy secret, like key-exchange output, into separate keys; give each purpose its own info string so keys never mix. For passwords use crypto/pbkdf2 instead: it repeats HMAC hundreds of thousands of times with a unique random salt per password to slow guessing. Never use a raw shared secret or password as a cipher key directly.",
	},
	{
		title:    "ecdh key exchange",
		keywords: []string{"ecdh", "key exchange", "ecdh key exchange", "crypto/ecdh", "shared secret", "diffie-hellman"},
		body:     "ECDH lets two parties agree on a shared secret over an open network: each generates a key pair with crypto/ecdh, exchanges public keys, and both compute the same secret from their private key and the other's public key. Use ecdh.P256() for most work, or X25519. Never use the raw secret as an encryption key: run it through HKDF first, use fresh ephemeral keys per exchange, and validate peer keys.",
	},
	{
		title:    "ml-dsa post-quantum signatures",
		keywords: []string{"mldsa", "ml-dsa signatures", "crypto/mldsa", "post-quantum signatures", "ml-dsa", "fips 204"},
		body:     "New in Go 1.27, crypto/mldsa implements ML-DSA, the FIPS 204 post-quantum digital signature scheme that stays secure against quantum computers. Generate keys with mldsa.GenerateKey(mldsa.MLDSA44()) — the recommended starting size — sign with the private key, and verify with mldsa.Verify; an optional context string separates signatures between protocols. Keys and signatures are far larger than ECDSA ones, and Go 1.27 also wires ML-DSA into crypto/x509 and TLS 1.3.",
	},
	{
		title:    "password hashing",
		keywords: []string{"password hashing", "hash a password", "password hash", "argon2id", "bcrypt", "store passwords securely"},
		body:     "Store passwords as slow salted hashes, never as plain or fast hashes. Prefer Argon2id from the external golang.org/x/crypto/argon2 package, with a fresh random salt per password and the parameters stored beside the hash in one encoded string; bcrypt from the same library is the older alternative. Verify by deriving again and comparing with crypto/subtle constant-time comparison, target roughly 100 to 250 milliseconds per check, and rehash on login when parameters age.",
	},
	{
		title:    "post-quantum key exchange with ml-kem",
		keywords: []string{"mlkem", "ml-kem", "crypto/mlkem", "post-quantum key exchange", "key encapsulation", "ml-kem key exchange"},
		body:     "ML-KEM is a post-quantum key encapsulation mechanism: one side generates a key pair with crypto/mlkem GenerateKey768, the other calls Encapsulate on the public key to create a ciphertext and shared secret, and the owner decapsulates to recover it. Since Go 1.24 it is in the standard library, and crypto/tls already uses hybrid ML-KEM key exchange by default. Always pass the shared secret through HKDF before using it as a cipher key.",
	},
	{
		title:    "random number generation",
		keywords: []string{"random numbers", "random number generation", "crypto/rand", "math/rand/v2", "secure random numbers", "generate random numbers"},
		body:     "Go has two random packages and picking the right one matters. math/rand/v2, the standard random package since Go 1.22, gives fast predictable numbers for simulations and shuffling, and seeds itself automatically; never call the old Seed. crypto/rand gives unpredictable numbers from the operating system: use it for keys, tokens, salts, and nonces, filling bytes with rand.Read or picking with rand.Int. Using math/rand/v2 for anything secret lets attackers predict and forge your values.",
	},
	{
		title:    "gosec security scanner",
		keywords: []string{"gosec", "gosec tool", "run gosec", "security scanner", "scan code for vulnerabilities", "securego/gosec"},
		body:     "gosec is an external security scanner for Go source code: install it with go install github.com/securego/gosec/v2/cmd/gosec@latest, then run gosec ./... to flag problems like hardcoded credentials, weak random numbers, and unsafe SQL construction. Narrow scans with -include or -exclude rule IDs such as G101, and emit JSON or JUnit reports with -fmt for CI. Treat findings as leads to review, not proof, and never silence real ones just to go green.",
	},
	{
		title:    "vulnerability scanning with govulncheck",
		keywords: []string{"govulncheck", "what is govulncheck", "vulnerability scanning", "scan for vulnerabilities", "go tool govulncheck", "vulnerable dependencies", "check dependencies for vulnerabilities"},
		body:     "govulncheck is the official Go vulnerability scanner: it checks your module's dependencies against the Go vulnerability database and, unlike simple version matching, reports only advisories your code actually calls. Since Go 1.24 you can pin it as a tool dependency and run go tool govulncheck ./... in CI. Fix findings by upgrading the dependency or toolchain to a patched version, rerun until clean, and rescan on a schedule because new advisories appear constantly.",
	},
	{
		title:    "analyzing memory allocations",
		keywords: []string{"memory allocations", "analyze memory allocations", "heap profile", "net/http/pprof", "allocations", "memory profiling", "memory allocation", "allocation profiling"},
		body:     "Memory allocation profiling shows where your program creates heap objects and how much memory stays live. Import net/http/pprof in a server, or take a heap profile with runtime/pprof, then inspect it with go tool pprof using top and list. Profile with real workloads, watch Alloc and TotalAlloc through runtime.ReadMemStats for quick snapshots, and cut repeated allocations in the hottest paths first.",
	},
	{
		title:    "cpu profiling with pprof",
		keywords: []string{"cpu profiling", "cpu profile", "pprof", "runtime/pprof", "pprof.StartCPUProfile", "go tool pprof", "performance bottlenecks", "bottleneck"},
		body:     "CPU profiling shows where your program actually spends its execution time. Wrap a run with pprof.StartCPUProfile and defer pprof.StopCPUProfile, or serve net/http/pprof from a server, then explore the result with go tool pprof: top lists the hottest functions, list annotates source, and web draws the call graph. Profile realistic workloads, fix the hottest functions first, and combine with heap and goroutine profiles when the bottleneck is not CPU alone.",
	},
	{
		title:    "debugging with delve",
		keywords: []string{"delve", "go-delve", "debugger", "debugging with delve", "debug with delve", "dlv debug", "breakpoint", "dlv attach"},
		body:     "Delve is the standard debugger for Go programs, installed with go install github.com/go-delve/delve/cmd/dlv@latest and started with dlv debug. Set breakpoints with break, run with continue, step with next and step, and inspect values with print and goroutines. Build with optimizations disabled, using -gcflags=all=-N -l, or variables and stepping will mislead you. Delve can also attach to a running process by its PID.",
	},
	{
		title:    "erasing secrets with runtime/secret",
		keywords: []string{"runtime/secret", "secret.Do", "erasing secrets", "secret erasure", "runtimesecret", "GOEXPERIMENT"},
		body:     "Experimental in Go 1.26: runtime/secret asks the runtime to erase stack and register data after short secret work finishes. Build with GOEXPERIMENT=runtimesecret, run the handling inside secret.Do, and keep that callback tiny, returning only a derived value like a digest. It works only on linux/amd64 and linux/arm64, heap copies you keep are not promptly erased, and it has no Go 1 compatibility guarantee, so treat it as defense in depth.",
	},
	{
		title:    "flight recorder",
		keywords: []string{"flight recorder", "trace.FlightRecorder", "FlightRecorder", "flight recording", "runtime/trace"},
		body:     "Since Go 1.25, the flight recorder in runtime/trace keeps a rolling in-memory execution trace, so a long-running service can save the seconds before a problem instead of tracing from startup. Start a trace.NewFlightRecorder with MinAge and MaxBytes, and call WriteTo to snapshot the buffer when a request turns slow or a health check fails. Inspect snapshots with go tool trace, and guard captures so one incident does not flood your disk.",
	},
	{
		title:    "go scheduler",
		keywords: []string{"go scheduler", "scheduler", "goroutine scheduler", "GOMAXPROCS", "work stealing", "runtime.Gosched"},
		body:     "The Go scheduler runs many goroutines on fewer operating system threads, multiplexing M goroutines onto N threads. GOMAXPROCS caps how many threads execute Go code at once, work stealing moves ready goroutines from busy processors to idle ones, and runtime.Gosched politely yields. Goroutines yield at blocking operations, so never rely on creation order or scheduling order for correctness; synchronize with channels and wait groups instead.",
	},
	{
		title:    "goroutineleak profile",
		keywords: []string{"goroutine leak profile", "leak profile", "pprof.Lookup", "goroutineleak profile", "leak profiling", "goroutine leaks"},
		body:     "The goroutineleak profile lists goroutines the garbage collector can prove are blocked forever on channels or other Go primitives, rather than every running goroutine. Read it with pprof.Lookup(\"goroutineleak\") after the work should have finished, in tests or from the net/http/pprof endpoint. Experimental in Go 1.26 behind GOEXPERIMENT=goroutineleakprofile; new in Go 1.27 it is generally available. It cannot see goroutines stuck on blocking I/O, so fix leaks with context cancellation and clear shutdown paths.",
	},
	{
		title:    "profiling with goroutine labels",
		keywords: []string{"goroutine labels", "pprof labels", "pprof.Do", "profiling labels", "labels"},
		body:     "Goroutine labels tag work so profiles can separate jobs that share the same functions. Run code inside pprof.Do with a context carrying labels for a stable dimension like an operation or queue, and child goroutines inherit them; worker pools need pprof.Do around each job. Labels filter CPU and goroutine profiles, not heap profiles. In Go 1.27, labels also appear in goroutine traceback headers, so never put secrets or personal data in them.",
	},
	{
		title:    "reading go stack traces",
		keywords: []string{"stack trace", "stack traces", "reading stack traces", "runtime.Stack", "goroutine stack trace", "call stack"},
		body:     "A stack trace lists the chain of function calls active when a panic or dump happened, with file names and line numbers, reading from main at the bottom up to the crash at the top. Find the first frame in your own code to locate the fault, and capture traces on demand with runtime.Stack into a byte buffer. Keep function names clear and traces out of logs that may carry sensitive data.",
	},
	{
		title:    "runtime metrics",
		keywords: []string{"runtime metrics", "runtime/metrics", "metrics.Read", "metrics.All", "runtime telemetry", "metrics samples"},
		body:     "The runtime/metrics package exposes stable, documented measurements from the Go runtime, such as live heap bytes, garbage collection cycles, and the goroutine count. Build a small list of metrics.Sample values with names like /gc/heap/live:bytes, fill them in one metrics.Read call, and switch on Value.Kind before reading a number. Prefer a reviewed handful over exporting everything from metrics.All, and resample on a timer, not per request.",
	},
	{
		title:    "runtime package statistics",
		keywords: []string{"runtime package", "runtime statistics", "runtime.ReadMemStats", "MemStats", "memory statistics", "runtime.NumGoroutine", "runtime information"},
		body:     "The runtime package reports live facts about your process: runtime.ReadMemStats fills a MemStats with Alloc, TotalAlloc, and NumGC, runtime.NumGoroutine counts goroutines, and runtime.Version, GOOS, GOARCH, and GOMAXPROCS describe the build and scheduler limits. Use these for occasional snapshots and diagnostics, prefer runtime/metrics and pprof for steady monitoring, and avoid routine manual tuning like forcing runtime.GC or resetting GOMAXPROCS yourself.",
	},
	{
		title:    "tracing go programs",
		keywords: []string{"execution tracing", "runtime/trace", "trace.Start", "go tool trace", "tracing"},
		body:     "Execution tracing records when goroutines run, block, and wake, which makes it ideal for latency spikes and scheduler puzzles that profiles cannot explain. Wrap a short window with trace.Start and trace.Stop from runtime/trace, then open the file with go tool trace to inspect timelines. Keep traces short and focused on a suspicious operation; for long-running services, use the flight recorder instead of tracing the whole lifetime.",
	},
	{
		title:    "godebug environment variable",
		keywords: []string{"GODEBUG", "godebug environment variable", "gctrace", "schedtrace", "go:debug", "debug settings"},
		body:     "GODEBUG is an environment variable that switches on runtime diagnostics and compatibility knobs before your program starts, as in GODEBUG=gctrace=1 for garbage collector traces or schedtrace=1000 for scheduler traces every second. Since Go 1.21, named settings also revert behavior changes between releases, and a //go:debug line in the main package or go.mod can pin them per module. Enable only what you need, because verbose tracing adds real overhead.",
	},
	{
		title:    "cleanup functions with runtime.AddCleanup",
		keywords: []string{"runtime.AddCleanup", "cleanup functions", "add cleanup", "runtime.SetFinalizer", "cleanup"},
		body:     "A cleanup function frees a resource the garbage collector cannot see, like a file handle or C memory. Since Go 1.24, runtime.AddCleanup(ptr, fn, arg) runs fn(arg) after ptr becomes unreachable; arg must not reference ptr itself. It replaces runtime.SetFinalizer, allows several cleanups per object, and returns a handle whose Stop cancels it. Always provide an explicit Close too: cleanups are a safety net, never guaranteed timing.",
	},
	{
		title:    "experimental simd with archsimd",
		keywords: []string{"archsimd", "simd/archsimd", "experimental simd", "vector operations", "GOEXPERIMENT"},
		body:     "Experimental: in Go 1.26, simd/archsimd exposes amd64 vector types that do arithmetic on many values at once, built only with GOEXPERIMENT=simd and with no compatibility guarantee. You load a slice into a vector such as Uint8x16, apply an operation, and store the result back. Keep a plain Go fallback, confine vector types to internal packages, and benchmark first, because memory bandwidth often limits the gain.",
	},
	{
		title:    "garbage collection tuning",
		keywords: []string{"garbage collection tuning", "garbage collection", "GOGC", "debug.SetGCPercent", "tune garbage collection", "gc tuning"},
		body:     "Garbage collection tuning adjusts how often the collector runs so your program trades memory for speed deliberately. GOGC, default 100, sets how much the heap may grow before a collection; lower it with debug.SetGCPercent for tighter memory, or raise it to collect less often. Read runtime.MemStats to watch the effect. Since Go 1.26 the Green Tea collector is the default. Reduce allocations first; tuning cannot fix a leak.",
	},
	{
		title:    "go assembly",
		keywords: []string{"go assembly", "assembly", "plan 9 assembly", "assembly functions"},
		body:     "Go assembly is Plan 9 style assembly you write in .s files for the rare hot spot plain Go cannot express, like a crypto kernel or a CPU instruction the compiler never emits. A Go declaration pairs with the assembly body, and go vet checks frame sizes and offsets. Keep functions tiny and leaf-only, provide a pure Go fallback, and exhaust intrinsics like math/bits first: assembly blocks inlining and ages badly.",
	},
	{
		title:    "compiler optimizations",
		keywords: []string{"compiler optimizations", "escape analysis", "inlining", "gcflags", "compiler"},
		body:     "The Go compiler optimizes automatically: it inlines small hot functions, and escape analysis decides whether a value can live cheaply on the stack or must escape to the heap. Run go build -gcflags=-m to see those decisions and find surprise heap allocations. Flags like -N and -l disable optimizations for debugging, so never ship them. Write plain idiomatic Go, keep hot functions small, and confirm any hand optimization with benchmarks.",
	},
	{
		title:    "gomaxprocs",
		keywords: []string{"gomaxprocs", "runtime.GOMAXPROCS", "container aware gomaxprocs", "set gomaxprocs"},
		body:     "GOMAXPROCS sets how many operating system threads can run Go code at the same time. Since Go 1.25 the default is container-aware: the runtime reads Linux CPU limits and adjusts when they change, so you normally leave it alone. Read it with runtime.GOMAXPROCS and change it with runtime.SetDefaultGOMAXPROCS or the GOMAXPROCS environment variable. Setting it yourself disables the automatic adjustment, so override only with a measured reason.",
	},
	{
		title:    "memory profiling",
		keywords: []string{"memory profiling", "runtime/pprof", "heap profile", "pprof.WriteHeapProfile", "profile memory usage"},
		body:     "Memory profiling shows where your program allocates and what it keeps alive, so you can find leaks and hot allocation paths. Capture a heap snapshot with pprof.WriteHeapProfile from runtime/pprof, or serve profiles from a running server, then explore with go tool pprof using top and list. A profile is one snapshot in time: profile under realistic load, and compare inuse and alloc views before changing code.",
	},
	{
		title:    "memory management in go",
		keywords: []string{"memory management", "garbage collector", "memory management in go", "heap and stack"},
		body:     "Go manages memory for you: the garbage collector frees values once nothing references them, and escape analysis keeps short-lived values on the fast stack instead of the heap. You still shape usage: preallocate slices and maps you grow, clear maps you reuse, reuse buffers with sync.Pool, and avoid globals that quietly retain data forever. When memory grows unexpectedly, profile before guessing: retained references, not the collector, are the usual cause.",
	},
	{
		title:    "portable simd",
		keywords: []string{"portable simd", "simd package", "simd", "vector arithmetic", "the simd package"},
		body:     "New in Go 1.27: an experimental simd package does vector arithmetic portably, choosing the vector width for the machine running your code and emulating operations the hardware lacks. Unlike simd/archsimd you write one version for amd64, arm64, and wasm, built with GOEXPERIMENT=simd and a build tag. Handle the leftover tail elements with a plain loop, keep a scalar fallback file, and benchmark realistic sizes before trusting the speedup.",
	},
	{
		title:    "preallocating slices",
		keywords: []string{"preallocating slices", "preallocate a slice", "preallocate slice", "make with capacity", "slice capacity", "preallocate"},
		body:     "Preallocating a slice means reserving its backing array once, at the size you expect, instead of paying for repeated growing and copying as append runs. Use make([]T, 0, expected) so length starts empty while capacity is ready, then append normally. Watch the common mix-up: make([]T, n) creates n zero values already in the slice. Preallocate when you know or can estimate the final count, especially in loops building large results.",
	},
	{
		title:    "profile-guided optimization",
		keywords: []string{"profile-guided optimization", "pgo", "default.pgo", "profile guided"},
		body:     "Profile-guided optimization, or PGO, stable since Go 1.21, feeds a CPU profile from a real run back into the compiler, so it optimizes the paths your program actually takes. Collect a representative profile with runtime/pprof, save it as default.pgo in the main package directory, and a normal go build picks it up automatically. Keep the profile fresh after big changes, and verify with benchmarks: a stale or narrow profile can optimize the wrong workload.",
	},
	{
		title:    "reducing allocations",
		keywords: []string{"reducing allocations", "reduce allocations", "sync.Pool", "benchmem", "fewer allocations"},
		body:     "Reducing allocations means creating less heap garbage, so the garbage collector runs less and latency steadies. The reliable moves are preallocating slice and map capacity when sizes are known, reusing buffers instead of rebuilding them, and pooling hot temporary objects with sync.Pool, resetting each one before you Put it back. Measure with go test -benchmem before and after: fewer allocs/op should be visible, and readable code beats clever code everywhere except proven hot paths.",
	},
	{
		title:    "size-specialized allocations",
		keywords: []string{"size-specialized allocations", "small allocations", "size specialized malloc", "nosizespecializedmalloc", "go 1.27"},
		body:     "New in Go 1.27: the heap allocator is size-specialized for small objects up to 80 bytes, making each small allocation roughly 20 to 30 percent faster with no new API and nothing to enable. Whole programs gain much less, so measure your own hot path with repeated benchmarks and benchstat; GOEXPERIMENT=nosizespecializedmalloc restores the old allocator only for comparison. Values on the stack or over 80 bytes get nothing, and removing allocations still beats speeding them up.",
	},
	{
		title:    "unsafe package",
		keywords: []string{"unsafe package", "unsafe.Pointer", "unsafe", "pointer arithmetic"},
		body:     "The unsafe package steps outside Go's type safety for low-level tricks: unsafe.Pointer converts between pointer types, and unsafe.Sizeof, Alignof, and Offsetof expose layout details. That power can corrupt memory, break garbage collector assumptions, and tie code to one architecture layout. Reach for it only for proven hot paths or calling C, keep it behind a small safe API, and document why every use is sound.",
	},
	{
		title:    "value interning",
		keywords: []string{"value interning", "unique package", "unique.Make", "interning", "unique.Handle"},
		body:     "Value interning stores only one copy of each equal value, so many duplicates share a single allocation. The unique package does this for comparable values: unique.Make(value) returns a unique.Handle, handles for equal values are identical and compare fast, and handle.Value gives the stored value back. Intern repeated strings and configuration values you keep for a long time. Skip rare or tiny values: interning itself costs a lookup, and interned values live while handles do.",
	},
	{
		title:    "weak references",
		keywords: []string{"weak references", "weak.Pointer", "weak package", "weak.Make"},
		body:     "Since Go 1.24, the weak package makes weak pointers that reference an object without keeping it alive: weak.Make(ptr) creates one, and Value returns the pointer, or nil after the collector frees the object. That makes caches and canonical maps that shrink on their own instead of leaking forever. Check for nil on every use, because the object can vanish between storing and reading, and clean dead entries out of your cache lazily.",
	},
	{
		title:    "zero-copy techniques",
		keywords: []string{"zero-copy techniques", "zero copy", "zero-copy", "mmap"},
		body:     "Zero-copy techniques move data by passing references instead of duplicating bytes. Work with slices of a shared buffer, stream through io.Reader and io.Writer instead of loading everything into memory, and for very large files map the file with syscall.Mmap so you read its bytes in place. The danger is lifetime: a slice is only valid while its backing data is, and a mapped file stays open until you unmap it explicitly.",
	},
	{
		title:    "build tags and platform-specific code",
		keywords: []string{"build tags", "build constraints", "platform-specific code", "go:build", "_linux.go file suffix", "cross-compiling", "GOOS", "GOARCH"},
		body:     "Build tags decide which files join a build. A //go:build line at the top of a file, followed by a blank line, gives conditions like linux, windows, amd64, or your own tags, combined with &&, ||, and ! and switched on with go build -tags debug. Simple cases need no tag: a _linux.go or _windows.go filename suffix selects the platform automatically. Set GOOS and GOARCH to cross-compile for another system from your own machine.",
	},
	{
		title:    "abstract syntax trees and go/ast",
		keywords: []string{"abstract syntax tree", "go/ast", "ast.Inspect", "parser.ParseFile", "syntax tree", "traversing go code", "go/printer"},
		body:     "An abstract syntax tree (AST) is your Go source turned into a tree of nodes: declarations, expressions, and statements that tools can walk and change. parser.ParseFile with a token.FileSet builds the tree, ast.Inspect visits each node, and you can rename, insert, or delete nodes, then print the result back to source with go/printer. Linters, generators, and refactoring tools are built this way. Check node types before casting, or your tool panics.",
	},
	{
		title:    "building command-line applications",
		keywords: []string{"command-line applications", "cli applications", "flag package", "command line flags", "subcommands", "flag.String", "cobra"},
		body:     "A command-line application reads its instructions from arguments and flags instead of a window. The standard flag package covers simple tools: flag.String, flag.Int, and friends define options with defaults and help text, flag.Parse reads them, and os.Args holds any leftover words. Validate inputs and print clear errors. When a tool grows subcommands like add and list, the external cobra package organizes them; many small tools never need it.",
	},
	{
		title:    "cgo and calling c libraries",
		keywords: []string{"cgo", "calling c code", "c libraries", "import C", "C.CString", "cgo directives", "CGO_ENABLED", "what is cgo"},
		body:     "cgo lets Go call C functions and use C libraries when no pure Go option exists. Write the C code or includes in a comment directly above import C, link libraries with #cgo directives like #cgo LDFLAGS: -lm, and convert values between C and Go types explicitly. Memory from C.CString is yours to free with C.free. Each call crosses a boundary with real overhead, builds get less portable, and CGO_ENABLED=0 turns cgo off.",
	},
	{
		title:    "go generate and code generation",
		keywords: []string{"go generate", "code generation", "generate code", "go:generate directive", "stringer", "generated code"},
		body:     "go generate runs commands written in //go:generate comments in your source files; running go generate ./... executes them in order to produce code for you. A classic use is stringer, which writes the String method for a set of constants so the names never drift from the values. Commit the generated files, note the generator at the top of each one, and rerun generation after changing the inputs instead of editing generated code by hand.",
	},
	{
		title:    "embedding files into the binary",
		keywords: []string{"embed files", "embed package", "go:embed", "embed.FS", "embedded templates", "single binary deployment"},
		body:     "The embed package bakes files into your compiled binary at build time, so templates, web assets, and default data travel inside the single executable. A //go:embed comment above a string, []byte, or embed.FS variable names the files to include; embed.FS works like a read-only folder you can serve or read at runtime. Added in Go 1.16. Remember the files are frozen when you build: changing one means rebuilding, and large files grow the binary.",
	},
	{
		title:    "exporting go functions to webassembly",
		keywords: []string{"webassembly", "wasm", "wasmexport", "wasip1", "go functions to wasm", "wasi"},
		body:     "Since Go 1.24, the //go:wasmexport comment marks a Go function so a WebAssembly host program can call it directly. Build the module with GOOS=wasip1 GOARCH=wasm go build to produce a .wasm file for a WASI runtime. Keep the exported surface small: a few functions with simple number or byte arguments cross the boundary far more easily than pointer-heavy designs, and treat the exported names as a public API your host depends on.",
	},
	{
		title:    "generating code with text templates",
		keywords: []string{"generating code with templates", "code templates", "template FuncMap", "generate code", "generate code with templates"},
		body:     "A code generator can fill a text/template with data and write the result out as a .go file, turning repetitive declarations into a small template plus a table of values. template.New, Parse or ParseFiles, and Execute do the work, and template.FuncMap adds your own helpers like uppercase for use inside the template. Parse once and reuse, check every error, and run gofmt on the output so generated code looks hand-written.",
	},
	{
		title:    "go compiler directives",
		keywords: []string{"compiler directives", "go directives", "go:noinline", "noinline", "linkname directive", "go linkname", "being inlined"},
		body:     "Compiler directives are //go: comments that instruct the compiler or toolchain instead of describing the code: //go:build selects files, //go:generate declares generation commands, and //go:noinline stops a function being inlined. Most code needs only the first two. Reach for advanced ones like //go:linkname, which binds a name to another package's symbol, only with real need: they are undocumented contracts that can break on upgrade and make code hard to follow. Document every directive you use.",
	},
	{
		title:    "go telemetry",
		keywords: []string{"go telemetry", "telemetry", "telemetry modes", "GOTELEMETRY", "anonymous usage data", "toolchain usage"},
		body:     "Go telemetry counts how the toolchain itself is used: which commands run, which Go version, and how long builds take, never your source code or your own programs. It has three modes: on uploads the anonymous counters to the Go team, local keeps them on your machine only, and off collects nothing. Local is the default. Check or change yours with the go telemetry command, and set GOTELEMETRY=off in CI so build machines do not skew the counts.",
	},
	{
		title:    "tool dependencies in go.mod",
		keywords: []string{"tool directive", "tool dependencies", "pin developer tools", "go tool", "tools.go file", "go.mod tools"},
		body:     "Since Go 1.24, go.mod can pin the developer tools your project uses, like linters and generators, with a tool directive. Add one with go get -tool followed by the tool's package path, then run it at exactly that version with go tool name, no separate install step. This replaces the old tools.go file of blank imports. Upgrade or drop a tool the same go get way, and run go mod tidy afterwards.",
	},
	{
		title:    "reducing binary size",
		keywords: []string{"binary size", "smaller binary", "ldflags", "trimpath", "strip symbols", "upx compression"},
		body:     "Go binaries carry runtime and type information, so they start large, but you can shrink them. Building with go build -ldflags -s -w strips the symbol table and debug information, and -trimpath removes your local file paths from the binary. Setting CGO_ENABLED=0 and importing only what you need help too. Inspect what is inside with go version -m and go tool nm. The trade-off: stripped binaries are harder to debug, and compressors like UPX slow startup.",
	},
	{
		title:    "text/scanner package",
		keywords: []string{"text/scanner", "scanner.Scanner", "tokenize text", "parsing structured text", "scanner modes", "TokenText"},
		body:     "The text/scanner package splits structured text into a stream of tokens, handy for small parsers and little languages. Create a scanner.Scanner, call Init with a strings.Reader or file, then loop on Scan until it returns scanner.EOF, reading each token's text with TokenText and its place with Position. Mode settings control whether whitespace and comments appear as tokens. Set an Error handler so bad input reports problems instead of failing silently.",
	},
	{
		title:    "running external commands with os/exec",
		keywords: []string{"os/exec", "exec.Command", "run external command", "external processes", "process management", "exec.CommandContext"},
		body:     "The os/exec package starts other programs from your Go code. exec.Command builds a command from a program name and separate arguments, cmd.Run starts it and waits, and buffers wired to Stdout and Stderr capture what it prints; Stdin feeds it input. exec.CommandContext adds a timeout or cancellation through context. Prefer Signal over Kill when stopping a process so it can clean up, and pass arguments as separate values, never one shell string.",
	},

	{
		title:    "go/token and source positions",
		keywords: []string{"go/token", "token package", "token.FileSet", "source positions", "go/scanner"},
		body:     "The go/token package tracks where every piece of a parsed Go file came from. A token.FileSet records the files you parse into it, and fset.Position turns any token.Pos value back into a filename, line, and column for error messages. The companion go/scanner package splits raw Go source into individual tokens when a tool needs them. Create one FileSet per parsing session and share it across the files of a package so positions stay consistent.",
	},
	{
		title:    "go/types and type checking",
		keywords: []string{"go/types", "type checking", "types.Info", "type checker", "go/types package"},
		body:     "The go/types package runs the same type checking the compiler does on a parsed syntax tree, so your own tools can know what every name means. Parse with go/parser, then check with a types.Config and go/importer for imported packages. Results land in types.Info: its Defs and Types maps link identifiers to their objects and types. For real modules with build tags and dependencies, load packages with the external golang.org/x/tools/go/packages package instead.",
	},
	{
		title:    "go fix",
		keywords: []string{"go fix", "modernize code", "modernizers", "go:fix inline", "deprecated apis", "fixing old code"},
		body:     "go fix rewrites your source in place with safe, automatic modernizations, such as swapping a retired standard library call for its current replacement. It was rebuilt on the vet analysis framework in Go 1.26, and Go 1.27 adds more modernizers. Run go fix ./... after a toolchain upgrade, then review the diff and run your tests before committing: it only applies rewrites it knows are safe. Library authors can mark a helper with //go:fix inline so callers get migrated automatically.",
	},
	{
		title:    "versioning and releasing go modules",
		keywords: []string{"module versioning", "semantic versioning", "versioning go modules", "release a module version", "git tag version", "module versions", "go module versioning", "new module version", "module version"},
		body:     "Go modules are released with semantic version tags in git. The tag v1.2.3 means major, minor, patch: raise major for breaking API changes, minor for backward-compatible features, and patch for fixes. Tag the commit, push the tag, and users fetch that exact version with go get; go list -m -versions shows what is published. Never move or reuse a published tag, since users' builds pin it, and use suffixes like v1.3.0-beta for pre-releases.",
	},
	{
		title:    "go workspaces and private modules",
		keywords: []string{"go workspaces", "go.work file", "private modules", "GOPRIVATE", "go work init", "multiple modules", "workspace", "go workspace"},
		body:     "Since Go 1.18, a go.work file makes several local modules build together, so a change in a shared module is used by your services immediately, without temporary replace lines in go.mod. Create one with go work init and add modules with go work use. Keep go.work out of version control and test with GOWORK=off before releasing. For company modules, set GOPRIVATE to your private path patterns so the Go command skips the public proxy and checksum database when fetching them.",
	},
	{
		title:    "building an mcp client",
		keywords: []string{"building an mcp client", "mcp client", "model context protocol", "mcp.NewClient", "ListTools", "CommandTransport"},
		body:     "An MCP client connects your program to an MCP server so a model can use tools the server offers. It needs the external github.com/modelcontextprotocol/go-sdk package; the standard library has no MCP support. Create the client with mcp.NewClient, connect over a stdio CommandTransport or HTTP, discover tools with ListTools, and invoke one with CallTool. Always close the session and set deadlines.",
	},
	{
		title:    "building an mcp server",
		keywords: []string{"building an mcp server", "mcp server", "mcp.NewServer", "mcp.AddTool", "model context protocol server", "serve tools over mcp", "stdio transport"},
		body:     "An MCP server exposes your Go functions as tools that AI applications can discover and call over the Model Context Protocol. It needs the external github.com/modelcontextprotocol/go-sdk package; the standard library has no MCP support. Create one with mcp.NewServer, register typed handlers with mcp.AddTool, and run it over stdio or HTTP. Keep tools narrow, validate inputs, and send logs to stderr, never stdout.",
	},
	{
		title:    "tool calling loop",
		keywords: []string{"tool calling loop", "tool-calling loop", "implementing a tool calling loop", "agent tool calls", "tool registry", "call tools from a model", "bounded iterations"},
		body:     "A tool-calling loop lets a model ask your program to run functions, then reads the results and either asks again or gives a final answer. You write the loop yourself; the standard library has no agent framework, and model access needs an external provider package. Keep a typed registry of allowed tools, validate arguments, cap steps, time, and output size, and stop on cancellation.",
	},
	{
		title:    "loading agent skills with genkit",
		keywords: []string{"genkit agent skills", "loading agent skills", "agent skills genkit", "SKILL.md", "use_skill", "firebase genkit", "progressive disclosure", "genkit", "agent skills"},
		body:     "Agent skills are folders holding a SKILL.md file with specialized instructions a model loads only when a task needs them, instead of carrying everything in every prompt. In Go this needs the external github.com/firebase/genkit/go package; the standard library has no agent skills support. Genkit first shows each skill description, then the model calls use_skill for the full text. Treat loaded skills as untrusted instructions, never as permission.",
	},
	{
		title:    "human approval for agent actions",
		keywords: []string{"human approval for agent actions", "requiring human approval", "human in the loop", "approve agent actions", "pending action", "human approval", "agent approval gate"},
		body:     "Human approval pauses an agent before a consequential tool call, like deleting data or spending money, until a person approves that exact action. You build this gate yourself; no standard library or framework feature does it for you. Store an immutable pending action with an expiry and idempotency key, show its real effects clearly, and run it only once, only for the approver who approved it.",
	},
	{
		title:    "streaming llm responses with sse",
		keywords: []string{"streaming llm responses", "server-sent events", "stream llm responses", "sse streaming", "bufio.Scanner", "parse sse events", "llm streaming"},
		body:     "Many model APIs stream answers as server-sent events, where one event can span several data: lines and ends with a blank line. Parse the stream yourself with bufio.Scanner, raising its default buffer limit, joining the data lines, and handling an event only when the blank line arrives. Respect context cancellation, close the response body, and treat partial output as incomplete until the final event arrives.",
	},
	{
		title:    "testing agent tool loops",
		keywords: []string{"testing agent tool loops", "test agent tool loops", "scripted model", "fake tools", "agent loop tests", "tool loop testing", "deterministic agent tests"},
		body:     "Testing an agent tool loop means testing your orchestration code without calling a live model. Replace the model with a scripted fake that returns planned turns, and replace real tools with fakes that record their arguments and call counts. Assert which tools ran, in what order, and that step limits, cancellation, and rejected actions stop the loop and run nothing they should not.",
	},
	{
		title:    "distributed locks",
		keywords: []string{"distributed locks", "distributed lock", "lock across servers", "etcd lock", "redis lock", "SetNX", "lock with ttl"},
		body:     "A distributed lock stops several machines or processes from changing the same shared resource at the same time, where a local sync.Mutex cannot help. Go has no built-in distributed lock; you need an external backing store and client, like etcd with go.etcd.io/etcd/client/v3 or Redis with SetNX. Give every lock a short expiry and a unique owner token, and only release a lock you still own.",
	},
	{
		title:    "function closures",
		keywords: []string{"function closures", "closures in go", "closure", "captured variables", "functions remember state", "anonymous function closure", "a closure", "closures work"},
		body:     "A closure is a function that remembers and can change variables from the scope where it was created, even after that scope has finished. You make one by returning an anonymous function that uses an outer variable, like a counter that keeps its count between calls. Closures suit small stateful helpers and callbacks, but shared captured variables need care, and synchronization, in concurrent code.",
	},
	{
		title:    "functional options pattern",
		keywords: []string{"functional options", "function options pattern", "functional options pattern", "configure a constructor", "optional parameters", "WithTimeout option", "flexible api configuration"},
		body:     "Functional options configure a constructor with readable, optional settings instead of a long parameter list or a big config struct callers must fill fully. Define an option type as a function that changes a config value, give helpers like WithTimeout, and let the constructor apply sensible defaults before running each option passed in. Add new options later without breaking existing callers.",
	},
	{
		title:    "iterators and the iter package",
		keywords: []string{"iterators", "iter package", "iter.Seq", "iter.Seq2", "range over iterator", "slices.Collect", "yield function", "iterator", "an iterator"},
		body:     "Since Go 1.23, an iterator is a function that hands values to a range loop one at a time, using the iter package types iter.Seq and iter.Seq2. Write one as a function returning a Seq whose inner function calls yield for each value and stops when yield returns false. Slices and maps helpers produce iterators, and slices.Collect gathers one back into a slice.",
	},
	{
		title:    "leader election",
		keywords: []string{"leader election", "elect a leader", "leader election etcd", "etcd concurrency", "campaign for leadership", "distributed leader", "single leader"},
		body:     "Leader election makes several copies of a service agree that only one of them leads and does leader-only work at a time. Do not write consensus yourself: use a coordination service like etcd, through the external go.etcd.io/etcd/client/v3 concurrency package, where candidates campaign on a lease-backed session. If the leader dies, its lease expires and another candidate takes over. Stop leading immediately when the session ends.",
	},
	{
		title:    "project layout conventions",
		keywords: []string{"project layout", "project layout conventions", "organize a go project", "cmd internal pkg folders", "internal package folder", "go project structure"},
		body:     "A Go project layout is a convention for organizing folders so readers know where things live. Put each program entry point under cmd, with its main package there, code only this project may import under internal, and reusable library code in ordinary packages, often under pkg. These folders are conventions, not requirements: a small project can keep one package at the root, and over-splitting hurts beginners.",
	},
	{
		title:    "retries with exponential backoff",
		keywords: []string{"retries with exponential backoff", "exponential backoff", "retry transient failures", "backoff and jitter", "retry a request", "jitter"},
		body:     "Retries with exponential backoff repeat a failed operation after waits that double each time, capped at a maximum, so a struggling service gets breathing room. Add jitter, random variation, so many clients do not retry in lockstep, and honor a Retry-After header when one arrives. Retry only transient, idempotent operations, mark permanent errors as unretryable, and bound the whole loop with a context deadline.",
	},
	{
		title:    "variadic functions",
		keywords: []string{"variadic functions", "variadic parameters", "variable number of arguments", "pass a slice to variadic", "three dots parameter", "variadic", "variadic function"},
		body:     "A variadic function accepts any number of final arguments of one type, declared with three dots, like sum(nums ...int) int. Inside, those arguments arrive as a slice you can range over, and calling with zero arguments is allowed. A variadic parameter must come last, after any fixed parameters. Pass an existing slice by following it with three dots, like sum(values...).",
	},
	{
		title:    "assertions in tests",
		keywords: []string{"test assertions", "assertions", "assert", "testify", "cmp.Diff", "t.Errorf"},
		body:     "Go's testing package has no built-in assert helpers, so the plain style compares got with want and reports mismatches with t.Errorf, using t.Fatalf when continuing makes no sense. For structs and other nested values, the external go-cmp package prints a readable diff from cmp.Diff. The external testify package offers assert and require helpers instead; require stops the test, assert records and continues.",
	},
	{
		title:    "benchmarks",
		keywords: []string{"benchmarks", "benchmark", "go benchmark", "run a benchmark", "measure performance", "benchmem"},
		body:     "Benchmarks measure how fast code runs. Write func BenchmarkName(b *testing.B) in a _test.go file and run go test -bench=. to repeat the code and time it. Since Go 1.24, loop with for b.Loop() instead of counting b.N yourself. Do setup before the loop or call b.ResetTimer, and add -benchmem to count allocations while you compare approaches.",
	},
	{
		title:    "fuzzing",
		keywords: []string{"fuzzing", "fuzz test", "f.Add", "fuzz", "seed inputs", "go test -fuzz"},
		body:     "Fuzzing feeds your code many random and mutated inputs to find crashes and broken rules that hand-picked examples miss. Since Go 1.18, write func FuzzName(f *testing.F), add seed inputs with f.Add, then inside f.Fuzz check properties that should always hold. Run it with go test -fuzz=FuzzName; failing inputs are saved as seeds for later runs.",
	},
	{
		title:    "in-memory test server",
		keywords: []string{"httptest.NewTestServer", "in-memory test server", "in-memory server", "fake network"},
		body:     "New in Go 1.27: httptest.NewTestServer creates a test HTTP server on an in-memory fake network instead of a real network port. Use srv.Client() to call it, because a normal http.Client cannot reach the fake network. It exists mainly for testing/synctest bubbles, where real network sockets are not allowed, so client timeouts follow the fake clock.",
	},
	{
		title:    "integration testing",
		keywords: []string{"integration testing", "integration test", "integration tests", "integration", "testcontainers-go package", "end-to-end tests", "skip integration test"},
		body:     "Integration tests check that real pieces of your program work together, like your code against a real database or HTTP service, instead of testing one function alone. Set up shared dependencies once in TestMain, and skip with t.Skip when a needed service or environment variable is missing so plain go test still passes. The external testcontainers-go package can start real databases in containers for these tests.",
	},
	{
		title:    "mocks",
		keywords: []string{"mocks", "mocking", "mock object", "gomock", "mockgen", "test doubles", "fake dependencies"},
		body:     "A mock is a fake version of an interface that your test controls: it returns what you tell it to and records how your code called it, so you can test one unit without a real database or service. Go has no standard mock library; the external go.uber.org/mock package and its mockgen tool generate mocks, and you set expectations with EXPECT. Mock only the dependencies you need, or tests turn brittle.",
	},
	{
		title:    "parallel tests",
		keywords: []string{"parallel tests", "t.Parallel", "run tests in parallel", "parallel test execution", "-parallel flag", "concurrent tests"},
		body:     "Calling t.Parallel() at the top of a test or subtest lets it pause and then run at the same time as other parallel tests, which shortens a slow suite. The go test -parallel flag caps how many run at once. Only mark tests parallel when they are fully independent: no shared variables, files, ports, or global state, or you will trade speed for flaky failures and data races.",
	},
	{
		title:    "race detector",
		keywords: []string{"race detector", "data race detector", "race detection", "-race flag", "go test -race", "detect races", "detector"},
		body:     "The race detector watches your program as it runs and reports data races: two goroutines touching the same memory unsynchronized, with at least one writing. Run go test -race ./... or build and run with -race, then read the report, which names the goroutines and lines involved. Fix races with a mutex, channels, or atomics. The detector slows programs down, so use it in testing and CI, not in production.",
	},
	{
		title:    "test context working directory and artifacts",
		keywords: []string{"t.Context", "t.Chdir", "t.ArtifactDir", "test context", "test artifacts", "working directory", "artifactdir"},
		body:     "Recent Go adds three helpers on testing.T. Since Go 1.24, t.Context() returns a context that is canceled when the test ends, and t.Chdir(dir) changes the working directory for the test and restores it afterwards, though it panics in parallel tests. New in Go 1.26, t.ArtifactDir() gives a folder for output files you want kept, preserved when you run go test with -artifacts.",
	},
	{
		title:    "test coverage",
		keywords: []string{"test coverage", "coverage", "coverprofile", "coverage profile", "go tool cover", "covermode", "measure coverage"},
		body:     "Test coverage measures how much of your code your tests actually run. Plain go test -cover ./... prints a percentage; go test -coverprofile=coverage.out ./... saves a profile you can explore with go tool cover -html=coverage.out or summarize per function with -func. Use -covermode=atomic together with -race. Coverage shows untested lines, but a high number alone does not prove the tests check the right things.",
	},
	{
		title:    "synctest",
		keywords: []string{"synctest", "fake clock", "virtual time", "synctest bubble", "test concurrent code"},
		body:     "Stable since Go 1.25: testing/synctest runs a test inside a bubble where time is fake, so code waiting on timers or tickers runs instantly and always in the same order. Wrap the test in synctest.Test and call synctest.Wait to let every goroutine block before you check results. It replaces the experimental Go 1.24 API, so ignore older examples using synctest.Run.",
	},
	{
		title:    "cryptotest",
		keywords: []string{"cryptotest", "cryptotest.TestBlock", "crypto testing"},
		body:     "New in Go 1.26: testing/cryptotest is a standard test helper that checks your own cryptographic code behaves correctly. Pass a function that builds your value to cryptotest.TestBlock for a block cipher, TestHash for a hash, or TestAEAD for an AEAD, and it runs a shared set of edge-case checks for you. It is a correctness baseline only; it does not prove your key handling or nonce use is safe.",
	},
	{
		title:    "testing http handlers with httptest",
		keywords: []string{"httptest", "test handlers", "response recorder", "fake api server", "handler tests"},
		body:     "The net/http/httptest package tests HTTP handlers without a real network. Build a request with httptest.NewRequest, call your handler with an httptest.NewRecorder, then check the recorder's status code, headers, and body. When code under test must make real HTTP calls, httptest.NewServer starts a throwaway server on a local port instead; point your client at its URL and remember to Close it.",
	},
	{
		title:    "subtests",
		keywords: []string{"subtests", "subtest", "t.Run", "run a subtest", "named subtests", "test cases"},
		body:     "Subtests are named tests inside a test, started with t.Run(name, func(t *testing.T) {...}). Each subtest gets its own t, so a failure names the exact case that broke, and you can rerun just it with go test -run TestName/case. Give cases short, clear names and keep them independent of each other. Subtests are how table-driven tests usually run each row of the table.",
	},
	{
		title:    "example functions",
		keywords: []string{"example function", "example functions", "example test", "ExampleF", "ExampleT_M", "executable documentation"},
		body:     "Example functions are executable documentation: they show real code using your package and appear in its godoc and pkg.go.dev pages. Write them in _test.go files: func Example() covers the whole package, ExampleF() a function F, ExampleT() a type T, and ExampleT_M() a method M on T. Append a lowercase suffix, like ExampleF_suffix, for extra variants of the same subject.",
	},
	{
		title:    "example output comments",
		keywords: []string{"example output", "unordered output", "output comment", "Output:", "verify output", "expected output"},
		body:     "An example only runs as a test when its last lines carry an // Output: comment listing exactly what the code prints; go test runs the example and compares its real output line by line, failing on any difference. When line order does not matter, write // Unordered output: instead. Without an output comment, the example is compiled but never run.",
	},
	{
		title:    "new in go 1.24",
		keywords: []string{"new in go 1.24", "go 1.24", "go 1.24 features", "what is new in go 1.24", "go 1.24 release"},
		body:     "Go 1.24 is a past release whose features are part of today's defaults. It added generic type aliases, weak pointers in the weak package, os.OpenRoot for directory-scoped file access, runtime.AddCleanup as the modern replacement for finalizers, b.Loop() for benchmark loops, and the go.mod tool directive for pinning developer tools. testing/synctest's fake clock arrived as an experiment.",
	},
	{
		title:    "new in go 1.23",
		keywords: []string{"new in go 1.23", "go 1.23", "go 1.23 features", "what is new in go 1.23", "go 1.23 release"},
		body:     "Go 1.23 is a past release whose features are part of today's defaults. Its headline change was range-over-function iterators: the iter.Seq and iter.Seq2 types that slices and maps helpers now build on. It also added the unique package for canonical values, slices.Repeat, and timers and tickers that the garbage collector can reclaim as soon as nothing references them.",
	},
	{
		title:    "new in go 1.22",
		keywords: []string{"new in go 1.22", "go 1.22", "go 1.22 features", "what is new in go 1.22", "go 1.22 release"},
		body:     "Go 1.22 is a past release whose features are part of today's defaults. It fixed loop variables so each iteration gets fresh variables instead of sharing one, let for loops range over integers, added the math/rand/v2 package and slices.Concat, and upgraded the net/http ServeMux with method and wildcard routing patterns like GET /users/{id}.",
	},
	{
		title:    "new in go 1.21",
		keywords: []string{"new in go 1.21", "go 1.21", "go 1.21 features", "what is new in go 1.21", "go 1.21 release"},
		body:     "Go 1.21 is a past release whose features are part of today's defaults. It added the min, max, and clear builtins, structured logging with the log/slog package, the generic slices, maps, and cmp helper packages, sync.OnceValue and friends, and profile-guided optimization (PGO), which tunes a build using a profile from a real run.",
	},
	{
		title:    "new in go 1.20",
		keywords: []string{"new in go 1.20", "go 1.20", "go 1.20 features", "what is new in go 1.20", "go 1.20 release"},
		body:     "Go 1.20 is a past release whose features are part of today's defaults. Errors learned to wrap several causes at once, through errors.Join and format strings with multiple %w verbs, so errors.Is and errors.As check the whole tree. It also added context.WithCancelCause, which records why a context was canceled, and conversion from a slice to an array value.",
	},
	{
		title:    "new in go 1.19",
		keywords: []string{"new in go 1.19", "go 1.19", "go 1.19 features", "what is new in go 1.19", "go 1.19 release"},
		body:     "Go 1.19 is a past release whose features are part of today's defaults. It added the typed atomic values in sync/atomic — atomic.Int64, atomic.Bool, atomic.Pointer — a soft memory limit for the garbage collector set with GOMEMLIMIT or debug.SetMemoryLimit, and richer doc comments with links, lists, and headings.",
	},
	{
		title:    "new in go 1.18",
		keywords: []string{"new in go 1.18", "go 1.18", "go 1.18 features", "what is new in go 1.18", "go 1.18 release"},
		body:     "Go 1.18 was the biggest release in years: generics (type parameters for functions and types) landed after a decade of design. The same release added fuzzing to go test, go.work workspaces for building several modules together, and build and version-control information embedded in every binary, which go version -m can read back.",
	},
	{
		title:    "older go versions",
		keywords: []string{"older go versions", "go versions before 1.18", "go 1.17", "go 1.16", "go 1.15", "go 1.14", "go 1.13", "go 1.12", "go 1.11", "go 1.10", "go 1.9", "go 1.8", "go 1.7", "go 1.6", "go 1.5", "go 1.4", "go 1.3", "go 1.2", "go 1.1", "go 1.0"},
		body:     "The pre-generics releases laid the groundwork today's Go stands on. Modules arrived experimentally in Go 1.11 and became the default by Go 1.16, which also added the embed package and io/fs file-system interfaces. Go 1.13 brought error wrapping with %w plus errors.Is and errors.As, Go 1.7 added the context package and subtests, Go 1.8 gave HTTP servers graceful shutdown, and Go 1.9 added type aliases and sync.Map.",
	},
}

func init() {
	goKnowledge = append(goKnowledge, goKnowledgeCookbook...)
}
