package simple

import "strings"

// CommandExample is a plain, strongly-typed training example from the basic
// go_edit_agent update corpus. It covers social prompts as well as file and code
// create/edit/delete operations.
type CommandExample struct {
	Type      string // "social", "file_create", "file_edit", "file_delete", or "code_update"
	Prompt    string // natural-language request
	Response  string // social response (only for social)
	CodeAfter string // transformed code or file contents for the action
}

// CommandLabels is a stable label ordering for the command classifier.
var CommandLabels = []string{
	"social",
	"file_create",
	"file_edit",
	"file_delete",
	"folder_create",
	"folder_delete",
	"folder_query",
	"code_update",
}

// CommandVocab is the fixed bag-of-words vocabulary used by BagOfWords.
// Words are intentionally small and task-relevant so the dense model can
// separate social chatter from file and code create/edit/delete commands.
var CommandVocab = []string{
	"hello", "how", "are", "you", "what", "can", "do", "thank", "for",
	"help", "add", "opening", "brace", "function", "header", "if", "condition",
	"struct", "for", "loop", "switch", "statement", "return", "empty", "string",
	"error", "ok", "status", "formatted", "message", "success", "json", "content",
	"type", "update", "code", "err", "please", "say", "hi", "your", "need",
	"modify", "go", "methods", "sintax", "syntax", "fix", "missing", "browse",
	"create", "edit", "delete", "remove", "write", "file", "folder", "directory",
	"path", "new", "rename", "save", "replace", "append", "main", "config", "json",
	"is", "list", "show", "query", "inspect", "exists", "contents",
}

// Commands returns the provided basic update-command corpus as typed Go
// structs. This is the dataset used by dense_train / dense-llm.
func Commands() []CommandExample {
	return []CommandExample{
		{Type: "social", Prompt: "hello how are you", Response: "I am doing well, ready to help you update your Go code."},
		{Type: "social", Prompt: "what can you do", Response: "I can process natural language requests to modify Go code, adjust AST nodes, and return code snippets."},
		{Type: "social", Prompt: "thank you for your help", Response: "You're welcome! Let me know if you need any more Go code updates."},
		{Type: "file_create", Prompt: "create file main.go", CodeAfter: "package main\n\nfunc main() {}\n"},
		{Type: "file_create", Prompt: "create new config.json", CodeAfter: "{\n  \"name\": \"demo\"\n}\n"},
		{Type: "file_edit", Prompt: "edit file main.go", CodeAfter: "package main\n\nfunc main() {\n\tprintln(\"updated\")\n}\n"},
		{Type: "file_edit", Prompt: "modify file config.json", CodeAfter: "{\n  \"name\": \"updated\"\n}\n"},
		{Type: "file_edit", Prompt: "fix file main.go", CodeAfter: "package main\n\nfunc main() {\n\tprintln(\"fixed\")\n}\n"},
		{Type: "file_delete", Prompt: "delete file temp.txt", CodeAfter: ""},
		{Type: "file_delete", Prompt: "remove old.log", CodeAfter: ""},
		{Type: "folder_create", Prompt: "create folder jim", CodeAfter: "mkdir -p jim"},
		{Type: "folder_create", Prompt: "make directory internal", CodeAfter: "mkdir -p internal"},
		{Type: "folder_create", Prompt: "create new directory config", CodeAfter: "mkdir -p config"},
		{Type: "folder_delete", Prompt: "delete folder old", CodeAfter: "rm -rf old"},
		{Type: "folder_delete", Prompt: "remove directory tmp", CodeAfter: "rm -rf tmp"},
		{Type: "folder_delete", Prompt: "delete cached folder build", CodeAfter: "rm -rf build"},
		{Type: "folder_query", Prompt: "what is folder jim", Response: "Folder jim is a directory that can contain files and subfolders."},
		{Type: "folder_query", Prompt: "show folder app", Response: "Folder app contains files and subfolders."},
		{Type: "folder_query", Prompt: "list directory src", Response: "Directory src contains source files and folders."},
		{Type: "code_update", Prompt: "add missing opening brace to function header", CodeAfter: "func Ping(w http.ResponseWriter, r *http.Request) {"},
		{Type: "code_update", Prompt: "add opening brace to if condition", CodeAfter: "if err != nil {"},
		{Type: "code_update", Prompt: "add opening brace to struct definition", CodeAfter: "type User struct {"},
		{Type: "code_update", Prompt: "add opening brace to for loop", CodeAfter: "for i := 0; i < 10; i++ {"},
		{Type: "code_update", Prompt: "add opening brace to switch statement", CodeAfter: "switch status {"},
		{Type: "code_update", Prompt: "create function ping", CodeAfter: "func Ping() string {\n\treturn \"pong\"\n}"},
		{Type: "code_update", Prompt: "edit function response", CodeAfter: "func Response() string {\n\treturn \"updated\"\n}"},
		{Type: "code_update", Prompt: "delete function cleanup", CodeAfter: ""},
		{Type: "code_update", Prompt: "return empty string on error", CodeAfter: "if err != nil {\n\treturn \"\"\n}"},
		{Type: "code_update", Prompt: "return ok status string", CodeAfter: "func Status() string {\n\treturn \"OK\"\n}"},
		{Type: "code_update", Prompt: "return formatted error message string", CodeAfter: "func GetErr() string {\n\treturn fmt.Sprintf(\"failed with code %d\", code)\n}"},
		{Type: "code_update", Prompt: "return success message string", CodeAfter: "func Response() string {\n\treturn \"operation completed successfully\"\n}"},
		{Type: "code_update", Prompt: "return json content type string", CodeAfter: "func ContentType() string {\n\treturn \"application/json\"\n}"},
	}
}

// CommandDataset converts the corpus into a Dataset using BagOfWords encoding.
// Each sample's label is the index of its Type in CommandLabels.
func CommandDataset() *Dataset {
	cmds := Commands()
	samples := make([]Sample, len(cmds))
	for i, c := range cmds {
		samples[i] = Sample{
			Input: BagOfWords(c.Prompt, CommandVocab),
			Label: LabelForCommand(c.Type),
		}
	}
	return NewDataset(42, samples...)
}

// ClassifyCommandType uses keyword heuristics to identify social vs file/code
// create/edit/delete requests before relying on the dense model.
func ClassifyCommandType(prompt string) string {
	lower := strings.ToLower(strings.TrimSpace(prompt))
	if lower == "" {
		return "social"
	}
	if strings.Contains(lower, "hello") || strings.Contains(lower, "hi ") || strings.Contains(lower, "hey") ||
		strings.Contains(lower, "how are you") || strings.Contains(lower, "what can you do") ||
		strings.Contains(lower, "thank you") || strings.Contains(lower, "thanks") ||
		strings.Contains(lower, "good morning") || strings.Contains(lower, "good evening") ||
		strings.Contains(lower, "good night") {
		return "social"
	}
	if strings.Contains(lower, "create file") || strings.Contains(lower, "new file") ||
		(strings.Contains(lower, "create") && strings.Contains(lower, "file")) {
		return "file_create"
	}
	if strings.Contains(lower, "create folder") || strings.Contains(lower, "new folder") ||
		(strings.Contains(lower, "create") && (strings.Contains(lower, "folder") || strings.Contains(lower, "directory"))) ||
		(strings.Contains(lower, "make") && (strings.Contains(lower, "folder") || strings.Contains(lower, "directory"))) {
		return "folder_create"
	}
	if strings.Contains(lower, "edit file") || strings.Contains(lower, "modify file") ||
		strings.Contains(lower, "update file") || strings.Contains(lower, "change file") ||
		strings.Contains(lower, "fix file") || strings.Contains(lower, "repair file") ||
		(strings.Contains(lower, "edit ") && strings.Contains(lower, "file")) ||
		(strings.Contains(lower, "fix ") && strings.Contains(lower, "file")) {
		return "file_edit"
	}
	if strings.Contains(lower, "delete file") || strings.Contains(lower, "remove file") ||
		(strings.Contains(lower, "delete ") && strings.Contains(lower, "file")) ||
		(strings.Contains(lower, "remove ") && strings.Contains(lower, "file")) {
		return "file_delete"
	}
	if strings.Contains(lower, "what is folder") || strings.Contains(lower, "show folder") ||
		strings.Contains(lower, "list folder") || strings.Contains(lower, "what is directory") ||
		strings.Contains(lower, "show directory") || strings.Contains(lower, "list directory") {
		return "folder_query"
	}
	if strings.Contains(lower, "delete folder") || strings.Contains(lower, "remove folder") ||
		(strings.Contains(lower, "delete") && (strings.Contains(lower, "folder") || strings.Contains(lower, "directory"))) ||
		(strings.Contains(lower, "remove") && (strings.Contains(lower, "folder") || strings.Contains(lower, "directory"))) {
		return "folder_delete"
	}
	if strings.Contains(lower, "create function") || strings.Contains(lower, "create struct") ||
		strings.Contains(lower, "create type") || strings.Contains(lower, "add function") ||
		strings.Contains(lower, "modify function") || strings.Contains(lower, "edit function") ||
		strings.Contains(lower, "update function") || strings.Contains(lower, "delete function") ||
		strings.Contains(lower, "remove function") || strings.Contains(lower, "return ") ||
		strings.Contains(lower, "add opening brace") || strings.Contains(lower, "add missing") {
		return "code_update"
	}
	return "social"
}

// LabelForCommand returns the classifier label index for a command type.
func inferTargetFromPrompt(prompt string) (string, string) {
	fields := strings.Fields(strings.TrimSpace(prompt))
	if len(fields) == 0 {
		return "", ""
	}

	for i, token := range fields {
		switch strings.ToLower(token) {
		case "file", "/file":
			if i+1 < len(fields) {
				return strings.Join(fields[i+1:], " "), "file"
			}
		case "folder", "/folder", "directory", "/directory":
			if i+1 < len(fields) {
				return strings.Join(fields[i+1:], " "), "folder"
			}
		}
	}

	// Last-resort path extraction for explicit prompts that mention a file or
	// folder and then name a path-like token.
	for i := 0; i < len(fields)-1; i++ {
		lower := strings.ToLower(fields[i])
		if lower != "file" && lower != "/file" && lower != "folder" && lower != "/folder" && lower != "directory" && lower != "/directory" {
			continue
		}
		if i+1 < len(fields) {
			return strings.Join(fields[i+1:], " "), lower
		}
	}

	return "", ""
}

func InferTargetFromPrompt(prompt string) string {
	target, _ := inferTargetFromPrompt(prompt)
	return target
}

func LabelForCommand(commandType string) int {
	for i, l := range CommandLabels {
		if l == commandType {
			return i
		}
	}
	return 0 // default social
}

// WordOverlap counts how many vocabulary words a prompt shares with a corpus
// prompt. Used for nearest-neighbor fallback and demonstration output.
func WordOverlap(prompt string, example CommandExample) int {
	q := BagOfWords(prompt, CommandVocab)
	e := BagOfWords(example.Prompt, CommandVocab)
	count := 0
	for i := range q {
		if q[i] == 1 && e[i] == 1 {
			count++
		}
	}
	return count
}

// MatchCommand returns the corpus command whose prompt shares the most
// vocabulary words with the given query. Ties break toward earlier entries.
func MatchCommand(query string) CommandExample {
	return MatchCommandFromExamples(query, Commands())
}

// MatchCommandFromExamples returns the command from the given examples that
// best matches the query. Matching precedence:
//
//  1. Exact prompt match.
//  2. Substring containment (query contains the example prompt or vice versa).
//  3. Word-overlap with ties broken toward longer (more specific) prompts.
func MatchCommandFromExamples(query string, examples []CommandExample) CommandExample {
	lq := strings.ToLower(strings.TrimSpace(query))

	// 1. Exact match.
	for _, c := range examples {
		if strings.EqualFold(strings.TrimSpace(c.Prompt), lq) {
			return c
		}
	}

	// 2. Substring containment.
	for _, c := range examples {
		lp := strings.ToLower(strings.TrimSpace(c.Prompt))
		if strings.Contains(lq, lp) || strings.Contains(lp, lq) {
			return c
		}
	}

	// 3. Word-overlap with tie-breaking toward longer prompts.
	best := -1
	bestLen := -1
	var bestCmd CommandExample
	for _, c := range examples {
		ov := WordOverlap(query, c)
		if ov <= 0 {
			continue
		}
		promptLen := len(strings.Fields(c.Prompt))
		if ov > best || (ov == best && promptLen > bestLen) {
			best = ov
			bestLen = promptLen
			bestCmd = c
		}
	}
	return bestCmd
}
