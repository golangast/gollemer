package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"

	"github.com/golangast/gollemer/pkg/ast"
	"github.com/golangast/gollemer/pkg/engine"
)

func main() {
	runClassic(os.Args[1:])
}

// runClassic is the original flag-based gollemer CLI: scripted code
// generation, the full orchestration pipeline, mock demos, impact
// analysis, and style inference. The interactive shell in main.go
// delegates here whenever any dash-flag is present on the command line.
func runClassic(args []string) {
	fs := flag.NewFlagSet("gollemer", flag.ExitOnError)
	dir := fs.String("dir", ".", "Go module directory to generate code for and validate against")
	prompt := fs.String("prompt", "", "code generation task, e.g. \"write a test for Add\"")
	maxAttempts := fs.Int("max-attempts", 3, "maximum generate-and-repair attempts")
	mock := fs.Bool("mock", false, "run the full pipeline end-to-end with a scripted mock LLM (no API needed)")
	impact := fs.Bool("impact", false, "calculate the impact radius of a symbol instead of generating code")
	symbol := fs.String("symbol", "", "symbol for -impact, e.g. \"User\" or \"Store.Get\"")
	style := fs.Bool("style", false, "infer the repository's coding conventions instead of generating code")
	flow := fs.Bool("flow", false, "run one prompt through the beginner pipeline: synthesis, safety, tuning, trace (plain words)")
	pipeline := fs.Bool("pipeline", false, "run the full orchestration pipeline (graph context, MCTS, proving, self-heal, auto-tune, commit)")
	candidates := fs.Int("candidates", 3, "LLM candidates drafted per task in -pipeline mode")
	fs.Usage = func() {
		fmt.Fprintf(fs.Output(), "Usage:\n")
		fmt.Fprintf(fs.Output(), "  gollemer                        # interactive natural-language shell\n")
		fmt.Fprintf(fs.Output(), "  gollemer -dir ./myproject -prompt \"write a test for Add\"\n")
		fmt.Fprintf(fs.Output(), "  gollemer -dir ./myproject -prompt \"...\" -pipeline [-candidates 5]\n")
		fmt.Fprintf(fs.Output(), "  gollemer -flow -prompt \"build a list of squares\"\n")
		fmt.Fprintf(fs.Output(), "  gollemer -mock                   # full pipeline demo with a scripted mock LLM\n")
		fmt.Fprintf(fs.Output(), "  gollemer -dir ./myproject -impact -symbol \"User\"\n")
		fmt.Fprintf(fs.Output(), "\nFlags:\n")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		log.Fatalf("parse flags: %v", err)
	}

	if *mock {
		runPipelineDemo()
		return
	}
	abs, err := filepath.Abs(*dir)
	if err != nil {
		log.Fatalf("resolve dir: %v", err)
	}
	if *impact {
		if *symbol == "" {
			fmt.Fprintln(os.Stderr, "missing required -symbol for -impact")
			fs.Usage()
			os.Exit(2)
		}
		codeCtx, err := ast.LoadPackageContext(abs)
		if err != nil {
			log.Fatalf("load package context: %v", err)
		}
		refs, err := ast.CalculateImpactRadius(codeCtx, *symbol)
		if err != nil {
			log.Fatalf("%v", err)
		}
		for _, ref := range refs {
			fmt.Println(" ", ref)
		}
		return
	}
	if *style {
		codeCtx, err := ast.LoadPackageContext(abs)
		if err != nil {
			log.Fatalf("load package context: %v", err)
		}
		repoStyle, err := ast.InferRepositoryStyle(codeCtx)
		if err != nil {
			log.Fatalf("%v", err)
		}
		fmt.Printf("error handling: %s\n", repoStyle.ErrorHandlingStyle)
		fmt.Printf("struct tags:    %v\n", repoStyle.CommonStructTags)
		fmt.Printf("uses context:   %v\n", repoStyle.UsesContext)
		fmt.Printf("doc density:    %.2f\n", repoStyle.DocCommentDensity)
		fmt.Println("--- prompt guidelines ---")
		fmt.Print(repoStyle.SystemPromptGuidelines())
		return
	}
	if *flow {
		if *prompt == "" {
			fmt.Fprintln(os.Stderr, "missing required -prompt for -flow")
			fs.Usage()
			os.Exit(2)
		}
		// The codebase context is best-effort: it names the trace file
		// after the package, but the pipeline works without one.
		var codeCtx *ast.CodebaseContext
		if cc, err := ast.LoadPackageContext(abs); err == nil {
			codeCtx = cc
		}
		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		defer cancel()
		res, err := engine.ExecutePipeline(ctx, *prompt, codeCtx)
		if err != nil {
			log.Fatalf("flow: %v", err)
		}
		fmt.Println(res.RenderDiagram(*prompt) + "\n\n" + res.RenderBeginner())
		return
	}
	if *prompt == "" {
		fmt.Fprintln(os.Stderr, "missing required -prompt (or use -mock for a demo, -impact for impact analysis)")
		fs.Usage()
		os.Exit(2)
	}
	fmt.Printf("LLM: %s\n", llmClient.Describe())

	if *pipeline {
		opts := DefaultPipelineOptions()
		opts.NumCandidates = *candidates
		opts.MaxHealAttempts = *maxAttempts
		res, err := RunPipeline(abs, *prompt, opts)
		if err != nil {
			log.Fatalf("%v", err)
		}
		fmt.Println("--- pipeline result ---")
		fmt.Printf("winner:   %s (fitness %.1f)\n", res.WinnerID, res.Fitness)
		fmt.Printf("safety:   %d critical, %d warnings\n", res.SafetyCritical, res.SafetyWarnings)
		fmt.Printf("allocs:   %d -> %d /op\n", res.AllocsBefore, res.AllocsAfter)
		if len(res.Mutations) > 0 {
			fmt.Printf("mutations: %s\n", joinNames(res.Mutations))
		}
		if res.Healed {
			fmt.Printf("healed in %d attempt(s)\n", res.HealAttempts)
		}
		return
	}

	fmt.Printf("loading package context from %s...\n", abs)
	codeCtx, err := ast.LoadPackageContext(abs)
	if err != nil {
		log.Fatalf("load package context: %v", err)
	}
	chunks := loadChunks(abs, 300)
	fmt.Printf("context: %d structs, %d interfaces, %d code chunks\n",
		len(codeCtx.Structs), len(codeCtx.Interfaces), len(chunks))

	files, err := GenerateAndSelfHeal(abs, *prompt, codeCtx, chunks, *maxAttempts)
	if err != nil {
		log.Fatalf("%v", err)
	}
	fmt.Println("validated files:")
	for _, name := range sortedKeys(files) {
		fmt.Printf("  %s (%d bytes)\n", name, len(files[name]))
	}
}

// runPipelineDemo exercises the ENTIRE pipeline end-to-end against a
// tiny fixture project with a scripted mock LLM:
//
//   - candidate 1 drafts Concat with a trailing-space bug (fails the test)
//   - candidate 2 drafts a fmt.Sprintf version (also wrong, slower)
//   - candidate 3 does not compile
//   - the self-healing loop repairs candidate 1 from the exact test
//     failure, the auto-tuner converts its `+=` loop to strings.Builder,
//     and the result is committed transactionally.
func runPipelineDemo() {
	dir, err := os.MkdirTemp("", "pipeline-demo-*")
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(dir)
	write := func(name, content string) {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); err != nil {
			log.Fatal(err)
		}
	}
	write("go.mod", "module example.com/pipedemo\n\ngo 1.26.0\n")
	write("concat.go", "package pipedemo\n\n// Concat concatenates words with no separator.\n// (stub — the pipeline replaces this file)\nfunc Concat(words []string) string { return \"\" }\n")
	write("concat_test.go", `package pipedemo

import "testing"

func TestConcat(t *testing.T) {
	if got := Concat([]string{"go", "pher"}); got != "gopher" {
		t.Errorf("got %q, want %q", got, "gopher")
	}
}

func BenchmarkConcat(b *testing.B) {
	words := []string{"a", "b", "c"}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Concat(words)
	}
}
`)

	broken := "=== FILE: concat.go ===\npackage pipedemo\n\n// Concat concatenates words with no separator.\nfunc Concat(words []string) string {\n\tvar s string\n\tfor _, w := range words {\n\t\ts += w + \" \"\n\t}\n\treturn s\n}\n"
	sprintfBug := "=== FILE: concat.go ===\npackage pipedemo\n\nimport \"fmt\"\n\n// Concat concatenates words with no separator.\nfunc Concat(words []string) string {\n\ts := \"\"\n\tfor _, w := range words {\n\t\ts = fmt.Sprintf(\"%s%s \", s, w)\n\t}\n\treturn s\n}\n"
	compileErr := "=== FILE: concat.go ===\npackage pipedemo\n\n// Concat concatenates words with no separator.\nfunc Concat(words []string) string {\n\treturn\n}\n"
	fixed := "=== FILE: concat.go ===\npackage pipedemo\n\n// Concat concatenates words with no separator.\nfunc Concat(words []string) string {\n\tvar s string\n\tfor _, w := range words {\n\t\ts += w\n\t}\n\treturn s\n}\n"
	SetLLMClient(NewMockClient(broken, sprintfBug, compileErr, fixed))

	fmt.Println("demo: full pipeline — graph → prove → MCTS → self-heal → auto-tune → commit")
	res, err := RunPipeline(dir, "Write Concat(words []string) string that concatenates all words with no separator.", DefaultPipelineOptions())
	if err != nil {
		log.Fatalf("demo failed: %v", err)
	}
	fmt.Println("--- demo result ---")
	fmt.Printf("winner:    %s (fitness %.1f)\n", res.WinnerID, res.Fitness)
	fmt.Printf("healed:    %v in %d attempt(s)\n", res.Healed, res.HealAttempts)
	fmt.Printf("mutations: %v\n", res.Mutations)
	fmt.Printf("allocs:    %d -> %d /op\n", res.AllocsBefore, res.AllocsAfter)
	fmt.Println("committed files:")
	for _, name := range sortedKeys(res.Files) {
		fmt.Printf("  %s (%d bytes)\n", name, len(res.Files[name]))
	}
	committed, err := os.ReadFile(filepath.Join(dir, "concat.go"))
	if err != nil {
		log.Fatalf("demo: committed file missing: %v", err)
	}
	fmt.Println("--- committed concat.go ---")
	fmt.Print(string(committed))
}
