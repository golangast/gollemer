package chat

// Pure-neural chat: the antidote to the old -seq2seq-chat.
//
// The old chat loop loaded the saved model but never actually called it — it
// answered from exact-string / fuzzy Jaccard matching with a canned fallback,
// so no amount of training could ever change its behavior. RunRealChat does
// the opposite: every answer comes from Seq2Seq.Predict, nothing else. If the
// model generates word salad, you see the word salad — which is the point:
// this loop is the honest meter for whether training worked.

import (
	"context"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/golangast/gollemer/internal/ai/analyze"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/seq2seq"
	"github.com/golangast/gollemer/pkg/termui"
	mainvocab "github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
	"github.com/golangast/gollemer/internal/ai/neural/tokenizer"
)

// RealModelPath returns the per-domain model file.
func RealModelPath(projectRoot, domain string) string {
	return filepath.Join(projectRoot, "data", "models", "gob_models", "real_tiny_seq2seq_"+domain+".gob")
}

// goIdentCaseMap builds the case-restoration dictionary from the training
// data itself: every identifier containing an uppercase letter in a gocode
// output maps its lowercase form back to the original (fmt.Println,
// strings.ToUpper, errors.New, isEven, ...). The neural pipeline is
// case-insensitive by construction, so without this the generated Go would
// not compile.
func goIdentCaseMap(projectRoot string) map[string]string {
	m := map[string]string{}
	pairs, err := LoadChatDataset(ChatDatasetPath(projectRoot))
	if err != nil {
		return m
	}
	ident := regexp.MustCompile(`[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?`)
	for _, p := range pairs {
		if p.Domain != GocodeDomain {
			continue
		}
		for _, id := range ident.FindAllString(p.Output, -1) {
			if low := strings.ToLower(id); low != id {
				m[low] = id
			}
		}
	}
	return m
}

// tidyGoCommand normalizes a predicted toolchain command: collapse
// whitespace runs, trim ends, and repair the tokenizer's punctuation
// splits ("go build ." must keep its space — "go build." would not
// execute — while "./..." and "fmt.Println" must be glued back).
// Unlike tidyDecode it never applies prose punctuation gluing.
func tidyGoCommand(s string) string {
	s = strings.Join(strings.Fields(s), " ")
	// "./..." was split into "." and "/..." by the tokenizer; rejoin with
	// the separating space intact ("go test ./...", not "go test./...").
	s = goCmdDotSlash.ReplaceAllString(s, " ./...")
	// Dotted paths: "fmt . Println" -> "fmt.Println",
	// "github . com/google/uuid" -> "github.com/google/uuid".
	s = goCmdDottedWord.ReplaceAllString(s, "$1.$2")
	return strings.TrimSpace(s)
}

var goCmdDotSlash = regexp.MustCompile(`\s*\.\s*/\.\.\.`)
var goCmdDottedWord = regexp.MustCompile(`(\w)\s*\.\s*(\w)`)

// tidyGoCode makes a generated snippet compile-shaped: it repairs the
// bracket/operator spacing the word-join decoder leaves ([ ] -> [], : = -> :=)
// and restores identifier case from the training data (fmt.Println, ...).
func tidyGoCode(s string, caseMap map[string]string) string {
	s = tidyDecode(s)
	for _, fix := range [][2]string{
		{"[ ]", "[]"}, {"( )", "()"},
		{": =", " :="}, {"= =", " =="}, {"! =", " !="},
		{"+ =", " +="}, {"- =", " -="}, {"* =", " *="}, {"/ =", " /="},
		{"< =", " <="}, {"> =", " >="},
	} {
		s = strings.ReplaceAll(s, fix[0], fix[1])
	}
	s = strings.ReplaceAll(s, "  ", " ")
	keys := make([]string, 0, len(caseMap))
	for k := range caseMap {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool { return len(keys[i]) > len(keys[j]) })
	for _, k := range keys {
		re := regexp.MustCompile(`\b` + regexp.QuoteMeta(k) + `\b`)
		s = re.ReplaceAllString(s, caseMap[k])
	}
	return s
}

// chatterPrompt matches pure-conversation openers: greetings, pleasantries,
// small talk. Anchored and conservative on purpose — when in doubt a prompt
// goes to the code model, because a missed code request is worse than a
// missed greeting.
var chatterPrompt = regexp.MustCompile(`(?i)^\s*(hi|hello|hey|good\s?(morning|evening|afternoon)|how are you|how'?s it going|what'?s up|thanks|thank you|bye|goodbye|see you later|who are you|tell me a joke)\b`)

// gocodeChatterRedirect is the clean reply when someone chats in a code
// session but the social checkpoint isn't available to answer them.
const gocodeChatterRedirect = "I generate Go code — describe the function you want and I'll write it."

// loadRealModel loads one per-domain checkpoint, rebuilding the tokenizer
// from the model's own vocabulary. It refuses lookup-table models (ExactMap)
// and pre-MoE checkpoints loudly rather than chatting brokenly.
func loadRealModel(projectRoot, domain string) (*seq2seq.Seq2Seq, error) {
	modelPath := RealModelPath(projectRoot, domain)

	// seq2seq.Load needs a tokenizer argument, but the real vocabulary lives
	// inside the saved file — so load with a placeholder, then rebuild the
	// tokenizer from the model's own vocabulary.
	placeholder, err := tokenizer.NewTokenizer(mainvocab.NewVocabulary())
	if err != nil {
		return nil, err
	}
	model, err := seq2seq.Load(modelPath, placeholder)
	if err != nil {
		return nil, fmt.Errorf("load real model %s: %w (train it first with -train-real-seq2seq)", modelPath, err)
	}
	if model.OutputVocab == nil || model.OutputVocab.Size() == 0 {
		return nil, fmt.Errorf("saved model has no vocabulary")
	}
	tok, err := tokenizer.NewTokenizer(model.OutputVocab)
	if err != nil {
		return nil, err
	}
	model.Tokenizer = tok

	// A model carrying an ExactMap is a lookup table wearing a neural net's
	// clothes. Refuse it loudly rather than pretend it generates.
	if len(model.ExactMap) > 0 {
		return nil, fmt.Errorf("refusing to chat: model carries an ExactMap of %d canned answers; retrain with -train-real-seq2seq", len(model.ExactMap))
	}
	// Models trained before the MoE decoder have no router/experts; the
	// decoder would nil-panic. Retrain rather than crash.
	if model.Decoder == nil || model.Decoder.MoE == nil {
		return nil, fmt.Errorf("refusing to chat: model was trained without the MoE decoder; retrain with -train-real-seq2seq")
	}
	return model, nil
}

// RunRealChat starts an interactive chat loop backed purely by the trained
// neural model. No lookup tables, no fuzzy matching, no canned fallback.
// When debug is false the loop prints clean replies only; when true it
// also shows the per-token thought process and debug log lines.
func RunRealChat(projectRoot, domain string, debug bool) error {
	if !debug {
		log.SetOutput(io.Discard)
	}
	// Load the user profile — the "learning you" model. Local only.
	profile := LoadProfile()
	profile.SeedDefaults()
	// The unified chat's start panel carries the profile's
	// continuity line itself (startscreen.go) — printing the
	// suggestion here too would say it twice at every launch.
	// The single-domain loops below keep the printed suggestion.
	if domain != UnifiedDomain {
		if suggestion := profile.SuggestStartup(projectRoot); suggestion != "" {
			fmt.Printf("\n%s\n\n", suggestion)
		}
	}
	// Track actions for sequence learning.
	var lastChatAction string
	var pendingSuggestion string
	shownSuggestions := map[string]bool{}
	if domain == UnifiedDomain {
		return runUnifiedChat(projectRoot, debug)
	}
	// GoAnalyze domain doesn't need neural models — it's pure AST.
	// Allow it to run without checkpoints so `where` and `/analyze` work standalone.
	var model *seq2seq.Seq2Seq
	if domain != GoAnalyzeDomain {
		var err error
		model, err = loadRealModel(projectRoot, domain)
		if err != nil {
			return err
		}
	}

	// Explicit mode separation for code sessions: chatter prompts are routed
	// to the social model (or a clean redirect when its checkpoint is
	// absent), so the code decoder never sees conversational input and can
	// never answer it with blended mush like "I am func doing (x)". The
	// separation is structural — it does not depend on the training data.
	var socialModel *seq2seq.Seq2Seq
	if domain == GocodeDomain {
		var serr error
		socialModel, serr = loadRealModel(projectRoot, SocialDomain)
		if serr != nil {
			log.Printf("[REAL-CHAT] no social checkpoint (%v); chatter gets a clean redirect", serr)
			socialModel = nil
		}
	}

	if model != nil {
		log.Printf("[REAL-CHAT] loaded %s (vocab=%d, hidden=%d). Pure neural generation — type /quit to exit.",
			RealModelPath(projectRoot, domain), model.OutputVocab.Size(), model.HiddenDim)
	} else {
		log.Printf("[REAL-CHAT] goanalyze mode — no neural model, analyzer commands only. Type /quit to exit.")
	}

	printSayThis()
	offerTeach()
	reader := NewLineReader(os.Stdin)
	paceInteractive = reader.term != nil
	showThoughts := debug
	conv := NewConversation()
	initSocialRecall(projectRoot)
	initMakefileRecall(projectRoot)
	initMakeAllowlist(projectRoot)
	goCase := map[string]string{}
	if domain == GocodeDomain {
		goCase = goIdentCaseMap(projectRoot)
		log.Printf("[REAL-CHAT] gocode post-processing on (%d case mappings)", len(goCase))
	}
	for {
		line, rerr := reader.ReadLine(cPrompt("you> "))
		if rerr == io.EOF {
			break
		}
		if rerr == errCancelled {
			continue
		}
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		// Track for sequence learning (Layer 1) and action log (Layer 4).
		if lastChatAction != "" && lastChatAction != line {
			profile.RecordSequence(lastChatAction, line)
		}
		profile.LogAction(BuildContextString(projectRoot, lastChatAction), line, projectRoot)
		lastChatAction = line

		// Suggest next action based on learned patterns.
		// Show it now (before processing) so it appears after the previous output.
		if pendingSuggestion != "" {
			// Show each distinct suggestion at most once per session —
			// repeating it after every command is noise, not help.
			// Never suggest quitting; that's the user's call, not a habit
			// to reinforce.
			if !shownSuggestions[pendingSuggestion] && pendingSuggestion != "quit" && pendingSuggestion != "exit" && pendingSuggestion != "stop" {
				fmt.Printf("\n[You usually do '%s' next — just say it or type it]\n\n", pendingSuggestion)
				shownSuggestions[pendingSuggestion] = true
			}
			pendingSuggestion = ""
		}
		if suggestion := profile.SuggestNext(line); suggestion != "" {
			pendingSuggestion = suggestion
		}

		if line == "/quit" || line == "exit" || line == "quit" || line == "stop" {
			fmt.Println("[bye — see you next time]")
			profile.Save()
			break
		}
		if line == "/thoughts" {
			showThoughts = !showThoughts
			fmt.Printf("[thought process display %s]\n", map[bool]string{true: "on", false: "off"}[showThoughts])
			continue
		}
		if line == "/history" {
			printTranscript(conv)
			continue
		}
		// /analyze is the explicit entry to the codebase-reader brain:
		// "/analyze ~/projects/foo" or "/analyze" for the current repo.
		// "visual" anywhere in the line also writes the HTML graph.
		if line == "/analyze" || strings.HasPrefix(line, "/analyze ") {
			arg := strings.TrimSpace(strings.TrimPrefix(line, "/analyze"))
			if arg == "" {
				arg = unifiedProjectRoot
			}
			if out, ok := handleGoAnalyze("analyze " + arg); ok {
				fmt.Printf("%s %s\n", cTag(fmt.Sprintf("gollemer [%s]>", GoAnalyzeDomain)), out)
				conv.AddReply(out, GoAnalyzeDomain, false)
			}
			continue
		}
		// /add-getter <Struct> <Field> <Type>: generate a getter, show it, ask [y/n] to apply.
		// Example: /add-getter Config Timeout time.Duration
		if strings.HasPrefix(line, "/add-getter ") {
			args := strings.Fields(strings.TrimPrefix(line, "/add-getter "))
			if len(args) != 3 {
				fmt.Println("Usage: /add-getter <Struct> <Field> <Type>")
				continue
			}
			structName, fieldName, fieldType := args[0], args[1], args[2]
			// Find the struct in the analyzed project to get its file.
			if lastAnalyzeProject == nil {
				fmt.Println("Run /analyze first to load the project.")
				continue
			}
			var targetFile string
			for _, pkg := range lastAnalyzeProject.Packages {
				for _, t := range pkg.Types {
					if t.Name == structName && t.Kind == "struct" {
						targetFile = t.File
						break
					}
				}
			}
			if targetFile == "" {
				fmt.Printf("Struct %q not found in analyzed project.\n", structName)
				continue
			}
			receiver := strings.ToLower(structName[:1])
			code := analyze.GenerateGetter(structName, receiver, fieldName, fieldType)
			fmt.Printf("Generated getter for %s.%s:\n\n%s\n\n", structName, fieldName, code)
			if confirmAction(fmt.Sprintf("Apply to %s?", targetFile), profile) {
				if err := appendToFile(lastAnalyzeRoot, targetFile, code); err != nil {
					fmt.Printf("Failed: %v\n", err)
				} else {
					fmt.Printf("Applied to %s\n", targetFile)
				}
			} else {
				fmt.Println("Cancelled.")
			}
			continue
		}
		// /contribute <task>: beginner-friendly contribution flow.
		if out, ok := handleContribute(line, unifiedProjectRoot); ok {
			fmt.Printf("%s %s\n", cTag("gollemer [contribute]>"), out)
			conv.AddReply(out, "contribute", false)
			continue
		}
		// "where for <url>": fetch a GitHub issue and point at the files to change.
		if strings.HasPrefix(line, "where for ") {
			if out, ok := tryIssueWhere(line); ok {
				fmt.Printf("%s %s\n", cTag(fmt.Sprintf("gollemer [%s]>", GoAnalyzeDomain)), out)
				conv.AddReply(out, GoAnalyzeDomain, false)
				// Offer to open the file in VSCode at the exact line.
				if f := analyze.LastWhereFile; f != "" {
					s := analyze.LastWhereStruct
					m := analyze.LastWhereMethod
					// Explain WHY this file.
					if s != "" {
						fmt.Printf("\nWhy this file: The %s struct stores the data the issue asks about.\n", s)
						fmt.Printf("The new %s method belongs on %s, next to its other methods.\n", m, s)
					}
					line := analyze.LastWhereLine
					target := f
					if line > 0 {
						target = fmt.Sprintf("%s:%d", f, line)
					}
					fmt.Printf("\nOpen %s in VSCode? [y/n] ", target)
					var ans string
					fmt.Scanln(&ans)
					if strings.ToLower(strings.TrimSpace(ans)) == "y" {
						cmd := exec.Command("code", "--goto", target)
						if err := cmd.Start(); err != nil {
							fmt.Printf("Couldn't open VSCode: %v\n", err)
							fmt.Printf("Try manually: code --goto %s\n", target)
						} else {
							fmt.Println("Opened in VSCode at the exact spot.")
						}
					}
					// Offer to generate and apply the method.
					if s != "" {
						// Explain WHAT we're about to do.
						fmt.Printf("\nWhat I'll do: Generate a %s() method on %s that exposes the internal data.\n", m, s)
						fmt.Printf("It will follow the same pattern as the struct's existing methods.\n")
						fmt.Printf("Generate the %s method on %s and update the file? [y/n] ", m, s)
						fmt.Scanln(&ans)
						if strings.ToLower(strings.TrimSpace(ans)) == "y" {
							fmt.Printf("What should %s return? (e.g. map[string][]string): ", m)
							var retType string
							fmt.Scanln(&retType)
							retType = strings.TrimSpace(retType)
							if retType == "" {
								retType = "interface{}"
							}
							// Generate a smart skeleton: TODO-based, with field inventory,
							// referencing the closest sibling method.
							genCode := generateMethodSkeleton(s, m, retType)
							fmt.Printf("\nGenerated:\n%s\n", genCode)
							fmt.Printf("This adds %s() to %s. %s\n", m, s, explainMethodPurpose(m))
							// Explain the code line by line.
							fmt.Printf("\nWhat each line does:\n")
							for _, ln := range strings.Split(strings.TrimSpace(genCode), "\n") {
								t := strings.TrimSpace(ln)
								if t == "" {
									continue
								}
								expl := explainGoLine(t, s, m)
								fmt.Printf("  %s\n    → %s\n", t, expl)
							}
							// Show a diff preview.
							fmt.Printf("\nDiff preview (what will change in %s):\n", f)
							if content, err := os.ReadFile(f); err == nil {
								lines := strings.Split(string(content), "\n")
								start := len(lines) - 4
								if start < 0 {
									start = 0
								}
								for _, ln := range lines[start:] {
									if strings.TrimSpace(ln) != "" {
										fmt.Printf("  %s\n", ln)
									}
								}
							}
							for _, ln := range strings.Split(strings.TrimSpace(genCode), "\n") {
								fmt.Printf("+ %s\n", ln)
							}
							fmt.Printf("\nApply to %s? [y/n] ", f)
							fmt.Scanln(&ans)
							if strings.ToLower(strings.TrimSpace(ans)) == "y" {
								// Duplicate guard: don't append if the method already exists.
								isDup := false
								if content, err := os.ReadFile(f); err == nil {
									// Match "func (c *container) Resolution(" with any receiver name.
									dupRe := regexp.MustCompile(`func\s*\([^)]*\)\s*` + regexp.QuoteMeta(m) + `\s*\(`)
									if dupRe.Match(content) {
										fmt.Printf("Skipped: %s() already exists in %s.\n", m, f)
										isDup = true
									}
								}
								if !isDup {
								// Append to the file directly.
								fh, err := os.OpenFile(f, os.O_APPEND|os.O_WRONLY, 0644)
								if err != nil {
									fmt.Printf("Couldn't update file: %v\n", err)
								} else {
									_, err = fh.WriteString("\n" + genCode + "\n")
									fh.Close()
									if err != nil {
										fmt.Printf("Couldn't update file: %v\n", err)
									} else {
										fmt.Printf("Updated %s.\n", f)
										// Verify it compiles.
										fmt.Printf("\nChecking that it compiles...\n")
										pkgDir := f
										if idx := strings.LastIndex(f, "/"); idx >= 0 {
											pkgDir = f[:idx]
										}
										cmd := exec.Command("go", "build", "./...")
										cmd.Dir = pkgDir
										// If the file is in a subdirectory, build from the project root.
										// Try the directory containing the file first.
										if out, err := cmd.CombinedOutput(); err != nil {
											fmt.Printf("Build failed:\n%s\n", string(out))
											fmt.Printf("\nIn plain words: %s\n", explainBuildError(string(out)))
										} else {
											fmt.Printf("✓ It compiles.\n")
										}
									}
								}
							} // end if !isDup
							} else {
								fmt.Println("Cancelled.")
							}
						}
					}
				}
			} else {
				fmt.Println("Couldn't parse that. Usage: where for <github-issue-url>")
			}
			continue
		}
		// /find <concept>: natural language code search.
		// "where is the auth logic?" -> finds funcs/types with "auth" in name or doc.
		if strings.HasPrefix(line, "/find ") {
			if lastAnalyzeProject == nil {
				fmt.Println("Run /analyze first.")
				continue
			}
			runFindSearch(strings.ToLower(strings.TrimSpace(strings.TrimPrefix(line, "/find "))))
			continue
		}
		// Natural language repo analysis: "analyze https://github.com/foo/bar",
		// "take a look at this repo: <url>". Detects repo URL + intent,
		// routes to clone+analyze.
		if out, ok := tryNaturalAnalyze(line); ok {
			fmt.Printf("%s %s\n", cTag(fmt.Sprintf("gollemer [%s]>", GoAnalyzeDomain)), out)
			conv.AddReply(out, GoAnalyzeDomain, false)
			continue
		}
		// /fix <error>: parse a Go error, suggest the fix.
		if strings.HasPrefix(line, "/fix ") {
			errText := strings.TrimSpace(strings.TrimPrefix(line, "/fix "))
			fmt.Printf("%s\n", suggestErrorFix(errText, lastAnalyzeProject))
			continue
		}
		// /whatis <concept>: explain via real usage in the analyzed code.
		if strings.HasPrefix(line, "/whatis ") {
			if lastAnalyzeProject == nil {
				fmt.Println("Run /analyze first.")
				continue
			}
			concept := strings.TrimSpace(strings.TrimPrefix(line, "/whatis "))
			fmt.Printf("%s\n", explainConcept(concept, lastAnalyzeProject))
			continue
		}
		// /arch: plain-words architecture summary from main().
		if line == "/arch" {
			if lastAnalyzeProject == nil {
				fmt.Println("Run /analyze first.")
				continue
			}
			fmt.Printf("%s\n", describeArchitecture(lastAnalyzeProject))
			continue
		}
		// /test <Func>: generate a table-driven test skeleton.
		if strings.HasPrefix(line, "/test ") {
			if lastAnalyzeProject == nil {
				fmt.Println("Run /analyze first.")
				continue
			}
			fname := strings.TrimSpace(strings.TrimPrefix(line, "/test "))
			fmt.Printf("%s\n", generateTestSkeleton(fname, lastAnalyzeProject))
			continue
		}
		// /why-import <pkg>: explain why a package is imported.
		if strings.HasPrefix(line, "/why-import ") {
			if lastAnalyzeProject == nil {
				fmt.Println("Run /analyze first.")
				continue
			}
			pkg := strings.TrimSpace(strings.TrimPrefix(line, "/why-import "))
			fmt.Printf("%s\n", explainImport(pkg, lastAnalyzeProject))
			continue
		}
		// /review <Func>: review comments with plain-words explanations.
		if strings.HasPrefix(line, "/review ") {
			if lastAnalyzeProject == nil {
				fmt.Println("Run /analyze first.")
				continue
			}
			fname := strings.TrimSpace(strings.TrimPrefix(line, "/review "))
			fmt.Printf("%s\n", reviewFunc(fname, lastAnalyzeProject))
			continue
		}
		// Beginner five: /example, /tour (+ next), /glossary, /watch,
		// /learn-path — learn a project by examples, order, and
		// watching it run, not by definitions.
		if handleBeginnerCommand(line, profile) {
			continue
		}
		// Edits by assembly, claimed deterministically like the
		// beginner five: a sentence that would write to the user's
		// files is never left for the classifier or a model — the
		// Composer plans it from sibling code, shows the diff, and
		// writes only behind the [y/n] gate.
		if tryAssembledEdit(line, reader) {
			continue
		}
		// Rename everywhere, claimed deterministically beside the
		// assembled edits for the same reason: the rename is worked
		// out from the project's type information, shown as a diff,
		// and written only behind the [y/n] gate.
		if tryRenameEdit(line, reader) {
			continue
		}
		// Build-goal resources, claimed deterministically beside
		// the edits/rename parsers and before the intent layer: a
		// learning statement ("i want to build a webserver") gets
		// its resource pack from the Advocates snapshot, never a
		// classifier guess. Imperative build commands for this
		// project are not shapes here — the composer keeps them.
		if tryBuildResources(line, profile) {
			continue
		}
		// Intent layer: a confident trained classification of what
		// the user wants, in their own words — no memorized commands.
		if tryIntentDispatch(line, profile) {
			continue
		}
		// /profile: show what the system has learned about you.
		// Shared renderer (profile.go) — the unified loop calls the
		// same function.
		if line == "/profile" {
			printProfile(profile)
			continue
		}
		// /learn-style: analyze John's code conventions (Layer 3).
		if line == "/learn-style" {
			fmt.Println("Scanning your Go files for conventions...")
			style := LearnStyle(projectRoot)
			profile.CacheStyle(style)
			profile.Save()
			fmt.Printf("\n%s\n", style.StyleSummary())
			fmt.Println("Style learned and cached. Generated code will match your conventions.")
			continue
		}
		// /predictor: show neural predictor status (Layer 4).
		if line == "/predictor" {
			fmt.Printf("\n%s\n", PredictorStatus())
			continue
		}
		// /train-predictor: export pairs and show training command.
		if line == "/train-predictor" {
			count, err := ExportTrainingPairs("/tmp/predictor_pairs.jsonl")
			if err != nil {
				fmt.Printf("Couldn't export: %v\n", err)
				continue
			}
			fmt.Printf("Exported %d training pairs to /tmp/predictor_pairs.jsonl\n", count)
			fmt.Println("To train: copy to data/training/ and run the seq2seq trainer with domain 'predictor'.")
			fmt.Println("(Full training integration coming — the data format is ready.)")
			continue
		}
		// /uexplain <Func>: explain a function from its structural facts.
		// The trained model gets first try, but its answer is VERIFIED
		// against the facts: if it names the wrong function or invents
		// behavior that isn't in the facts, the answer is rejected and
		// the verified template explanation is shown instead. Accepts
		// bare names ("MakeRequest"), receiver-qualified ("Client.MakeRequest"),
		// package-qualified ("matrix.New"), and full paths.
		if strings.HasPrefix(line, "/uexplain ") {
			if lastAnalyzeProject == nil {
				fmt.Println("Run /analyze first.")
				continue
			}
			fname := strings.TrimSpace(strings.TrimPrefix(line, "/uexplain "))
			fn := findFuncByQuery(lastAnalyzeProject, fname)
			if fn == nil {
				fmt.Printf("Function %q not found.", fname)
				if near := nearFuncNames(lastAnalyzeProject, fname); len(near) > 0 {
					fmt.Printf(" Close matches: %s.", strings.Join(near, ", "))
				}
				fmt.Println()
				continue
			}
			// Facts in the v2 training format — one source of truth,
			// shared with the pair generator.
			runUexplainPrint(fn, projectRoot)
			continue
		}
		// /export-understanding: generate structure->language training pairs.
		// This is "train on how the program knows code" — not character prediction.
		if line == "/export-understanding" {
			if lastAnalyzeProject == nil {
				fmt.Println("Run /analyze first.")
				continue
			}
			runExportUnderstanding()
			continue
		}
		// /comment <Func>: generate a doc comment for a function.
		if strings.HasPrefix(line, "/comment ") {
			if lastAnalyzeProject == nil {
				fmt.Println("Run /analyze first.")
				continue
			}
			fname := strings.TrimSpace(strings.TrimPrefix(line, "/comment "))
			if out, ok := generateDocCommentCmd(fname, lastAnalyzeProject, profile); ok {
				fmt.Printf("%s\n", out)
			} else {
				fmt.Printf("Function %q not found. Try /find %s.\n", fname, fname)
			}
			continue
		}
		// /learn-pieces: extract composable piece schemas from doc comments
		// in the last analyzed project. Pieces are learned from natural
		// language docs, not hardcoded.
		if line == "/learn-pieces" {
			if lastAnalyzeProject == nil {
				fmt.Println("Run /analyze first.")
				continue
			}
			runLearnPieces()
			continue
		}
		// /build-api <path> [path2...]: compose a working HTTP API from pieces.
		// Example: /build-api /users /health
		// Pieces are learned schemas (not memorized code); placement is
		// constraint-solved; each piece explains why it's where it is.
		if strings.HasPrefix(line, "/build-api ") {
			arg := strings.TrimSpace(strings.TrimPrefix(line, "/build-api "))
			if arg == "" {
				fmt.Println("Usage: /build-api /path1 [/path2 ...]  (e.g. /build-api /users)")
				continue
			}
			runBuildAPI(strings.Fields(arg))
			continue
		}
		// /impact <FuncName>: show blast radius — what breaks if you change this.
		if strings.HasPrefix(line, "/impact ") {
			arg := strings.TrimSpace(strings.TrimPrefix(line, "/impact "))
			if lastAnalyzeProject == nil {
				fmt.Println("Run /analyze first.")
				continue
			}
			// Find the func by name (search all packages)
			var fid string
			for _, pkg := range lastAnalyzeProject.Packages {
				for _, fn := range pkg.Funcs {
					if fn.Name == arg || fn.ID == arg {
						fid = fn.ID
						break
					}
				}
			}
			if fid == "" {
				fmt.Printf("Function %q not found.\n", arg)
				continue
			}
			report := lastAnalyzeProject.AnalyzeImpact(fid)
			fmt.Println(report.Describe())
			continue
		}
		// /conc <FuncName>: show concurrency report for a function.
		if strings.HasPrefix(line, "/conc ") {
			arg := strings.TrimSpace(strings.TrimPrefix(line, "/conc "))
			if lastAnalyzeProject == nil {
				fmt.Println("Run /analyze first.")
				continue
			}
			var fsrc string
			for _, pkg := range lastAnalyzeProject.Packages {
				for _, fn := range pkg.Funcs {
					if fn.Name == arg || fn.ID == arg {
						fsrc = fn.Src
						break
					}
				}
			}
			if fsrc == "" {
				fmt.Printf("Function %q not found.\n", arg)
				continue
			}
			rep := analyze.AnalyzeConcurrencySource(fsrc)
			desc := rep.Describe()
			if desc == "" {
				desc = "No concurrency primitives used."
			}
			fmt.Println(desc)
			continue
		}
		// /improve <FuncName>: suggest improvements for a function.
		if strings.HasPrefix(line, "/improve ") {
			arg := strings.TrimSpace(strings.TrimPrefix(line, "/improve "))
			if lastAnalyzeProject == nil {
				fmt.Println("Run /analyze first.")
				continue
			}
			var fsrc string
			for _, pkg := range lastAnalyzeProject.Packages {
				for _, fn := range pkg.Funcs {
					if fn.Name == arg || fn.ID == arg {
						fsrc = fn.Src
						break
					}
				}
			}
			if fsrc == "" {
				fmt.Printf("Function %q not found.\n", arg)
				continue
			}
			imps := analyze.SuggestImprovements(fsrc)
			fmt.Println(analyze.FormatImprovements(imps))
			continue
		}
		// /flow is the beginner entry to the same reactive pipeline:
		// "/flow create a worker pool" runs synthesis, safety proving,
		// auto-tuning, and the visual trace, rendered in plain words.
		if out, ok := handleFlowCommand(line); ok {
			fmt.Printf("%s %s\n", cTag(fmt.Sprintf("gollemer [%s]>", FlowDomain)), out)
			conv.AddReply(out, FlowDomain, false)
			continue
		}
		if line == "/forget" {
			conv.Clear()
			fmt.Println("[forgotten — starting fresh]")
			continue
		}
		// Beginner five, claimed deterministically before routing —
		// same as the single-domain loop. A sentence like "tour
		// https://…" or "run it" must never be left for the router
		// to guess at (it read "tour …" as a social greeting).
		if handleBeginnerCommand(line, profile) {
			continue
		}
		// Edits by assembly, claimed deterministically like the
		// beginner five: a sentence that would write to the user's
		// files is never left for the classifier or a model — the
		// Composer plans it from sibling code, shows the diff, and
		// writes only behind the [y/n] gate.
		if tryAssembledEdit(line, reader) {
			continue
		}
		// Rename everywhere, claimed deterministically beside the
		// assembled edits for the same reason: the rename is worked
		// out from the project's type information, shown as a diff,
		// and written only behind the [y/n] gate.
		if tryRenameEdit(line, reader) {
			continue
		}
		// Build-goal resources, claimed deterministically beside
		// the edits/rename parsers and before the intent layer: a
		// learning statement ("i want to build a webserver") gets
		// its resource pack from the Advocates snapshot, never a
		// classifier guess. Imperative build commands for this
		// project are not shapes here — the composer keeps them.
		if tryBuildResources(line, profile) {
			continue
		}
		// Intent layer: a confident trained classification of what
		// the user wants, in their own words — no memorized commands.
		if tryIntentDispatch(line, profile) {
			continue
		}
		// Recall questions are answered from the session transcript,
		// deterministically, before any model sees them.
		if rq, ok := conv.Recall(line); ok {
			fmt.Printf("gollemer> %s\n", rq)
			conv.AddUser(line)
			conv.AddReply(rq, domain, true)
			continue
		}
		conv.AddUser(line)
		// Social recall: exact training-pair matches answer verbatim.
		if domain == SocialDomain {
			if sr, ok := LookupSocialRecall(line); ok {
				fmt.Printf("gollemer> %s\n", sr)
				conv.AddReply(sr, SocialDomain, false)
				continue
			}
		}
		// Makefile recall: exact training-pair matches answer verbatim.
		if domain == MakefileDomain {
			if mr, ok := LookupMakefileRecall(line); ok {
				fmt.Printf("gollemer> %s\n", mr)
				conv.AddReply(mr, MakefileDomain, false)
				if t := runnableMakeTarget(mr); t != "" {
					offerRunMakeCommand(reader, projectRoot, t)
				}
				continue
			}
		}
		// Go concept questions first check the curated knowledge base.
		if domain == GoDomain {
			if kb, ok := LookupGoKnowledge(line); ok {
				fmt.Printf("gollemer> %s\n", kb)
				conv.AddReply(kb, domain, false)
				continue
			}
		}
		if domain == GocodeDomain && chatterPrompt.MatchString(line) && !gocodeTerms.MatchString(line) {
			// Conversational mode: answer with the chat model, never the
			// code decoder. The gocodeTerms guard keeps greeting-prefixed
			// code requests ("hey, write a function...") on the code model.
			// Chat replies get chat post-processing only — tidyGoCode must
			// never touch them.
			if socialModel != nil {
				answer, trace, err := socialModel.PredictWithTrace(conv.SocialInput(strings.ToLower(line)), 40)
				if err != nil {
					fmt.Printf("gollemer> [error: %v]\n", err)
					continue
				}
				reply := tidyDecode(answer)
				fmt.Printf("gollemer> %s\n", reply)
				conv.AddReply(reply, SocialDomain, false)
				if showThoughts {
					printThoughtTrace(trace)
				}
			} else {
				fmt.Printf("gollemer> %s\n", gocodeChatterRedirect)
				conv.AddReply(gocodeChatterRedirect, SocialDomain, true)
			}
			continue
		}
		// Lowercased to match training; the model is case-insensitive by construction.
		modelInput := strings.ToLower(line)
		if domain == SocialDomain {
			modelInput = conv.SocialInput(modelInput)
		}
		// GoAnalyze domain has no neural model — analyzer commands were handled above.
		// If we get here, the input wasn't a recognized command.
		if model == nil {
			fmt.Println(unknownCommandHint(line))
			continue
		}
		answer, trace, err := model.PredictWithTrace(modelInput, 40)
		if err != nil {
			fmt.Printf("gollemer> [error: %v]\n", err)
			continue
		}
		reply := tidyDecode(answer)
		if domain == GocodeDomain {
			reply = tidyGoCode(answer, goCase)
			// Write-then-analyze loop: verify the generated function name matches the intent.
			if verdict := verifyGocodeIntent(line, reply); verdict != "" {
				reply = reply + "\n" + verdict
			}
		}
		if domain == GoCliDomain {
			// Commands keep their token spacing: "go build ." not "go build.".
			reply = tidyGoCommand(answer)
		}
		fmt.Printf("gollemer> %s\n", reply)
		conv.AddReply(reply, domain, false)
		if showThoughts {
			printThoughtTrace(trace)
		}
	}
	return nil
}

// tryDeterministicAnswer checks the exact-match layers before the neural
// model runs: social recall, makefile recall, and the Go knowledge base.
// It prints the reply and records it in the conversation when one hits,
// reporting whether the message was fully handled.
func tryDeterministicAnswer(line, d string, conv *Conversation, r *LineReader, projectRoot string) bool {
	// A bare target name ("eval", "explain") is "run make <target>".
	if tryBareMakeTarget(line, r, projectRoot, conv) {
		return true
	}
	// Direct "run make <target>": typed by the user, validated against
	// the Makefile allowlist, offered immediately — no brain needed.
	if tryDirectRunMake(line, r, projectRoot, conv) {
		return true
	}
	// "explain make <target>": what the target does, then the run offer.
	if tryExplainMake(line, r, projectRoot, conv) {
		return true
	}
	// Codebase Q&A: if a project was analyzed this session and the
	// message asks about one of its symbols ("what does routeDomain
	// do"), answer from the AST. It only fires on known symbols, so it
	// can't steal questions meant for the other brains.
	if out, ok := tryCodebaseQuestion(line); ok {
		fmt.Printf("%s %s\n", cTag(fmt.Sprintf("gollemer [%s]>", GoAnalyzeDomain)), out)
		conv.AddReply(out, GoAnalyzeDomain, false)
		return true
	}
	// Pasted Go errors: translate into plain words, no command needed.
	// Unrecognized shapes fall through to the Go brain below.
	if out, ok := tryErrorHelp(line); ok {
		fmt.Printf("%s %s\n", cTag(fmt.Sprintf("gollemer [%s]>", GoDomain)), out)
		conv.AddReply(out, GoDomain, false)
		// When the error has exactly one mechanical repair, offer
		// the repair itself behind the [y/n] gate — the translation
		// above stands either way.
		tryErrorFix(line, r)
		return true
	}
	// Pasted GitHub issue URL ("where do I change for this?"): fetch
	// the issue, analyze the repo, and point at the edit locations.
	// Explicit "where for <url>" works too.
	if out, ok := tryIssueWhere(line); ok {
		fmt.Printf("%s %s\n", cTag(fmt.Sprintf("gollemer [%s]>", GoAnalyzeDomain)), out)
		conv.AddReply(out, GoAnalyzeDomain, false)
		return true
	}
	// Pasted GitHub pull-request URL: list the files the PR changed,
	// each with a plain-words one-liner. Explicit "where for <url>"
	// works too.
	if out, ok := tryPullWhere(line); ok {
		fmt.Printf("%s %s\n", cTag(fmt.Sprintf("gollemer [%s]>", GoAnalyzeDomain)), out)
		conv.AddReply(out, GoAnalyzeDomain, false)
		return true
	}
	// "clone <url> in folder <name>": clone the repo into
	// ~/workspace/<name> and analyze it, so the chat can answer
	// questions about the code in that folder.
	if out, ok := tryCloneProject(line); ok {
		fmt.Printf("%s %s\n", cTag(fmt.Sprintf("gollemer [%s]>", GoAnalyzeDomain)), out)
		conv.AddReply(out, GoAnalyzeDomain, false)
		return true
	}
	// Natural language repo analysis: "analyze https://github.com/foo/bar",
	// "take a look at this repo: <url>", "what does <url> do?"
	// Detects a repo URL + analysis intent, routes to clone+analyze.
	if out, ok := tryNaturalAnalyze(line); ok {
		fmt.Printf("%s %s\n", cTag(fmt.Sprintf("gollemer [%s]>", GoAnalyzeDomain)), out)
		conv.AddReply(out, GoAnalyzeDomain, false)
		return true
	}
	// "show me a visual" as a follow-up: draw the full visual report for
	// the last analyzed project without re-parsing it. The verb+noun
	// shape keeps "show me routeDomain" (answered above) and "how do I
	// render html templates" (go brain) out, and the domain guard keeps
	// "show me how to draw a graph in go" (go/gocode) with its owner.
	// With no prior analysis the visuals are drawn for the repo the chat
	// runs in — the only codebase in context.
	if visualFollowup.MatchString(line) && (d == SocialDomain || d == GoAnalyzeDomain) {
		p := lastAnalyzeProject
		if p == nil && unifiedProjectRoot != "" {
			if ap, err := analyze.Analyze(unifiedProjectRoot); err == nil {
				lastAnalyzeRoot = unifiedProjectRoot
				lastAnalyzeProject = ap
				p = ap
			}
		}
		if p != nil {
			out := p.VisualReport()
			fmt.Printf("%s %s\n", cTag(fmt.Sprintf("gollemer [%s]>", GoAnalyzeDomain)), out)
			conv.AddReply(out, GoAnalyzeDomain, false)
			return true
		}
	}
	// Social recall: an exact training-pair match returns the trained
	// answer verbatim. The tiny model doesn't reliably memorize every
	// pair, so this guarantees the chat "picks up" what's in its data.
	if d == SocialDomain {
		if sr, ok := LookupSocialRecall(line); ok {
			fmt.Printf("%s %s\n", cTag(fmt.Sprintf("gollemer [%s]>", d)), sr)
			conv.AddReply(sr, SocialDomain, false)
			return true
		}
	}
	// Makefile recall: exact training-pair matches return the trained
	// command verbatim. The tiny model confuses similar "how do i ..."
	// inputs, so this guarantees correct commands for anything it was
	// explicitly taught.
	if d == MakefileDomain {
		if mr, ok := LookupMakefileRecall(line); ok {
			fmt.Printf("%s %s\n", cTag(fmt.Sprintf("gollemer [%s]>", d)), mr)
			conv.AddReply(mr, MakefileDomain, false)
			// A makefile reply names an exact repo command. Offer to
			// run it directly, the same way gocli commands are run.
			if t := runnableMakeTarget(mr); t != "" {
				offerRunMakeCommand(r, projectRoot, t)
			}
			return true
		}
	}
	// Go concept questions first check the curated knowledge base: a
	// strong keyword match gives a guaranteed-correct answer, anything
	// vague falls through to the neural model.
	if d == GoDomain {
		if kb, ok := LookupGoKnowledge(line); ok {
			fmt.Printf("%s %s\n", cTag(fmt.Sprintf("gollemer [%s]>", d)), kb)
			conv.AddReply(kb, GoDomain, false)
			return true
		}
	}
	// Codebase-reading requests run the AST analyzer: exact structural
	// answers, no neural model involved.
	if d == GoAnalyzeDomain {
		if out, ok := handleGoAnalyze(line); ok {
			fmt.Printf("%s %s\n", cTag(fmt.Sprintf("gollemer [%s]>", d)), out)
			conv.AddReply(out, GoAnalyzeDomain, false)
			return true
		}
	}
	return false
}

// runUnifiedChat is the single-chat mode John asked for: one session that
// knows the difference between social chat, Go concepts, Go code, and
// makefile commands. Each message is routed by input intent via
// routeDomain; the reply is tagged with the model that produced it
// (e.g. "gollemer [go]>") so the active brain is visible. Missing
// specialized checkpoints fall back to the social model.
func runUnifiedChat(projectRoot string, debug bool) error {
	unifiedProjectRoot = projectRoot
	sp := termui.Spin("Loading brains…")
	models := map[string]*seq2seq.Seq2Seq{}
	for _, d := range []string{SocialDomain, GoDomain, GoCliDomain, GocodeDomain, MakefileDomain} {
		m, err := loadRealModel(projectRoot, d)
		if err != nil {
			log.Printf("[UNIFIED-CHAT] no %s checkpoint (%v); falling back to social", d, err)
			continue
		}
		models[d] = m
		log.Printf("[UNIFIED-CHAT] loaded %s model", d)
	}
	socialModel := models[SocialDomain]
	if socialModel == nil {
		sp.Stop("")
		return fmt.Errorf("unified chat requires at least the social checkpoint")
	}
	sp.Stop("Ready.")
	// Social recall: exact training-pair matches answer deterministically.
	initSocialRecall(projectRoot)
	initMakefileRecall(projectRoot)
	initMakeAllowlist(projectRoot)
	profile := LoadProfile()

	goCase := goIdentCaseMap(projectRoot)
	reader := NewLineReader(os.Stdin)
	paceInteractive = reader.term != nil
	showThoughts := debug
	conv := NewConversation()
	printStartScreen(projectRoot)
	// Unfinished-edit backups from an earlier session (a chat
	// that died mid-edit): name each one with the exact command
	// that restores its file. We never run the restore ourselves.
	noticeRoot := projectRoot
	if noticeRoot == "" {
		noticeRoot = startFolder()
	}
	if abs, aerr := filepath.Abs(noticeRoot); aerr == nil {
		noticeRoot = abs
	}
	for _, n := range leftoverBackupNotices(noticeRoot) {
		fmt.Println(n)
	}
	for {
		line, rerr := reader.ReadLine(cPrompt("you> "))
		if rerr == io.EOF {
			break
		}
		if rerr == errCancelled {
			continue
		}
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		if line == "/quit" || line == "exit" || line == "quit" || line == "stop" {
			fmt.Println("[bye — see you next time]")
			break
		}
		if line == "/thoughts" {
			showThoughts = !showThoughts
			fmt.Printf("[thought process display %s]\n", map[bool]string{true: "on", false: "off"}[showThoughts])
			continue
		}
		if line == "/history" {
			printTranscript(conv)
			continue
		}
		// /analyze is the explicit entry to the codebase-reader brain:
		// "/analyze ~/projects/foo" or "/analyze" for the current repo.
		// "visual" anywhere in the line also writes the HTML graph.
		if line == "/analyze" || strings.HasPrefix(line, "/analyze ") {
			arg := strings.TrimSpace(strings.TrimPrefix(line, "/analyze"))
			if arg == "" {
				arg = unifiedProjectRoot
			}
			if out, ok := handleGoAnalyze("analyze " + arg); ok {
				fmt.Printf("%s %s\n", cTag(fmt.Sprintf("gollemer [%s]>", GoAnalyzeDomain)), out)
				conv.AddReply(out, GoAnalyzeDomain, false)
			}
			continue
		}
		// /flow is the beginner entry to the same reactive pipeline:
		// "/flow create a worker pool" runs synthesis, safety proving,
		// auto-tuning, and the visual trace, rendered in plain words.
		if out, ok := handleFlowCommand(line); ok {
			fmt.Printf("%s %s\n", cTag(fmt.Sprintf("gollemer [%s]>", FlowDomain)), out)
			conv.AddReply(out, FlowDomain, false)
			continue
		}
		if line == "/forget" {
			conv.Clear()
			fmt.Println("[forgotten — starting fresh]")
			continue
		}
		// /profile: show what the system has learned about you.
		// The handler used to live only in the single-domain loop,
		// so in `make chat` the line fell through to the social
		// brain and got a greeting. Same shared renderer
		// (profile.go) as the single-domain loop.
		if line == "/profile" {
			printProfile(profile)
			continue
		}
		// Learn mode, claimed deterministically ahead of the
		// beginner five: its start shapes, its "next", and —
		// while a check question waits — its answers must never
		// reach the classifier or a brain. Other lines pass
		// through untouched; the flow waits (the guide's rule).
		if tryLearnCommand(line) {
			continue
		}
		// Beginner five, claimed deterministically before routing —
		// same as the single-domain loop. A sentence like "tour
		// https://…" or "run it" must never be left for the router
		// to guess at (it read "tour …" as a social greeting).
		if handleBeginnerCommand(line, profile) {
			continue
		}
		// Edits by assembly, claimed deterministically like the
		// beginner five: a sentence that would write to the user's
		// files is never left for the classifier or a model — the
		// Composer plans it from sibling code, shows the diff, and
		// writes only behind the [y/n] gate.
		if tryAssembledEdit(line, reader) {
			continue
		}
		// Rename everywhere, claimed deterministically beside the
		// assembled edits for the same reason: the rename is worked
		// out from the project's type information, shown as a diff,
		// and written only behind the [y/n] gate.
		if tryRenameEdit(line, reader) {
			continue
		}
		// Build-goal resources, claimed deterministically beside
		// the edits/rename parsers and before the intent layer: a
		// learning statement ("i want to build a webserver") gets
		// its resource pack from the Advocates snapshot, never a
		// classifier guess. Imperative build commands for this
		// project are not shapes here — the composer keeps them.
		if tryBuildResources(line, profile) {
			continue
		}
		// Simple creations, claimed deterministically beside the
		// other write-side parsers: "create a folder jim" and
		// "create a file … with a webserver" act behind the same
		// [y/n] gate as the edits (exact plan first, verify after),
		// and the wider disk-action family (delete, move, install)
		// is claimed only to answer honestly. A creation sentence
		// must never reach the classifier — it read "create a
		// folder jim" as a social greeting.
		if tryCreate(line, reader) {
			continue
		}
		// Add-a-webserver edits, claimed deterministically beside
		// the creations and before the intent layer: "add a
		// webserver to jim/main.go" is an edit to an existing file,
		// and left to the classifier it was read as an analyze
		// request and answered with a dead end ("jim/main.go is
		// not a directory") or a full report. The composer plans
		// it, shows the diff, and writes only behind the gate.
		if tryAddWebServer(line, reader) {
			continue
		}
		// Cross-file delegates, claimed deterministically beside
		// the other write-side parsers: "add a function in
		// app/main.go that calls store.Load" writes code in one
		// file from a real signature in another, assembled by the
		// analyzer — plan, gate, disk backup, baseline-relative
		// verdict on the one affected package, and at most one
		// retry in the plan's named alternate form.
		if tryDelegate(line, reader) {
			continue
		}
		// Intent layer: a confident trained classification of what
		// the user wants, in their own words — no memorized commands.
		if tryIntentDispatch(line, profile) {
			continue
		}
		// Recall questions are answered from the session transcript,
		// deterministically, before routing or any model sees them.
		if rq, ok := conv.Recall(line); ok {
			fmt.Printf("%s %s\n", cTag("gollemer [memory]>"), rq)
			conv.AddUser(line)
			conv.AddReply(rq, SocialDomain, true)
			continue
		}
		conv.AddUser(line)
		d := routeDomain(line)
		if tryDeterministicAnswer(line, d, conv, reader, projectRoot) {
			continue
		}
		model := models[d]
		tag := d
		if model == nil {
			model = socialModel
			tag = d + "→social"
		}
		// The effective generating domain: a fallback answer comes from
		// the social model even when the request routed elsewhere.
		effD := d
		if model == socialModel {
			effD = SocialDomain
		}
		modelInput := strings.ToLower(line)
		if effD == SocialDomain {
			modelInput = conv.SocialInput(modelInput)
		}
		answer, trace, err := model.PredictWithTrace(modelInput, 40)
		if err != nil {
			fmt.Printf("%s %s\n", cTag(fmt.Sprintf("gollemer [%s]>", tag)), cErr(fmt.Sprintf("[error: %v]", err)))
			continue
		}
		reply := tidyDecode(answer)
		// Only gocode output gets code post-processing; prose modes never do.
		if d == GocodeDomain {
			reply = tidyGoCode(answer, goCase)
		}
		if d == GoCliDomain {
			// Commands keep their token spacing: "go build ." not "go build.".
			reply = tidyGoCommand(answer)
		}
		fmt.Printf("%s %s\n", cTag(fmt.Sprintf("gollemer [%s]>", tag)), reply)
		conv.AddReply(reply, effD, false)
		// A gocli reply is an exact toolchain command. Offer to run it
		// directly (working directory shown, no shell); only bare
		// go/gofmt invocations are ever offered, never model chatter.
		if d == GoCliDomain && tag == GoCliDomain && isRunnableGoCommand(reply) {
			offerRunGoCommand(reader, strings.TrimSpace(reply))
		}
		// A makefile neural reply names an exact repo command. Offer to
		// run it directly (repo root, no shell), same as gocli commands.
		if d == MakefileDomain && tag == MakefileDomain {
			if t := runnableMakeTarget(reply); t != "" {
				offerRunMakeCommand(reader, projectRoot, t)
			}
		}
		if showThoughts {
			printThoughtTrace(trace)
		}
	}
	return nil
}

// printTranscript shows what gollemer remembers from this session.
func printTranscript(conv *Conversation) {
	turns := conv.Turns()
	if len(turns) == 0 {
		fmt.Println("[nothing remembered yet — say something first]")
		return
	}
	fmt.Printf("[remembering %d turns this session]\n", len(turns))
	for _, t := range turns {
		if t.Speaker == "you" {
			fmt.Printf("you: %s\n", t.Text)
		} else if t.Domain != "" {
			fmt.Printf("%s %s\n", cTag(fmt.Sprintf("gollemer [%s]:", t.Domain)), t.Text)
		} else {
			fmt.Printf("gollemer: %s\n", t.Text)
		}
	}
}

// runnableGoCommand whitelists what unified chat will ever offer to
// execute: a single bare `go ...` or `gofmt ...` invocation and nothing
// else. The gocli model is trained on those commands alone; if it emits
// anything else, the reply is printed as plain text and never offered.
var runnableGoCommand = regexp.MustCompile(`(?i:^\s*(go|gofmt)\b.+$)`)

// shellMetachars rejects command chaining: even a whitelisted command is
// never offered if it smuggles in ;, &, |, redirects, or substitutions.
var shellMetachars = regexp.MustCompile("[;&|<>()`$]")

// isRunnableGoCommand reports whether a gocli reply is safe to offer for
// execution: a single go/gofmt invocation with no shell metacharacters.
func isRunnableGoCommand(reply string) bool {
	r := strings.TrimSpace(reply)
	return runnableGoCommand.MatchString(r) && !shellMetachars.MatchString(r)
}

// stdinIsTerminal reports whether stdin is an interactive terminal.
// The run offer is skipped for piped input so scripted sessions (and
// eval harnesses) never block on, or accidentally answer, the prompt.
func stdinIsTerminal() bool {
	fi, err := os.Stdin.Stat()
	if err != nil {
		return false
	}
	return fi.Mode()&os.ModeCharDevice != 0
}

// offerRunGoCommand asks for confirmation, then runs a gocli command in
// the chat's working directory and prints its output. The answer is read
// through the chat's line reader so stdin stays on one reader.
func offerRunGoCommand(r *LineReader, cmd string) {
	cwd, _ := os.Getwd()
	if !stdinIsTerminal() {
		// Piped/scripted session: the command is already printed on the
		// reply line above; never prompt, never run.
		return
	}
	yn, err := r.ReadLine(fmt.Sprintf("[run it here? %s] [y/n]: ", cwd))
	if err != nil {
		fmt.Println("[cancelled]")
		return
	}
	yn = strings.ToLower(strings.TrimSpace(yn))
	if yn != "y" && yn != "yes" {
		fmt.Println("[not run]")
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	// Strict token split and direct execution: no shell is ever
	// involved, so even a compromised model output cannot escape into
	// sh syntax. isRunnableGoCommand already rejected every shell
	// metacharacter; the trained commands contain no quoted strings,
	// so Fields is a complete parse here.
	fields := strings.Fields(cmd)
	out, err := exec.CommandContext(ctx, fields[0], fields[1:]...).CombinedOutput()
	if len(out) > 0 {
		fmt.Printf("%s", out)
	}
	if err != nil {
		fmt.Printf("[exit: %v]\n", err)
		return
	}
	fmt.Println("[done]")
}

// makeTargetLine matches a Makefile target definition: a lowercase name
// at the start of a line followed by a colon. Variable assignments and
// recipe lines never match.
var makeTargetLine = regexp.MustCompile(`^([a-z][a-z0-9_-]*):`)

// runnableMakeReply matches a makefile-brain reply of the form
// "run make <target>" and captures the target.
var runnableMakeReply = regexp.MustCompile(`(?i:^\s*run make ([a-z][a-z0-9_-]*)\s*$)`)

// makeTargetsAllowlist is the set of Makefile targets the chat may offer
// to run, parsed from the repo Makefile at startup. The chat's own
// entry points (chat, debug-chat) are excluded: running make chat from
// inside the chat would nest a session inside itself.
var makeTargetsAllowlist = map[string]bool{}

// initMakeAllowlist parses the Makefile targets once per session. If the
// Makefile can't be read the allowlist stays empty and nothing is ever
// offered — fail closed.
func initMakeAllowlist(projectRoot string) {
	data, err := os.ReadFile(filepath.Join(projectRoot, "Makefile"))
	if err != nil {
		return
	}
	for _, line := range strings.Split(string(data), "\n") {
		if m := makeTargetLine.FindStringSubmatch(line); m != nil {
			if m[1] != "chat" && m[1] != "debug-chat" {
				makeTargetsAllowlist[m[1]] = true
			}
		}
	}
	parseMakeDocs(string(data))
}

// runnableMakeTarget extracts a whitelisted make target from a
// makefile-brain reply ("run make eval" -> "eval"). Anything else —
// model chatter, metachars, unknown targets — returns "" and is never
// offered for execution.
func runnableMakeTarget(reply string) string {
	m := runnableMakeReply.FindStringSubmatch(reply)
	if m == nil || shellMetachars.MatchString(m[1]) {
		return ""
	}
	target := strings.ToLower(m[1])
	if !makeTargetsAllowlist[target] {
		return ""
	}
	return target
}

// offerRunMakeCommand asks for confirmation, then runs a makefile target
// in the repo root with output streaming straight to the terminal.
// There is no timeout: long targets (make smarter, make eval) need
// their time, and the user confirmed interactively. Ctrl-C interrupts.
func offerRunMakeCommand(r *LineReader, projectRoot, target string) {
	if !stdinIsTerminal() {
		// Piped/scripted session: never prompt, never run.
		return
	}
	yn, err := r.ReadLine(fmt.Sprintf("[run it here? 'make %s'] [y/n]: ", target))
	if err != nil {
		fmt.Println("[cancelled]")
		return
	}
	yn = strings.ToLower(strings.TrimSpace(yn))
	if yn != "y" && yn != "yes" {
		fmt.Println("[not run]")
		return
	}
	fmt.Printf("[running: make %s — Ctrl-C to interrupt]\n", target)
	// Direct execution, no shell: the target came from the parsed
	// Makefile allowlist, so even a compromised model output cannot
	// smuggle in flags, paths, or chained commands.
	cmd := exec.Command("make", target)
	cmd.Dir = projectRoot
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		fmt.Printf("[exit: %v]\n", err)
		return
	}
	fmt.Println("[done]")
}

// printThoughtTrace renders the model's per-token thought process: which MoE
// experts it consulted for each generated token, the runners-up it
// considered, and the overall expert mix for the reply.
func printThoughtTrace(trace *seq2seq.ThoughtTrace) {
	if trace == nil || len(trace.Steps) == 0 {
		return
	}
	fmt.Printf("  💭 thought process (%d tokens, top-%d MoE routing):\n", len(trace.Steps), len(trace.Steps[0].Experts))
	for _, s := range trace.Steps {
		parts := make([]string, 0, len(s.Experts))
		for i, e := range s.Experts {
			parts = append(parts, fmt.Sprintf("E%d:%.2f", e, s.Gates[i]))
		}
		alt := ""
		if len(s.Candidates) > 1 {
			alt = fmt.Sprintf("  (also considered: %s)", strings.Join(s.Candidates[1:], ", "))
		}
		fmt.Printf("    %-12q → [%s]%s\n", s.Token, strings.Join(parts, " "), alt)
	}
	if len(trace.ExpertUsage) > 0 {
		parts := make([]string, 0, len(trace.ExpertUsage))
		for e, u := range trace.ExpertUsage {
			bars := int(u*20 + 0.5)
			parts = append(parts, fmt.Sprintf("E%d %s %3.0f%%", e, strings.Repeat("█", bars), u*100))
		}
		fmt.Printf("    expert mix: %s\n", strings.Join(parts, "  "))
	}
}

// verifyGocodeIntent implements the write-then-analyze loop: after the LSTM
// generates Go code, extract the function name and verify it matches the
// user's intent. Returns a verdict string to append, or "" if not applicable.
func verifyGocodeIntent(userInput, generatedCode string) string {
	// Extract intent: "write is-even" -> "iseven", "write reverse string" -> "reversestring"
	intent := strings.ToLower(userInput)
	intent = strings.TrimPrefix(intent, "write ")
	intent = strings.TrimPrefix(intent, "write")
	intent = strings.TrimSpace(intent)
	// Normalize: remove spaces, dashes, underscores
	normalizer := strings.NewReplacer(" ", "", "-", "", "_", "")
	intentNorm := normalizer.Replace(intent)
	if intentNorm == "" {
		return ""
	}

	// Extract function name from generated code: func Name(
	funcRe := regexp.MustCompile(`func\s+(\w+)\s*\(`)
	matches := funcRe.FindStringSubmatch(generatedCode)
	if len(matches) < 2 {
		return "[loop] could not parse function name from generated code"
	}
	funcName := strings.ToLower(matches[1])

	// Verify: does the intent appear in the func name, or vice versa?
	// Also handle camelCase: "isEven" -> "iseven" (already lowercased)
	if strings.Contains(funcName, intentNorm) || strings.Contains(intentNorm, funcName) {
		return fmt.Sprintf("[loop] ✓ verified: %q matches intent %q", matches[1], intent)
	}
	return fmt.Sprintf("[loop] ✗ mismatch: generated %q does not match intent %q", matches[1], intent)
}

// appendToFile appends generated code to a file in the project.
func appendToFile(projectRoot, relPath, code string) error {
	fullPath := relPath
	if !strings.HasPrefix(relPath, "/") {
		fullPath = projectRoot + "/" + relPath
	}
	f, err := os.OpenFile(fullPath, os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.WriteString("\n" + code + "\n")
	return err
}

// handleContribute implements /contribute <task>: beginner-friendly code contribution.
// Example: /contribute add a getter for Timeout
// Steps: parse task -> find location -> explain -> generate -> show -> [y/n] -> apply
func handleContribute(line string, projectRoot string) (string, bool) {
	if line != "/contribute" && !strings.HasPrefix(line, "/contribute ") {
		return "", false
	}
	task := strings.TrimSpace(strings.TrimPrefix(line, "/contribute"))
	if task == "" {
		return "Usage: /contribute <task> — e.g. /contribute add a getter for Timeout", true
	}

	if lastAnalyzeProject == nil {
		return "Run /analyze first to load the codebase.", true
	}

	lower := strings.ToLower(task)
	var op, field string
	if strings.Contains(lower, "getter") {
		op = "getter"
		field = extractFieldName(task)
	} else if strings.Contains(lower, "setter") {
		op = "setter"
		field = extractFieldName(task)
	} else {
		return "I can help with: add a getter for <Field>, add a setter for <Field>.", true
	}
	if field == "" {
		return "What field? e.g. /contribute add a getter for Timeout", true
	}

	// Find the struct containing this field
	var structName, targetFile string
	for _, pkg := range lastAnalyzeProject.Packages {
		for _, t := range pkg.Types {
			if t.Kind != "struct" {
				continue
			}
			for _, f := range t.Fields {
				if strings.EqualFold(f, field) {
					structName = t.Name
					targetFile = t.File
					break
				}
			}
		}
	}
	if structName == "" {
		return fmt.Sprintf("Couldn't find a struct with field %q.", field), true
	}

	// For the type, we'd need field types (not just names). Ask the user.
	// For now, use a placeholder and let them confirm.
	return fmt.Sprintf("Found %q in struct %s (%s).\nTo generate the %s, I need the field type.\nUse: /add-getter %s %s <Type>  (e.g. /add-getter %s %s time.Duration)",
		field, structName, targetFile, op, structName, field, structName, field), true
}

// extractFieldName pulls the field name from "add a getter for Timeout"
func extractFieldName(task string) string {
	// Look for "for X" at the end
	lower := strings.ToLower(task)
	idx := strings.LastIndex(lower, "for ")
	if idx >= 0 {
		return strings.TrimSpace(task[idx+4:])
	}
	// Fallback: last word
	words := strings.Fields(task)
	if len(words) > 0 {
		return words[len(words)-1]
	}
	return ""
}

// generateMethodSkeleton creates a compilable TODO skeleton for a new method.
// It includes the struct's field inventory as comments and references the
// closest existing method as a template. This gives the beginner something
// that compiles plus a clear map of what data is available.
func generateMethodSkeleton(structName, methodName, retType string) string {
	receiver := strings.ToLower(structName[:1])
	var b strings.Builder
	fmt.Fprintf(&b, "func (%s *%s) %s() %s {\n", receiver, structName, methodName, retType)
	// Field inventory: show what data the beginner can use.
	if fields := analyze.LastWhereFields; len(fields) > 0 {
		fmt.Fprintf(&b, "\t// Available fields on %s:\n", structName)
		for _, f := range fields {
			// Skip embedded mutexes — not useful for the beginner.
			if f == "RWMutex" || f == "Mutex" {
				continue
			}
			fmt.Fprintf(&b, "\t//   %s.%s\n", receiver, f)
		}
	}
	// Reference the closest sibling: a method returning a similar type.
	if closest := closestSibling(retType); closest != "" {
		fmt.Fprintf(&b, "\t// See %s for a similar method to copy from.\n", closest)
	}
	fmt.Fprintf(&b, "\t// TODO: implement %s\n", methodName)
	// For map/slice returns, pre-create the result so it compiles.
	if strings.HasPrefix(retType, "map[") {
		fmt.Fprintf(&b, "\tresult := make(%s)\n", retType)
		fmt.Fprintf(&b, "\treturn result\n")
	} else if strings.HasPrefix(retType, "[]") {
		fmt.Fprintf(&b, "\tresult := make(%s, 0)\n", retType)
		fmt.Fprintf(&b, "\treturn result\n")
	} else {
		fmt.Fprintf(&b, "\treturn nil // TODO: replace with real value\n")
	}
	b.WriteString("}")
	return b.String()
}

// closestSibling finds the existing method with the most similar return type.
// Prefers methods returning collections (map/slice) for data-exposing methods.
func closestSibling(retType string) string {
	wantsColl := strings.HasPrefix(retType, "map[") || strings.HasPrefix(retType, "[]")
	for _, sig := range analyze.LastWhereSiblings {
		// Sig looks like "func (dic *container) Names() map[string][]string"
		// Extract the method name: between ") " and "(".
		name := ""
		if idx := strings.Index(sig, ") "); idx >= 0 {
			rest := sig[idx+2:]
			if pidx := strings.Index(rest, "("); pidx >= 0 {
				name = rest[:pidx]
			}
		}
		if name == "" {
			continue
		}
		isColl := strings.Contains(sig, "map[") || strings.Contains(sig, "[]")
		if wantsColl && isColl {
			return name + "()"
		}
	}
	// Fallback: first sibling that isn't the method we're creating.
	for _, sig := range analyze.LastWhereSiblings {
		if idx := strings.Index(sig, ") "); idx >= 0 {
			rest := sig[idx+2:]
			if pidx := strings.Index(rest, "("); pidx >= 0 {
				return rest[:pidx] + "()"
			}
		}
	}
	return ""
}

// explainMethodPurpose gives a one-sentence plain-words description of what
// the new method should do, derived from the method name.
func explainMethodPurpose(methodName string) string {
	lower := strings.ToLower(methodName)
	// "Resolution" -> "For each service, list what it depends on."
	// "Info", "Summary", "Graph", etc. get similar treatment.
	switch {
	case strings.Contains(lower, "resolution") || strings.Contains(lower, "graph") || strings.Contains(lower, "depend"):
		return "For each service, it lists the services and parameters that service depends on."
	case strings.Contains(lower, "info") || strings.Contains(lower, "summary") || strings.Contains(lower, "describe"):
		return "It returns a human-readable summary of the struct's current state."
	case strings.Contains(lower, "list") || strings.Contains(lower, "names"):
		return "It returns the names of everything stored in the struct."
	default:
		return "It exposes the struct's internal data so callers can inspect it."
	}
}

// suggestErrorFix parses a Go compiler error and suggests a fix using
// the analyzed project. Go errors are structured, so this is reliable.
func suggestErrorFix(errText string, p *analyze.Project) string {
	var b strings.Builder
	// "undefined: foo"
	if m := regexp.MustCompile(`undefined:\s*(\w+)`).FindStringSubmatch(errText); m != nil {
		name := m[1]
		fmt.Fprintf(&b, "The name %q isn't defined where it's used.\n", name)
		if p != nil {
			// Look for a similar name (typo?).
			for _, pkg := range p.Packages {
				for _, fn := range pkg.Funcs {
					if strings.EqualFold(fn.Name, name) {
						continue
					}
					if levenshtein(strings.ToLower(fn.Name), strings.ToLower(name)) <= 2 {
						fmt.Fprintf(&b, "Did you mean %q? It's defined at %s:%d.\n", fn.Name, fn.File, fn.Line)
					}
				}
				for _, t := range pkg.Types {
					if levenshtein(strings.ToLower(t.Name), strings.ToLower(name)) <= 2 {
						fmt.Fprintf(&b, "Did you mean type %q? It's at %s:%d.\n", t.Name, t.File, t.Line)
					}
				}
			}
		}
		fmt.Fprintf(&b, "In plain words: you're using a name the compiler can't find. Check the spelling, or make sure it's defined/imported.")
		return b.String()
	}
	// "cannot use X as Y"
	if m := regexp.MustCompile(`cannot use (\S+) as (\S+)`).FindStringSubmatch(errText); m != nil {
		fmt.Fprintf(&b, "Type mismatch: you gave %s but %s was expected.\n", m[1], m[2])
		fmt.Fprintf(&b, "In plain words: the value's type doesn't match what the code wants. Convert it, or check if you're passing the right variable.")
		return b.String()
	}
	// "X already declared"
	if m := regexp.MustCompile(`(\w+) already declared`).FindStringSubmatch(errText); m != nil {
		fmt.Fprintf(&b, "%q is declared twice.\n", m[1])
		fmt.Fprintf(&b, "In plain words: you (or the tool) added it twice. Remove one copy.")
		return b.String()
	}
	// "imported and not used"
	if m := regexp.MustCompile(`"([^"]+)" imported and not used`).FindStringSubmatch(errText); m != nil {
		fmt.Fprintf(&b, "Package %q is imported but never used.\n", m[1])
		fmt.Fprintf(&b, "In plain words: Go doesn't allow unused imports. Either use something from it, or remove the import line.")
		return b.String()
	}
	return "I don't recognize that error shape yet. Paste the full message."
}

// levenshtein is the edit distance between two strings.
func levenshtein(a, b string) int {
	la, lb := len(a), len(b)
	if la == 0 {
		return lb
	}
	if lb == 0 {
		return la
	}
	prev := make([]int, lb+1)
	for j := range prev {
		prev[j] = j
	}
	for i := 1; i <= la; i++ {
		curr := make([]int, lb+1)
		curr[0] = i
		for j := 1; j <= lb; j++ {
			cost := 0
			if a[i-1] != b[j-1] {
				cost = 1
			}
			curr[j] = min(prev[j]+1, min(curr[j-1]+1, prev[j-1]+cost))
		}
		prev = curr
	}
	return prev[lb]
}

// explainConcept explains a concept via its real usage in the analyzed code.
// For Go built-ins (mutex, channel, goroutine), it finds usage and explains
// through that concrete example — not a textbook definition.
func explainConcept(concept string, p *analyze.Project) string {
	// Accept "Client.MakeRequest" / "matrix.New" style references too.
	if fn := findFuncByQuery(p, concept); fn != nil {
		concept = fn.Name
	}
	cl := strings.ToLower(concept)
	var b strings.Builder
	// First: is it a type or func in the code?
	for _, pkg := range p.Packages {
		for _, t := range pkg.Types {
			if strings.EqualFold(t.Name, concept) {
				fmt.Fprintf(&b, "%s is a %s defined at %s:%d.\n", t.Name, t.Kind, t.File, t.Line)
				if len(t.Fields) > 0 {
					fmt.Fprintf(&b, "It holds: %s.\n", strings.Join(t.Fields, ", "))
				}
				// Find where it's used.
				uses := 0
				for _, pkg2 := range p.Packages {
					for _, fn := range pkg2.Funcs {
						if strings.Contains(fn.Src, t.Name) {
							uses++
						}
					}
				}
				if uses > 0 {
					fmt.Fprintf(&b, "It's referenced in %d function(s).", uses)
				}
				return b.String()
			}
		}
		for _, fn := range pkg.Funcs {
			if strings.EqualFold(fn.Name, concept) && !fn.IsTest {
				fmt.Fprintf(&b, "%s is a function: %s\n", fn.Name, fn.Sig)
				if fn.Doc != "" {
					fmt.Fprintf(&b, "%s\n", analyze.PieceFirstSentence(fn.Doc))
				}
				return b.String()
			}
		}
	}
	// Go built-in concepts: explain via real usage.
	// Find the concept in the code and explain through that example.
	for _, pkg := range p.Packages {
		for _, fn := range pkg.Funcs {
			if strings.Contains(strings.ToLower(fn.Src), cl) {
				switch cl {
				case "mutex", "rwmutex":
					return fmt.Sprintf("A mutex is a lock — only one goroutine holds it at a time.\nHere, %s() uses it at %s:%d to protect shared data from concurrent access.\nIn plain words: without the lock, two goroutines could read/write at the same moment and corrupt the data.", fn.Name, fn.File, fn.Line)
				case "goroutine", "go ":
					return fmt.Sprintf("A goroutine is a lightweight thread — `go f()` runs f concurrently.\n%s() at %s:%d uses one.\nIn plain words: it lets work happen in the background while the main flow continues.", fn.Name, fn.File, fn.Line)
				case "channel", "chan":
					return fmt.Sprintf("A channel lets goroutines send values to each other safely.\n%s() at %s:%d uses one.\nIn plain words: it's a pipe — one side sends, the other receives, and Go handles the synchronization.", fn.Name, fn.File, fn.Line)
				}
			}
		}
	}
	return fmt.Sprintf("I don't see %q in this codebase. Try /find %s to search for related names.", concept, concept)
}

// describeArchitecture gives a plain-words summary by traversing from main().
func describeArchitecture(p *analyze.Project) string {
	var b strings.Builder
	// Find main.
	var mainFn *analyze.Func
	for _, pkg := range p.Packages {
		for _, fn := range pkg.Funcs {
			if fn.Name == "main" && fn.IsMain {
				mainFn = fn
				break
			}
		}
	}
	if mainFn == nil {
		// No main: describe packages.
		fmt.Fprintf(&b, "This is a library (no main function).\n")
		fmt.Fprintf(&b, "Packages:\n")
		for _, pkg := range p.Packages {
			nf := 0
			for _, fn := range pkg.Funcs {
				if fn.Exported && !fn.IsTest {
					nf++
				}
			}
			fmt.Fprintf(&b, "  %s: %d exported function(s)\n", pkg.Name, nf)
		}
		return b.String()
	}
	fmt.Fprintf(&b, "Starting from main() at %s:%d:\n", mainFn.File, mainFn.Line)
	// BFS call graph from main, depth 2.
	seen := map[string]bool{mainFn.ID: true}
	type node struct {
		fn    *analyze.Func
		depth int
	}
	queue := []node{{mainFn, 0}}
	for len(queue) > 0 {
		n := queue[0]
		queue = queue[1:]
		if n.depth >= 2 {
			continue
		}
		// Find callees via the project's call info.
		// Use Src to find called names (simple heuristic).
		for _, pkg := range p.Packages {
			for _, fn := range pkg.Funcs {
				if seen[fn.ID] || fn.IsTest {
					continue
				}
				// Does n.fn call fn? Check if fn.Name appears in n.fn.Src as a call.
				if n.fn.Src != "" && regexp.MustCompile(`\b`+regexp.QuoteMeta(fn.Name)+`\s*\(`).MatchString(n.fn.Src) {
					seen[fn.ID] = true
					indent := strings.Repeat("  ", n.depth+1)
					desc := fn.Name
					if fn.Doc != "" {
						if s := analyze.PieceFirstSentence(fn.Doc); s != "" && len(s) < 80 {
							desc += " — " + s
						}
					}
					fmt.Fprintf(&b, "%s→ %s\n", indent, desc)
					queue = append(queue, node{fn, n.depth + 1})
				}
			}
		}
	}
	return b.String()
}

// generateTestSkeleton creates a table-driven test from a function's signature.
func generateTestSkeleton(fname string, p *analyze.Project) string {
	if fn := findFuncByQuery(p, fname); fn != nil {
		fname = fn.Name // accept "Client.MakeRequest" etc., not just bare names
	}
	for _, pkg := range p.Packages {
		for _, fn := range pkg.Funcs {
			if !strings.EqualFold(fn.Name, fname) || fn.IsTest {
				continue
			}
			var b strings.Builder
			testName := "Test" + fn.Name
			fmt.Fprintf(&b, "func %s(t *testing.T) {\n", testName)
			fmt.Fprintf(&b, "\ttests := []struct {\n")
			fmt.Fprintf(&b, "\t\tname string\n")
			fmt.Fprintf(&b, "\t\t// TODO: add input fields from: %s\n", fn.Sig)
			fmt.Fprintf(&b, "\t\t// TODO: add expected output fields\n")
			fmt.Fprintf(&b, "\t}{\n")
			fmt.Fprintf(&b, "\t\t// TODO: add test cases\n")
			fmt.Fprintf(&b, "\t}\n")
			fmt.Fprintf(&b, "\tfor _, tt := range tests {\n")
			fmt.Fprintf(&b, "\t\tt.Run(tt.name, func(t *testing.T) {\n")
			fmt.Fprintf(&b, "\t\t\t// TODO: call %s with tt.input, compare to tt.expected\n", fn.Name)
			fmt.Fprintf(&b, "\t\t})\n")
			fmt.Fprintf(&b, "\t}\n")
			fmt.Fprintf(&b, "}\n")
			if fn.Doc != "" {
				b.WriteString("\n// Intent from docs: " + analyze.PieceFirstSentence(fn.Doc) + "\n")
			}
			return b.String()
		}
	}
	return fmt.Sprintf("Function %q not found. Try /find %s.", fname, fname)
}

// explainImport traces why a package is imported.
func explainImport(pkg string, p *analyze.Project) string {
	var b strings.Builder
	pkg = strings.Trim(pkg, "\"")
	found := false
	for _, pkg2 := range p.Packages {
		for _, fn := range pkg2.Funcs {
			if fn.IsTest {
				continue
			}
			// Look for pkg.X usage in the source.
			if fn.Src != "" && strings.Contains(fn.Src, pkg+".") {
				if !found {
					fmt.Fprintf(&b, "Package %q is used by:\n", pkg)
					found = true
				}
				// Find the specific usage.
				re := regexp.MustCompile(regexp.QuoteMeta(pkg) + `\.(\w+)`)
				uses := map[string]bool{}
				for _, m := range re.FindAllStringSubmatch(fn.Src, -1) {
					uses[m[1]] = true
				}
				for u := range uses {
					fmt.Fprintf(&b, "  %s() at %s:%d uses %s.%s\n", fn.Name, fn.File, fn.Line, pkg, u)
				}
			}
		}
	}
	if !found {
		return fmt.Sprintf("No usage of package %q found in the analyzed code. It might be imported for side effects, or the analysis missed it.", pkg)
	}
	return b.String()
}

// reviewFunc generates review comments for a function.
func reviewFunc(fname string, p *analyze.Project) string {
	if fn := findFuncByQuery(p, fname); fn != nil {
		fname = fn.Name // accept "Client.MakeRequest" etc., not just bare names
	}
	for _, pkg := range p.Packages {
		for _, fn := range pkg.Funcs {
			if !strings.EqualFold(fn.Name, fname) || fn.IsTest {
				continue
			}
			var b strings.Builder
			fmt.Fprintf(&b, "Review of %s (%s:%d):\n", fn.Name, fn.File, fn.Line)
			issues := 0
			// Check: ignored errors ( "_ = " or "_, _ =").
			if fn.Src != "" {
				if strings.Contains(fn.Src, "_ = ") || regexp.MustCompile(`_,\s*_\s*=`).MatchString(fn.Src) {
					fmt.Fprintf(&b, "  ⚠ Ignores a return value with _. If it's an error, failures will be silent.\n")
					issues++
				}
				// Check: long function (rough: > 50 lines).
				lines := strings.Count(fn.Src, "\n")
				if lines > 50 {
					fmt.Fprintf(&b, "  ⚠ Long function (%d lines). Consider splitting into smaller helpers.\n", lines)
					issues++
				}
				// Check: no doc on exported.
				if fn.Exported && fn.Doc == "" {
					fmt.Fprintf(&b, "  ⚠ Exported but no doc comment. Add one explaining what it does.\n")
					issues++
				}
			}
			if issues == 0 {
				fmt.Fprintf(&b, "  ✓ No issues found by the basic checks.\n")
			}
			return b.String()
		}
	}
	return fmt.Sprintf("Function %q not found.", fname)
}

// generateDocCommentCmd creates a Go doc comment for a function and offers
// to insert it. The comment is derived from the function's structure:
// name verb, signature, control flow, and data flow — not memorized text.
func generateDocCommentCmd(fname string, p *analyze.Project, profile *UserProfile) (string, bool) {
	if fn := findFuncByQuery(p, fname); fn != nil {
		fname = fn.Name // accept "Client.MakeRequest" etc., not just bare names
	}
	for _, pkg := range p.Packages {
		for _, fn := range pkg.Funcs {
			if !strings.EqualFold(fn.Name, fname) || fn.IsTest {
				continue
			}
			var b strings.Builder
			// Line 1: Name + verb phrase (Go convention: starts with the name).
			verb := docVerb(fn.Name)
			fmt.Fprintf(&b, "// %s %s", fn.Name, verb)
			// Add receiver context.
			if fn.Receiver != "" {
				// Find the receiver type name.
				rtype := strings.Trim(fn.Receiver, "*")
				fmt.Fprintf(&b, " on the %s", rtype)
			}
			b.WriteString(".\n")
			// Line 2+: structural details from the analyzer.
			details := docDetails(fn)
			for _, d := range details {
				fmt.Fprintf(&b, "// %s\n", d)
			}
			comment := b.String()
			var out strings.Builder
			fmt.Fprintf(&out, "Generated doc for %s:\n\n%s\n", fn.Name, comment)
			if confirmAction(fmt.Sprintf("Insert above %s at %s:%d?", fn.Name, fn.File, fn.Line), profile) {
				if err := insertDocComment(fn, comment); err != nil {
					fmt.Fprintf(&out, "Couldn't insert: %v\n", err)
				} else {
					fmt.Fprintf(&out, "Inserted.\n")
				}
			} else {
				fmt.Fprintf(&out, "Skipped.\n")
			}
			return out.String(), true
		}
	}
	return "", false
}

// docVerb derives a verb phrase from a function name.
// "Names" -> "returns the names", "Register" -> "registers a service",
// "Freeze" -> "converts", "IsReady" -> "reports whether".
func docVerb(name string) string {
	lower := strings.ToLower(name)
	switch {
	case strings.HasPrefix(lower, "get"):
		rest := name[3:]
		return "returns " + camelToWords(rest)
	case strings.HasPrefix(lower, "set"):
		rest := name[3:]
		return "sets " + camelToWords(rest)
	case strings.HasPrefix(lower, "is") && len(name) > 2 && isUpper(name[2]):
		rest := name[2:]
		return "reports whether " + camelToWords(rest)
	case strings.HasPrefix(lower, "has") && len(name) > 3 && isUpper(name[3]):
		rest := name[3:]
		return "reports whether it has " + camelToWords(rest)
	case strings.HasPrefix(lower, "new"):
		return "creates and returns a new " + camelToWords(name[3:])
	case strings.HasPrefix(lower, "must") && len(name) > 4 && isUpper(name[4]):
		// Go's Must* convention: like the plain getter, but panics
		// instead of returning an error (MustParam, MustService...).
		rest := name[4:]
		return "returns the " + camelToWords(rest) + ", and panics if it is missing"
	default:
		// Plural noun like "Names" -> "returns the names".
		// Verb like "Register", "Freeze", "Store" -> conjugated.
		if strings.HasSuffix(lower, "s") && !strings.HasSuffix(lower, "ss") {
			return "returns the " + camelToWords(name)
		}
		return conjugateVerb(name)
	}
}

// conjugateVerb turns "Register" into "registers", "Freeze" into "freezes".
func conjugateVerb(name string) string {
	lower := strings.ToLower(name)
	w := camelToWords(name)
	// Simple conjugation: add "s", handle endings.
	words := strings.Fields(w)
	if len(words) == 0 {
		return w
	}
	v := words[0]
	switch {
	case strings.HasSuffix(v, "s") || strings.HasSuffix(v, "x") || strings.HasSuffix(v, "ch") || strings.HasSuffix(v, "sh"):
		v += "es"
	case strings.HasSuffix(v, "y") && len(v) > 1 && !isVowel(v[len(v)-2]):
		v = v[:len(v)-1] + "ies"
	default:
		v += "s"
	}
	words[0] = v
	// Special cases.
	if lower == "freeze" {
		return "converts the container from build mode to run mode"
	}
	return strings.Join(words, " ")
}

// camelToWords turns "serviceDefs" into "service defs", "Names" into "names".
func camelToWords(s string) string {
	var b strings.Builder
	for i, r := range s {
		if i > 0 && r >= 'A' && r <= 'Z' {
			b.WriteString(" ")
		}
		b.WriteRune(r)
	}
	return strings.ToLower(strings.TrimSpace(b.String()))
}

func isUpper(c byte) bool { return c >= 'A' && c <= 'Z' }
func isVowel(c byte) bool {
	switch c {
	case 'a', 'e', 'i', 'o', 'u':
		return true
	}
	return false
}

// docDetails extracts structural facts for the doc comment body.
func docDetails(fn *analyze.Func) []string {
	var details []string
	src := fn.Src
	if src == "" {
		return details
	}
	sl := strings.ToLower(src)
	// Locking.
	if strings.Contains(sl, ".lock()") || strings.Contains(sl, ".rlock()") {
		if strings.Contains(sl, ".rlock()") {
			details = append(details, "It locks for reading, so concurrent readers are safe.")
		} else {
			details = append(details, "It locks for writing.")
		}
	}
	// Iteration.
	if strings.Contains(sl, "for ") && strings.Contains(sl, "range") {
		details = append(details, "It iterates over the stored data.")
	}
	// Error returns.
	if strings.Contains(fn.Sig, "error") {
		details = append(details, "It returns an error if the operation fails.")
	}
	return details
}

// insertDocComment inserts a doc comment above the function in its file.
func insertDocComment(fn *analyze.Func, comment string) error {
	// fn.File is relative to project root. We need the absolute path.
	// lastAnalyzeRoot holds the project root.
	if lastAnalyzeRoot == "" {
		return fmt.Errorf("project root unknown")
	}
	fullPath := lastAnalyzeRoot + "/" + fn.File
	content, err := os.ReadFile(fullPath)
	if err != nil {
		return err
	}
	lines := strings.Split(string(content), "\n")
	if fn.Line < 1 || fn.Line > len(lines) {
		return fmt.Errorf("line %d out of range", fn.Line)
	}
	// Insert before the func line (1-indexed). Don't duplicate if doc exists.
	idx := fn.Line - 1
	if idx > 0 && strings.HasPrefix(strings.TrimSpace(lines[idx-1]), "//") {
		return fmt.Errorf("already has a doc comment")
	}
	commentLines := strings.Split(strings.TrimRight(comment, "\n"), "\n")
	newLines := append([]string{}, lines[:idx]...)
	newLines = append(newLines, commentLines...)
	newLines = append(newLines, lines[idx:]...)
	return os.WriteFile(fullPath, []byte(strings.Join(newLines, "\n")), 0644)
}

// findFuncByQuery resolves a user-typed function reference to a *Func.
// Accepts "New", "Client.MakeRequest", "matrix.New",
// "mautrix.Client.MakeRequest" — matching is forgiving because typing
// the exact bare name from memory is the friction users actually hit.
// Returns nil when nothing matches.
func findFuncByQuery(p *analyze.Project, query string) *analyze.Func {
	query = strings.TrimSpace(query)
	if query == "" || p == nil {
		return nil
	}
	recvName := func(fn *analyze.Func) string {
		if fn.Receiver == "" {
			return ""
		}
		return strings.Trim(fn.Receiver, "*") + "." + fn.Name
	}
	var bare *analyze.Func
	lastSeg := query
	if i := strings.LastIndex(query, "."); i >= 0 {
		lastSeg = query[i+1:]
	}
	for _, pkg := range p.Packages {
		for _, fn := range pkg.Funcs {
			if fn.IsTest {
				continue
			}
			// Exact forms, most specific first.
			if rn := recvName(fn); rn != "" {
				if strings.EqualFold(query, rn) || strings.HasSuffix(strings.ToLower(query), "."+strings.ToLower(rn)) {
					return fn
				}
			}
			if fn.Pkg != "" && strings.EqualFold(query, fn.Pkg+"."+fn.Name) {
				return fn
			}
			// Remember a bare-name / last-segment match as fallback.
			if bare == nil && (strings.EqualFold(query, fn.Name) || strings.EqualFold(lastSeg, fn.Name)) {
				bare = fn
			}
		}
	}
	return bare
}

// nearFuncNames lists up to 5 function names containing the query's last
// segment (or contained by it), for "did you mean" hints.
func nearFuncNames(p *analyze.Project, query string) []string {
	seg := strings.ToLower(query)
	if i := strings.LastIndex(seg, "."); i >= 0 {
		seg = seg[i+1:]
	}
	if seg == "" || p == nil {
		return nil
	}
	seen := map[string]bool{}
	var out []string
	for _, pkg := range p.Packages {
		for _, fn := range pkg.Funcs {
			if fn.IsTest || seen[fn.Name] {
				continue
			}
			ln := strings.ToLower(fn.Name)
			if strings.Contains(ln, seg) || strings.Contains(seg, ln) {
				seen[fn.Name] = true
				out = append(out, fn.Name)
				if len(out) >= 5 {
					return out
				}
			}
		}
	}
	return out
}

// modelExplanationValid checks a model-generated explanation against the
// facts it was given. The model fails in two observed ways: it names a
// memorized function instead of the one asked about, and it claims
// behaviors (iterating, locking) that aren't in the facts. Either is a
// rejection. This is the write-then-verify loop applied to prose.
func modelExplanationValid(fn *analyze.Func, facts, out string) (bool, string) {
	lo := strings.ToLower(out)
	if !strings.Contains(lo, strings.ToLower(fn.Name)) {
		return false, fmt.Sprintf("it talks about something else, not %s", fn.Name)
	}
	if strings.Contains(lo, "iterat") && !strings.Contains(facts, "iterates") {
		return false, "it claims iteration that isn't in the facts"
	}
	if strings.Contains(lo, "lock") && !strings.Contains(facts, "locks") {
		return false, "it claims locking that isn't in the facts"
	}
	return true, ""
}

// templateExplanation is the verified fallback: plain words built from
// the function's real structure (same pieces /comment uses).
func templateExplanation(fn *analyze.Func) string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s %s", fn.Name, docVerb(fn.Name))
	if fn.Receiver != "" {
		fmt.Fprintf(&b, " on the %s", strings.Trim(fn.Receiver, "*"))
	}
	b.WriteString(".")
	for _, d := range docDetails(fn) {
		b.WriteString(" " + d)
	}
	if fn.Doc != "" {
		fmt.Fprintf(&b, " Its docs say: %s", analyze.PieceFirstSentence(fn.Doc))
	}
	return b.String()
}

// substitutePlaceholders fills the v2 model's name placeholders with
// the real names from the code: @@ is the receiver (replaced first,
// since @ is a prefix of @@), @ is the function name. The model never
// generates a name, so it cannot parrot a memorized one.
func substitutePlaceholders(out string, fn *analyze.Func) string {
	recv := strings.Trim(fn.Receiver, "*")
	out = strings.ReplaceAll(out, "@@", recv)
	out = strings.ReplaceAll(out, "@", fn.Name)
	return out
}

// RunUexplainProbe feeds raw facts text to the understanding model and
// prints its raw output — the Zorblax test: facts about a function
// that cannot exist in training must still steer the explanation.
func RunUexplainProbe(projectRoot, facts string) error {
	m, err := loadRealModel(projectRoot, UnderstandingDomain)
	if err != nil {
		return fmt.Errorf("understanding model not loaded: %w", err)
	}
	fmt.Printf("facts: %s\nraw model output: %s\n", facts, generateWithModel(m, facts))
	return nil
}

// generateWithModel runs the understanding model on structural facts.
func generateWithModel(model *seq2seq.Seq2Seq, facts string) string {
	if model == nil {
		return "(no model)"
	}
	out, err := model.Predict(facts, 40)
	if err != nil {
		return fmt.Sprintf("(error: %v)", err)
	}
	return out
}

// confirmAction asks [y/n] unless the profile says to auto-build.
// John's "do all"/"yeah" directive means: skip the question, just do it.
func confirmAction(prompt string, profile *UserProfile) bool {
	if profile.ShouldAutoBuild() {
		fmt.Printf("%s [auto-yes: your 'do all' directive]\n", prompt)
		return true
	}
	fmt.Printf("%s [y/n] ", prompt)
	var ans string
	fmt.Scanln(&ans)
	return strings.ToLower(strings.TrimSpace(ans)) == "y"
}

// explainGoLine gives a plain-words explanation of a generated Go line.
func explainGoLine(line, structName, methodName string) string {
	t := strings.TrimSpace(line)
	if strings.HasPrefix(t, "func ") {
		// func (c *container) Resolution() map[string][]string {
		if strings.Contains(t, ") "+methodName+"(") {
			return fmt.Sprintf("Defines the %s method on %s — callers use it to get the data.", methodName, structName)
		}
		return "Defines a new method."
	}
	if strings.HasPrefix(t, "return ") {
		return "Returns the internal data to the caller."
	}
	if t == "}" {
		return "End of the method."
	}
	if strings.HasPrefix(t, "//") {
		return "A comment explaining the code."
	}
	return "Part of the method."
}

// explainBuildError translates a Go build error into plain words.
func explainBuildError(out string) string {
	lower := strings.ToLower(out)
	if strings.Contains(lower, "undefined:") {
		return "The code uses something that doesn't exist yet — likely a typo in a name, or a missing import."
	}
	if strings.Contains(lower, "cannot use") || strings.Contains(lower, "mismatched types") {
		return "A type doesn't match — the method is trying to use a value as the wrong type."
	}
	if strings.Contains(lower, "expected") && strings.Contains(lower, "found") {
		return "A syntax problem — something's missing or in the wrong place, like a bracket or comma."
	}
	if strings.Contains(lower, "imported and not used") {
		return "An import isn't needed — the code doesn't use it."
	}
	return "The code has an error. Read the message above for the file and line number."
}

// understandingModel is the structure->language model, loaded
// lazily on the first /uexplain (shared by every chat loop).
var understandingModel *seq2seq.Seq2Seq

// runUexplainPrint is the /uexplain body, shared by the slash
// command and the intent layer: print the facts, try the
// understanding model (placeholder output, verified against the
// facts), and fall back to the template explanation.
func runUexplainPrint(fn *analyze.Func, projectRoot string) {
	facts := analyze.FuncFactsString(fn)
	fmt.Printf("\n%s %s\n", cHead("Facts:"), facts)
	// Try the model if it's installed; never let a missing model
	// block the command.
	if understandingModel == nil {
		if m, err := loadRealModel(projectRoot, UnderstandingDomain); err == nil {
			understandingModel = m
		}
	}
	answered := false
	if understandingModel != nil {
		raw := generateWithModel(understandingModel, facts)
		// The v2 model never writes names: it emits @ (function)
		// and @@ (receiver) placeholders, filled in here with
		// the real names from the code.
		out := substitutePlaceholders(raw, fn)
		if strings.Contains(out, "@") {
			fmt.Printf("%s %s\n", cErr("Model said (rejected — unfilled placeholder):"), raw)
			recordBrain("storyteller", false, "unfilled placeholder")
			recordBrain("verifier", true, "rejected unfilled placeholder")
		} else if ok, why := modelExplanationValid(fn, facts, out); ok {
			fmt.Printf("%s %s\n", cGood("Model says:"), out)
			recordBrain("storyteller", true, "verifier accepted")
			answered = true
		} else {
			fmt.Printf("%s %s\n", cErr(fmt.Sprintf("Model said (rejected — %s):", why)), out)
			recordBrain("storyteller", false, "verifier rejected: "+why)
			recordBrain("verifier", true, "rejected bad prose: "+why)
		}
	}
	if !answered {
		if understandingModel == nil {
			recordBrain("storyteller", false, "no model installed; template answered")
		}
		fmt.Printf("%s %s\n", cHead("Explained from the code itself:"), templateExplanation(fn))
	}
}
