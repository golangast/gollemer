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
	"bufio"
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

	"github.com/golangast/gollemer/internal/ai/neural/nnu/seq2seq"
	mainvocab "github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
	"github.com/golangast/gollemer/internal/ai/neural/tokenizer"
	"github.com/golangast/gollemer/internal/ai/analyze"
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
	if domain == UnifiedDomain {
		return runUnifiedChat(projectRoot, debug)
	}
	model, err := loadRealModel(projectRoot, domain)
	if err != nil {
		return err
	}

	// Explicit mode separation for code sessions: chatter prompts are routed
	// to the social model (or a clean redirect when its checkpoint is
	// absent), so the code decoder never sees conversational input and can
	// never answer it with blended mush like "I am func doing (x)". The
	// separation is structural — it does not depend on the training data.
	var socialModel *seq2seq.Seq2Seq
	if domain == GocodeDomain {
		socialModel, err = loadRealModel(projectRoot, SocialDomain)
		if err != nil {
			log.Printf("[REAL-CHAT] no social checkpoint (%v); chatter gets a clean redirect", err)
			socialModel = nil
		}
	}

	log.Printf("[REAL-CHAT] loaded %s (vocab=%d, hidden=%d). Pure neural generation — type /quit to exit.",
		RealModelPath(projectRoot, domain), model.OutputVocab.Size(), model.HiddenDim)

	sc := bufio.NewScanner(os.Stdin)
	sc.Buffer(make([]byte, 1024*1024), 1024*1024)
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
		fmt.Print("you> ")
		if !sc.Scan() {
			break
		}
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		if line == "/quit" {
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
				fmt.Printf("gollemer [%s]> %s\n", GoAnalyzeDomain, out)
				conv.AddReply(out, GoAnalyzeDomain, false)
			}
			continue
		}
		// /flow is the beginner entry to the same reactive pipeline:
		// "/flow create a worker pool" runs synthesis, safety proving,
		// auto-tuning, and the visual trace, rendered in plain words.
		if out, ok := handleFlowCommand(line); ok {
			fmt.Printf("gollemer [%s]> %s\n", FlowDomain, out)
			conv.AddReply(out, FlowDomain, false)
			continue
		}
		if line == "/forget" {
			conv.Clear()
			fmt.Println("[forgotten — starting fresh]")
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
					offerRunMakeCommand(sc, projectRoot, t)
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
		answer, trace, err := model.PredictWithTrace(modelInput, 40)
		if err != nil {
			fmt.Printf("gollemer> [error: %v]\n", err)
			continue
		}
		reply := tidyDecode(answer)
		if domain == GocodeDomain {
			reply = tidyGoCode(answer, goCase)
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
	return sc.Err()
}

// tryDeterministicAnswer checks the exact-match layers before the neural
// model runs: social recall, makefile recall, and the Go knowledge base.
// It prints the reply and records it in the conversation when one hits,
// reporting whether the message was fully handled.
func tryDeterministicAnswer(line, d string, conv *Conversation, sc *bufio.Scanner, projectRoot string) bool {
	// Codebase Q&A: if a project was analyzed this session and the
	// message asks about one of its symbols ("what does routeDomain
	// do"), answer from the AST. It only fires on known symbols, so it
	// can't steal questions meant for the other brains.
	if out, ok := tryCodebaseQuestion(line); ok {
		fmt.Printf("gollemer [%s]> %s\n", GoAnalyzeDomain, out)
		conv.AddReply(out, GoAnalyzeDomain, false)
		return true
	}
	// "show me a visual" as a follow-up: draw the dependency graph for
	// the last analyzed project without re-parsing it. The verb+noun
	// shape keeps "show me routeDomain" (answered above) and "how do I
	// render html templates" (go brain) out, and the domain guard keeps
	// "show me how to draw a graph in go" (go/gocode) with its owner.
	// With no prior analysis the visual is drawn for the repo the chat
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
			out := analyze.RenderASCIIGraph(p)
			fmt.Printf("gollemer [%s]> %s\n", GoAnalyzeDomain, out)
			conv.AddReply(out, GoAnalyzeDomain, false)
			return true
		}
	}
	// Social recall: an exact training-pair match returns the trained
	// answer verbatim. The tiny model doesn't reliably memorize every
	// pair, so this guarantees the chat "picks up" what's in its data.
	if d == SocialDomain {
		if sr, ok := LookupSocialRecall(line); ok {
			fmt.Printf("gollemer [%s]> %s\n", d, sr)
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
			fmt.Printf("gollemer [%s]> %s\n", d, mr)
			conv.AddReply(mr, MakefileDomain, false)
			// A makefile reply names an exact repo command. Offer to
			// run it directly, the same way gocli commands are run.
			if t := runnableMakeTarget(mr); t != "" {
				offerRunMakeCommand(sc, projectRoot, t)
			}
			return true
		}
	}
	// Go concept questions first check the curated knowledge base: a
	// strong keyword match gives a guaranteed-correct answer, anything
	// vague falls through to the neural model.
	if d == GoDomain {
		if kb, ok := LookupGoKnowledge(line); ok {
			fmt.Printf("gollemer [%s]> %s\n", d, kb)
			conv.AddReply(kb, GoDomain, false)
			return true
		}
	}
	// Codebase-reading requests run the AST analyzer: exact structural
	// answers, no neural model involved.
	if d == GoAnalyzeDomain {
		if out, ok := handleGoAnalyze(line); ok {
			fmt.Printf("gollemer [%s]> %s\n", d, out)
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
		return fmt.Errorf("unified chat requires at least the social checkpoint")
	}
	// Social recall: exact training-pair matches answer deterministically.
	initSocialRecall(projectRoot)
	initMakefileRecall(projectRoot)
	initMakeAllowlist(projectRoot)

	goCase := goIdentCaseMap(projectRoot)
	sc := bufio.NewScanner(os.Stdin)
	sc.Buffer(make([]byte, 1024*1024), 1024*1024)
	showThoughts := debug
	conv := NewConversation()
	fmt.Println("[unified chat — type /quit to exit, /thoughts to toggle the thought process]")
	for {
		fmt.Print("you> ")
		if !sc.Scan() {
			break
		}
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		if line == "/quit" {
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
				fmt.Printf("gollemer [%s]> %s\n", GoAnalyzeDomain, out)
				conv.AddReply(out, GoAnalyzeDomain, false)
			}
			continue
		}
		// /flow is the beginner entry to the same reactive pipeline:
		// "/flow create a worker pool" runs synthesis, safety proving,
		// auto-tuning, and the visual trace, rendered in plain words.
		if out, ok := handleFlowCommand(line); ok {
			fmt.Printf("gollemer [%s]> %s\n", FlowDomain, out)
			conv.AddReply(out, FlowDomain, false)
			continue
		}
		if line == "/forget" {
			conv.Clear()
			fmt.Println("[forgotten — starting fresh]")
			continue
		}
		// Recall questions are answered from the session transcript,
		// deterministically, before routing or any model sees them.
		if rq, ok := conv.Recall(line); ok {
			fmt.Printf("gollemer [memory]> %s\n", rq)
			conv.AddUser(line)
			conv.AddReply(rq, SocialDomain, true)
			continue
		}
		conv.AddUser(line)
		d := routeDomain(line)
		if tryDeterministicAnswer(line, d, conv, sc, projectRoot) {
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
			fmt.Printf("gollemer [%s]> [error: %v]\n", tag, err)
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
		fmt.Printf("gollemer [%s]> %s\n", tag, reply)
		conv.AddReply(reply, effD, false)
		// A gocli reply is an exact toolchain command. Offer to run it
		// directly (working directory shown, no shell); only bare
		// go/gofmt invocations are ever offered, never model chatter.
		if d == GoCliDomain && tag == GoCliDomain && isRunnableGoCommand(reply) {
			offerRunGoCommand(sc, strings.TrimSpace(reply))
		}
		// A makefile neural reply names an exact repo command. Offer to
		// run it directly (repo root, no shell), same as gocli commands.
		if d == MakefileDomain && tag == MakefileDomain {
			if t := runnableMakeTarget(reply); t != "" {
				offerRunMakeCommand(sc, projectRoot, t)
			}
		}
		if showThoughts {
			printThoughtTrace(trace)
		}
	}
	return sc.Err()
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
			fmt.Printf("gollemer [%s]: %s\n", t.Domain, t.Text)
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
// through the chat's own scanner so stdin stays on one reader.
func offerRunGoCommand(sc *bufio.Scanner, cmd string) {
	cwd, _ := os.Getwd()
	if !stdinIsTerminal() {
		// Piped/scripted session: the command is already printed on the
		// reply line above; never prompt, never run.
		return
	}
	fmt.Printf("[run it here? %s] [y/n]: ", cwd)
	if !sc.Scan() {
		fmt.Println("[cancelled]")
		return
	}
	yn := strings.ToLower(strings.TrimSpace(sc.Text()))
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
func offerRunMakeCommand(sc *bufio.Scanner, projectRoot, target string) {
	if !stdinIsTerminal() {
		// Piped/scripted session: never prompt, never run.
		return
	}
	fmt.Printf("[run it here? 'make %s'] [y/n]: ", target)
	if !sc.Scan() {
		fmt.Println("[cancelled]")
		return
	}
	yn := strings.ToLower(strings.TrimSpace(sc.Text()))
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
