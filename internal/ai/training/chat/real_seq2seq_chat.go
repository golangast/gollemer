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
	"fmt"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/golangast/gollemer/internal/ai/neural/nnu/seq2seq"
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
func RunRealChat(projectRoot, domain string) error {
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
	showThoughts := true
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
		if domain == GocodeDomain && chatterPrompt.MatchString(line) && !gocodeTerms.MatchString(line) {
			// Conversational mode: answer with the chat model, never the
			// code decoder. The gocodeTerms guard keeps greeting-prefixed
			// code requests ("hey, write a function...") on the code model.
			// Chat replies get chat post-processing only — tidyGoCode must
			// never touch them.
			if socialModel != nil {
				answer, trace, err := socialModel.PredictWithTrace(strings.ToLower(line), 40)
				if err != nil {
					fmt.Printf("gollemer> [error: %v]\n", err)
					continue
				}
				fmt.Printf("gollemer> %s\n", tidyDecode(answer))
				if showThoughts {
					printThoughtTrace(trace)
				}
			} else {
				fmt.Printf("gollemer> %s\n", gocodeChatterRedirect)
			}
			continue
		}
		// Lowercased to match training; the model is case-insensitive by construction.
		answer, trace, err := model.PredictWithTrace(strings.ToLower(line), 40)
		if err != nil {
			fmt.Printf("gollemer> [error: %v]\n", err)
			continue
		}
		reply := tidyDecode(answer)
		if domain == GocodeDomain {
			reply = tidyGoCode(answer, goCase)
		}
		fmt.Printf("gollemer> %s\n", reply)
		if showThoughts {
			printThoughtTrace(trace)
		}
	}
	return sc.Err()
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
