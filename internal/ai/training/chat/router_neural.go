package chat

import (
	"log"
	"strings"
	"sync"

	"github.com/golangast/gollemer/internal/ai/neural/nnu/seq2seq"
)

// RouterDomain is the classifier domain: input is a user message, output
// is the domain label (social, go, gocode, gocli, makefile) it belongs to.
// It replaces the hand-written routeDomain regexes with a trained model;
// the regexes stay as the fallback when the model is absent.
const RouterDomain = "router"

// RouterDatasetRelPath is the router training data location relative to
// the project root: (message, domain-label) pairs derived from the
// quality-gated chat pairs.
const RouterDatasetRelPath = "data/training/router_pairs.jsonl"

// routerLabels are the only outputs the neural router may produce. The
// router never emits goanalyze (deterministic, handled before routing),
// unified, flow, or router itself.
var routerLabels = map[string]bool{
	SocialDomain:   true,
	GoDomain:       true,
	GocodeDomain:   true,
	GoCliDomain:    true,
	MakefileDomain: true,
}

var routerModelState = struct {
	sync.Mutex
	root  string
	model *seq2seq.Seq2Seq
	err   error
}{}

// routeNeural classifies input with the trained router model. It returns
// ok=false when the model file is absent or unusable, or when the model
// emits something that is not a valid domain label — callers fall back
// to the regex router in all those cases.
func routeNeural(projectRoot, input string) (string, bool) {
	if projectRoot == "" {
		return "", false
	}
	routerModelState.Lock()
	defer routerModelState.Unlock()
	if routerModelState.root != projectRoot || (routerModelState.model == nil && routerModelState.err == nil) {
		m, err := loadRealModel(projectRoot, RouterDomain)
		routerModelState.root = projectRoot
		routerModelState.model = m
		routerModelState.err = err
		if err != nil {
			log.Printf("[ROUTER] no router model (%v); using regex fallback", err)
		}
	}
	if routerModelState.err != nil || routerModelState.model == nil {
		return "", false
	}
	out, _, err := routerModelState.model.PredictWithTrace(strings.ToLower(input), 8)
	if err != nil {
		return "", false
	}
	label := strings.ToLower(strings.TrimSpace(tidyDecode(out)))
	// The label must be a single known domain token; anything else
	// (chatter, multi-word output, empty) means the model is unsure.
	if !routerLabels[label] || strings.ContainsAny(label, " \t\n") {
		return "", false
	}
	return label, true
}
