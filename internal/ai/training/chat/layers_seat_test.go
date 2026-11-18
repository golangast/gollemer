package chat

// Layers adoption (layers_seat.go) tests: the absence path is
// proven unconditionally (no gob under the root -> no brain,
// and a corrupt gob -> no brain, no panic). The presence path's
// plumbing proof — the chat's loader and predict entry agree
// with scripts/score_layers on the same gob, byte for byte
// except where the checked-sampling rescue (layers_decode.go,
// seated 2026-11-18) deliberately replaces a topic-dropping
// greedy reply with a topic-carrying sampled one — is
// env-gated, because it needs a real layers gob and a captured
// scorer run:
//
//	LAYERS_XCHECK_MODEL=<layers .gob> \
//	LAYERS_XCHECK_SCORER=<score_layers -eval output file> \
//	LAYERS_XCHECK_EVAL=<robustness_v1.jsonl> \
//	go test ./internal/ai/training/chat/ -run TestLayersSeatCrossCheckScorer -v
//
// The comparison feeds each frozen Section S input RAW to the
// seat's predict — the exact string the scorer feeds
// PredictLayersText — so it proves the wiring (loader state,
// options, session handling), not model quality and not the
// loops' input composition, which is pre-existing chat
// behavior shared with the production model.

import (
	"bufio"
	"encoding/json"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func TestLayersBrainAbsentRoot(t *testing.T) {
	if b := loadLayersSocialBrain(t.TempDir()); b != nil {
		t.Fatalf("expected nil brain for a root with no layers gob, got %+v", b)
	}
}

func TestLayersBrainCorruptGob(t *testing.T) {
	root := t.TempDir()
	path := LayersModelPath(root)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("this is not a gob"), 0o644); err != nil {
		t.Fatal(err)
	}
	if b := loadLayersSocialBrain(root); b != nil {
		t.Fatalf("expected nil brain for a corrupt layers gob, got %+v", b)
	}
}

// parseGoQuoted reads one Go %q-quoted string starting at or
// after index i (skipping spaces) and returns its value and
// the index just past the closing quote.
func parseGoQuoted(t *testing.T, line string, i int) (string, int) {
	t.Helper()
	for i < len(line) && line[i] == ' ' {
		i++
	}
	if i >= len(line) || line[i] != '"' {
		t.Fatalf("expected quoted string at %d in %q", i, line)
	}
	j := i + 1
	for j < len(line) {
		if line[j] == '\\' {
			j += 2
			continue
		}
		if line[j] == '"' {
			break
		}
		j++
	}
	if j >= len(line) {
		t.Fatalf("unterminated quoted string in %q", line)
	}
	s, err := strconv.Unquote(line[i : j+1])
	if err != nil {
		t.Fatalf("unquote %q: %v", line[i:j+1], err)
	}
	return s, j + 1
}

func TestLayersSeatCrossCheckScorer(t *testing.T) {
	modelPath := os.Getenv("LAYERS_XCHECK_MODEL")
	scorerPath := os.Getenv("LAYERS_XCHECK_SCORER")
	evalPath := os.Getenv("LAYERS_XCHECK_EVAL")
	if modelPath == "" || scorerPath == "" || evalPath == "" {
		t.Skip("LAYERS_XCHECK_MODEL / LAYERS_XCHECK_SCORER / LAYERS_XCHECK_EVAL not set")
	}

	// The eval inputs, for parse sanity: every scorer item
	// line's input must be one of them.
	evalFile, err := os.Open(evalPath)
	if err != nil {
		t.Fatal(err)
	}
	defer evalFile.Close()
	evalInputs := map[string]bool{}
	esc := bufio.NewScanner(evalFile)
	esc.Buffer(make([]byte, 1024*1024), 1024*1024)
	for esc.Scan() {
		var it struct {
			Section string `json:"section"`
			Input   string `json:"input"`
		}
		if err := json.Unmarshal(esc.Bytes(), &it); err != nil {
			t.Fatalf("bad eval line: %v", err)
		}
		if it.Section == "S" {
			evalInputs[it.Input] = true
		}
	}
	if err := esc.Err(); err != nil {
		t.Fatal(err)
	}

	// The scorer's per-item lines: PASS|FAIL <id> [fam/var]
	// %q -> %q (want-hit n/m).
	type item struct{ input, reply string }
	var items []item
	scorerFile, err := os.Open(scorerPath)
	if err != nil {
		t.Fatal(err)
	}
	defer scorerFile.Close()
	ssc := bufio.NewScanner(scorerFile)
	ssc.Buffer(make([]byte, 1024*1024), 1024*1024)
	for ssc.Scan() {
		line := ssc.Text()
		if !strings.HasPrefix(line, "PASS ") && !strings.HasPrefix(line, "FAIL ") {
			continue
		}
		bracket := strings.Index(line, "] ")
		if bracket < 0 {
			t.Fatalf("malformed scorer line: %q", line)
		}
		input, next := parseGoQuoted(t, line, bracket+2)
		if !strings.HasPrefix(line[next:], " -> ") {
			t.Fatalf("malformed scorer line (no ->): %q", line)
		}
		reply, _ := parseGoQuoted(t, line, next+4)
		if !evalInputs[input] {
			t.Fatalf("scorer input %q is not a Section S eval input", input)
		}
		items = append(items, item{input, reply})
	}
	if err := ssc.Err(); err != nil {
		t.Fatal(err)
	}
	if len(items) < 40 {
		t.Fatalf("only %d scorer items parsed; need a real sample", len(items))
	}

	// Load the same gob through the chat's own loader, under
	// a scratch root at the canonical path — the exact state
	// a chat loop constructs.
	root := t.TempDir()
	dst := LayersModelPath(root)
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(dst, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	brain := loadLayersSocialBrain(root)
	if brain == nil {
		t.Fatalf("chat loader returned no brain for %s", modelPath)
	}
	if layers := brain.model.TrainedLayers(); len(layers) != 1 || layers[0] != "copy" {
		t.Fatalf("throwaway model should be copy-only, trained layers: %v", layers)
	}

	// The seat's rng is reseeded so the rescue's draws are
	// deterministic across runs of this check.
	brain.rng = rand.New(rand.NewSource(20261118))
	mismatches, rescued := 0, 0
	for _, it := range items {
		got, err := brain.predict(it.input, it.input)
		if err != nil {
			t.Fatalf("predict %q: %v", it.input, err)
		}
		if got == it.reply {
			continue
		}
		// The one sanctioned divergence: the scorer's greedy
		// reply dropped the input's topic and the seat's
		// checked sampling replaced it with a reply that
		// carries it. Anything else is a wiring break.
		if !layersReplyCarries(it.input, it.reply) && layersReplyCarries(it.input, got) {
			rescued++
			continue
		}
		mismatches++
		if mismatches <= 5 {
			t.Errorf("input %q: chat seat %q != scorer %q (and not a topic rescue)", it.input, got, it.reply)
		}
	}
	t.Logf("cross-check: %d items compared, %d rescued by checked sampling, %d mismatches", len(items), rescued, mismatches)
	if mismatches > 0 {
		t.Fatalf("%d/%d replies differ between the chat seat and score_layers outside the topic rescue", mismatches, len(items))
	}
}
