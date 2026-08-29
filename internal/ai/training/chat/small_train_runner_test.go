package chat

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/golangast/gollemer/internal/ai/moe"
	seq2seq "github.com/golangast/gollemer/internal/ai/neural/nnu/seq2seq"
	mainvocab "github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
	"github.com/golangast/gollemer/internal/ai/neural/tokenizer"
)

func TestCleanSeq2SeqOutput(t *testing.T) {
	input := "hello hello hello how are you how are you today today today"
	output := cleanSeq2SeqOutput(input)
	if strings.Count(output, "hello") > 1 {
		t.Fatalf("expected repeated words to be deduped, got %q", output)
	}
	if strings.Count(output, "how") > 1 {
		t.Fatalf("expected repeated words to be deduped, got %q", output)
	}
	if strings.Count(output, "today") > 1 {
		t.Fatalf("expected repeated words to be deduped, got %q", output)
	}
	if output == "" {
		t.Fatal("cleaned output should not be empty")
	}
}

func TestTinySeq2SeqDiagnostic(t *testing.T) {
	root := filepath.Join("..", "..", "..", "..")
	loss, err := TrainTinySeq2SeqDiagnostic(root)
	if err != nil {
		t.Fatalf("tiny seq2seq diagnostic failed: %v", err)
	}
	if loss > 0.05 {
		t.Fatalf("tiny seq2seq diagnostic loss too high: %.6f", loss)
	}
}

func TestTinySeq2SeqCurriculumDoublesData(t *testing.T) {
	pairs := []moe.TrainPair{
		{Q: "hi", A: "hello"},
		{Q: "bye", A: "goodbye"},
	}
	doubled := doubleDatasetPairs(pairs)
	if len(doubled) != len(pairs)*2 {
		t.Fatalf("expected doubled dataset size %d, got %d", len(pairs)*2, len(doubled))
	}
	seen := map[string]bool{}
	for _, pair := range doubled {
		if seen[pair.Q] {
			t.Fatalf("doubled dataset contains duplicate question %q", pair.Q)
		}
		seen[pair.Q] = true
	}
	if len(seen) != len(doubled) {
		t.Fatal("doubled dataset did not produce unique prompt variants")
	}
}

func TestLoadTinyPairsFiltersNoisyRows(t *testing.T) {
	pairs := []moe.TrainPair{
		{Q: "hi", A: "hello there"},
		{Q: "what is your name", A: "i am gollemer, your ai assistant."},
		{Q: "host webhook tag syntax urgent feeling", A: "host host host urgent webhook webhook webhook tag certainly syntax feeling -"},
	}
	filtered := filterTinySeq2SeqPairs(pairs)
	if len(filtered) != 2 {
		t.Fatalf("expected 2 clean rows after filtering, got %d", len(filtered))
	}
	if filtered[0].Q != "hi" || filtered[1].Q != "what is your name" {
		t.Fatalf("unexpected surviving rows: %#v", filtered)
	}
}

func TestTinySeq2SeqCurriculumRunner(t *testing.T) {
	root := filepath.Join("..", "..", "..", "..")
	loss := RunTinySeq2SeqCurriculumCheck(root)
	if loss > 0.002 {
		t.Fatalf("curriculum runner did not hit the low-loss target: %.6f", loss)
	}

	modelPath := filepath.Join(root, "data", "models", "gob_models", "tiny_seq2seq_demo.gob")
	if _, err := os.Stat(modelPath); err != nil {
		t.Fatalf("curriculum runner did not save a usable model artifact at %s: %v", modelPath, err)
	}

	vocab := mainvocab.NewVocabulary()
	pairs, err := loadCustomSocialPairs(filepath.Join(root, "data", "training", "trainingdata", "small_social_demo.csv"))
	if err != nil {
		t.Fatalf("load small demo pairs: %v", err)
	}
	for _, pair := range pairs {
		for _, tok := range cleanTokenize(pair.Q + " " + pair.A) {
			vocab.AddToken(tok)
		}
	}
	if token, err := tokenizer.NewTokenizer(vocab); err != nil {
		t.Fatalf("new tokenizer: %v", err)
	} else if model, err := seq2seq.Load(modelPath, token); err != nil {
		t.Fatalf("load final tiny seq2seq model: %v", err)
	} else if answer, err := model.Predict("hi", 12); err != nil {
		t.Fatalf("predict with final tiny seq2seq model: %v", err)
	} else if answer == "" {
		t.Fatal("final tiny seq2seq model produced empty answer")
	}
}

func TestTinySeq2SeqSnapshotRoundTrip(t *testing.T) {
	root := filepath.Join("..", "..", "..", "..")
	modelPath := filepath.Join(root, "data", "models", "gob_models", "tiny_seq2seq_demo.gob")
	if err := os.MkdirAll(filepath.Dir(modelPath), 0o755); err != nil {
		t.Fatalf("mkdir model dir: %v", err)
	}

	vocab := mainvocab.NewVocabulary()
	vocab.AddToken("hello")
	vocab.AddToken("world")
	vocab.AddToken("hi")
	vocab.AddToken("bye")
	_, err := tokenizer.NewTokenizer(vocab)
	if err != nil {
		t.Fatalf("new tokenizer: %v", err)
	}

	model, err := seq2seq.NewSeq2Seq(vocab.Size(), vocab.Size(), 4, 8, nil, vocab)
	if err != nil {
		t.Fatalf("new seq2seq model: %v", err)
	}
	if err := model.Save(modelPath); err != nil {
		t.Fatalf("save seq2seq model: %v", err)
	}

	loaded, err := seq2seq.Load(modelPath, nil)
	if err != nil {
		t.Fatalf("load seq2seq model: %v", err)
	}
	if loaded == nil || loaded.OutputVocab == nil {
		t.Fatal("loaded seq2seq model was nil or missing vocab")
	}
}

func TestFormatChatResponse(t *testing.T) {
	raw := "Hello! I'm doing great, thank you for asking!\n\n" +
		"[PREDICTIVE_REASONING]\n" +
		"- ENTITIES: Subject=Go Module system | Object=Dependency management\n" +
		"- TARGET_GOAL: Explain go.mod\n" +
		"[RESPONSE] A Go module is defined by a `go.mod` file."
	got := FormatChatResponse(raw)
	if strings.Contains(got, "PREDICTIVE_REASONING") || strings.Contains(got, "TARGET_GOAL") {
		t.Fatalf("reasoning trace leaked into chat response:\n%s", got)
	}
	want := "Hello! I'm doing great, thank you for asking!\n\nA Go module is defined by a `go.mod` file."
	if got != want {
		t.Fatalf("FormatChatResponse mismatch:\n got: %q\nwant: %q", got, want)
	}

	// Plain answers without a [RESPONSE] tag pass through unchanged.
	plain := "I'm doing great, thanks for asking!"
	if got := FormatChatResponse(plain); got != plain {
		t.Fatalf("plain answer altered: got %q, want %q", got, plain)
	}
}
