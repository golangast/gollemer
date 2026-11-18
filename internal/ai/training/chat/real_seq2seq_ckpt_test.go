package chat

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/golangast/gollemer/internal/ai/neural/nnu/seq2seq"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
	"github.com/golangast/gollemer/internal/ai/neural/tokenizer"
)

// tinyDeclModel builds the smallest seq2seq model the checkpoint helpers
// can save and load, on an empty vocabulary (pad/unk/bos/eos only).
func tinyCheckpointModel(t *testing.T) (*seq2seq.Seq2Seq, *tokenizer.Tokenizer, *vocab.Vocabulary) {
	t.Helper()
	v := vocab.NewVocabulary()
	tok, err := tokenizer.NewTokenizer(v)
	if err != nil {
		t.Fatalf("NewTokenizer: %v", err)
	}
	m, err := seq2seq.NewSeq2Seq(v.Size(), v.Size(), 8, 16, tok, v)
	if err != nil {
		t.Fatalf("NewSeq2Seq: %v", err)
	}
	return m, tok, v
}

// The latest checkpoint must live beside the best checkpoint under its
// own name, so the two can never be confused for each other.
func TestLatestCkptPathsBesideBest(t *testing.T) {
	root := filepath.Join("some", "root")
	best, _ := ckptPaths(root, "decl")
	latest, latestMeta := latestCkptPaths(root, "decl")
	if !strings.HasSuffix(latest, "real_tiny_seq2seq_decl.gob.latest") {
		t.Fatalf("latest path = %q, want ...real_tiny_seq2seq_decl.gob.latest", latest)
	}
	if !strings.HasSuffix(latestMeta, "real_tiny_seq2seq_decl.gob.latest.json") {
		t.Fatalf("latest meta path = %q, want ...real_tiny_seq2seq_decl.gob.latest.json", latestMeta)
	}
	if filepath.Dir(latest) != filepath.Dir(best) {
		t.Fatalf("latest dir %q != best dir %q", filepath.Dir(latest), filepath.Dir(best))
	}
}

// The contract the decl run died without: a saved sidecar restores the
// epoch, the learning rate, and — above all — a NONZERO no-improvement
// counter, so early-stopping stagnation accumulates across restarts
// instead of resetting at every resume.
func TestCheckpointRoundTripRestoresResumeState(t *testing.T) {
	m, tok, v := tinyCheckpointModel(t)
	fp := vocabFingerprint(v)
	dir := t.TempDir()
	ckptPath := filepath.Join(dir, "model.gob.latest")
	metaPath := filepath.Join(dir, "model.gob.latest.json")
	want := ckptMeta{
		Epoch:              58,
		Best:               0.021247,
		LastCheckpointBest: 0.021247,
		LR:                 0.0005,
		EpochsNoImprove:    4,
		VocabFP:            fp,
	}
	if err := saveTrainingCheckpoint(m, want, ckptPath, metaPath); err != nil {
		t.Fatalf("saveTrainingCheckpoint: %v", err)
	}
	loaded, got, err := loadTrainingCheckpoint(tok, fp, ckptPath, metaPath)
	if err != nil {
		t.Fatalf("loadTrainingCheckpoint: %v", err)
	}
	if loaded == nil {
		t.Fatal("loadTrainingCheckpoint returned a nil model")
	}
	if got != want {
		t.Fatalf("meta round-trip mismatch:\n got %+v\nwant %+v", got, want)
	}
	// A vocab change must still refuse the resume.
	if _, _, err := loadTrainingCheckpoint(tok, fp+1, ckptPath, metaPath); err == nil {
		t.Fatal("loadTrainingCheckpoint accepted a mismatched vocab fingerprint")
	}
}

// copyWeightsInto must move values, not instances: the destination keeps
// its own tensor pointers (optimizer snapshots are keyed by pointer)
// while ending up with the source's values in every parameter.
func TestCopyWeightsIntoPreservesTensors(t *testing.T) {
	dst, _, _ := tinyCheckpointModel(t)
	src, _, _ := tinyCheckpointModel(t)
	dp, sp := dst.Parameters(), src.Parameters()
	if len(dp) != len(sp) || len(dp) == 0 {
		t.Fatalf("parameter count mismatch or empty: dst %d src %d", len(dp), len(sp))
	}
	// Give src distinctive values so equality afterwards is meaningful.
	for _, p := range sp {
		for i := range p.Data {
			p.Data[i] = 3.25
		}
	}
	firstPtr := dp[0]
	if err := copyWeightsInto(dst, src); err != nil {
		t.Fatalf("copyWeightsInto: %v", err)
	}
	if dst.Parameters()[0] != firstPtr {
		t.Fatal("copyWeightsInto replaced a destination tensor; snapshots keyed by pointer would be orphaned")
	}
	for i, p := range dp {
		for j, val := range p.Data {
			if val != 3.25 {
				t.Fatalf("param %d data %d = %v, want 3.25", i, j, val)
			}
		}
	}
}
