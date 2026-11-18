package chat

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestSpecDomainPreparesTrainAll pins the spec probe's trainer
// wiring: its dataset file is train-only by construction (the decl
// pipeline holds out combinations and frames before the pairs are
// derived), so prepareDomainTrainData must train on every pair, hold
// nothing back for in-trainer probes, and apply no synonym
// augmentation — the same discipline as the decl domain.
func TestSpecDomainPreparesTrainAll(t *testing.T) {
	root := t.TempDir()
	dir := filepath.Join(root, "data", "training")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	var b strings.Builder
	for i := 0; i < 12; i++ {
		fmt.Fprintf(&b, `{"input":"make a struct named thing%d with a field named f%d of type int","output":"spec struct thing%d field f%d int","domain":"spec"}`+"\n", i, i, i, i)
	}
	if err := os.WriteFile(filepath.Join(dir, "spec_pairs.jsonl"), []byte(b.String()), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := SpecDatasetPath(root); got != filepath.Join(root, SpecDatasetRelPath) {
		t.Fatalf("SpecDatasetPath = %q", got)
	}
	dd, err := prepareDomainTrainData(root, SpecDomain)
	if err != nil {
		t.Fatalf("prepareDomainTrainData(spec): %v", err)
	}
	if len(dd.heldBase) != 0 || len(dd.probes) != 0 {
		t.Fatalf("spec domain must hold nothing back: held=%d probes=%d", len(dd.heldBase), len(dd.probes))
	}
	if len(dd.trainPairs) != 12 {
		t.Fatalf("spec domain must train on all pairs unaugmented: got %d train pairs, want 12", len(dd.trainPairs))
	}
	if dd.v == nil || dd.v.Size() == 0 || dd.tok == nil {
		t.Fatal("spec domain prepared no vocabulary/tokenizer")
	}
}
