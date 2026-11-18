package chat

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/golangast/gollemer/internal/ai/neural/nn"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

// The speccopy optimizer sidecar must round-trip (the restored Adam
// takes the identical next step as the saved one) and must REFUSE
// to load against a checkpoint it does not belong to — wrong epoch,
// wrong vocabulary, missing file — so the trainer's loud-fresh-Adam
// fallback, never a silent mismatch, is what a bad sidecar gets.
func TestSpecCopyOptStateSidecarRoundTripAndValidation(t *testing.T) {
	mkAdam := func() (*nn.Adam, []*tensor.Tensor) {
		params := []*tensor.Tensor{
			tensor.NewTensor([]int{2}, []float32{0.5, -0.25}, true),
			tensor.NewTensor([]int{2}, []float32{0.1, 0.3}, true),
		}
		adam, ok := nn.NewOptimizer(params, 0.01, 1.0).(*nn.Adam)
		if !ok {
			t.Fatal("optimizer is not *nn.Adam")
		}
		for s := 0; s < 2; s++ {
			for _, p := range params {
				g := make([]float32, len(p.Data))
				for i := range g {
					g[i] = 0.1 + 0.05*float32(i) + 0.02*float32(s)
				}
				p.Grad = tensor.NewTensor(p.Shape, g, false)
			}
			adam.Step()
		}
		return adam, params
	}

	src, srcParams := mkAdam()
	path := filepath.Join(t.TempDir(), "model.gob.latest.opt")
	const fp = 424242
	if err := saveSpecCopyOptState(path, 57, fp, src); err != nil {
		t.Fatalf("saveSpecCopyOptState: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("sidecar not written: %v", err)
	}

	// Round-trip: load into a fresh Adam over identical weights and
	// confirm step count and state survived.
	dst, dstParams := mkAdam()
	// dst has already taken 2 steps of its own; wipe to a truly fresh
	// optimizer over the same weights to prove the load, not luck.
	fresh, ok := nn.NewOptimizer(dstParams, 0.01, 1.0).(*nn.Adam)
	if !ok {
		t.Fatal("optimizer is not *nn.Adam")
	}
	_ = dst
	if err := loadSpecCopyOptState(path, 57, fp, fresh); err != nil {
		t.Fatalf("loadSpecCopyOptState: %v", err)
	}
	got := fresh.SnapshotState()
	want := src.SnapshotState()
	if got.T != want.T {
		t.Fatalf("restored adam step = %d, want %d", got.T, want.T)
	}
	for i := range want.MData {
		for j := range want.MData[i] {
			if got.MData[i][j] != want.MData[i][j] || got.VData[i][j] != want.VData[i][j] {
				t.Fatalf("moment mismatch at param %d[%d]", i, j)
			}
		}
	}
	_ = srcParams

	// Validation: every mismatch is an error.
	other, _ := mkAdam()
	if err := loadSpecCopyOptState(path, 58, fp, other); err == nil {
		t.Fatal("sidecar loaded against the wrong epoch")
	}
	if err := loadSpecCopyOptState(path, 57, fp+1, other); err == nil {
		t.Fatal("sidecar loaded against the wrong vocab fingerprint")
	}
	if err := loadSpecCopyOptState(path+".missing", 57, fp, other); err == nil {
		t.Fatal("missing sidecar loaded without error")
	}
	if err := os.WriteFile(path+".corrupt", []byte("not a gob"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := loadSpecCopyOptState(path+".corrupt", 57, fp, other); err == nil {
		t.Fatal("corrupt sidecar loaded without error")
	}
}
