package simple

import (
	"math"
	"testing"
)

// TestDenseModelLearnAND verifies the full stack (struct dataset + dense MLP
// + deterministic trainer) learns a trivial AND classifier.
func TestDenseModelLearnAND(t *testing.T) {
	alphabet := "ab" // a=bit0, b=bit1 (one-hot features)
	ds := NewDataset(1,
		Sample{Input: CharEncode("", alphabet), Label: 0},   // 0,0
		Sample{Input: CharEncode("b", alphabet), Label: 0},  // 0,1
		Sample{Input: CharEncode("a", alphabet), Label: 0},  // 1,0
		Sample{Input: CharEncode("ab", alphabet), Label: 1}, // 1,1
	)

	model := NewDenseModel(ds.FeatureSize(), []int{8}, ds.NumClasses())
	trainer := NewTrainer(model, Config{
		Epochs:      300,
		BatchSize:   4,
		BaseLR:      5e-2,
		MinLR:       1e-3,
		WarmupSteps: 10,
		LogEvery:    1 << 30, // suppress per-step logging in tests
	})

	loss, err := trainer.Train(ds)
	if err != nil {
		t.Fatalf("Train: %v", err)
	}
	if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
		t.Fatalf("loss not finite: %v", loss)
	}

	pred := model.Predict(inputsForAll(ds))
	want := labelsForAll(ds)
	if acc := Accuracy(pred, want); acc < 1.0 {
		t.Fatalf("train accuracy = %.2f (pred=%v want=%v)", acc, pred, want)
	}
}

// TestDenseModelNaNGuard verifies the hard assertion: NaN loss aborts training.
func TestDenseModelNaNGuard(t *testing.T) {
	// 1-feature input with weights = MaxFloat32 forces Inf/NaN logits
	// deterministically: big*big overflows float32 to +Inf.
	ds := NewDataset(0,
		Sample{Input: []float32{math.MaxFloat32}, Label: 0},
	)

	model := NewDenseModel(1, nil, 2)
	model.Layers[0].Weights = []float32{math.MaxFloat32, math.MaxFloat32}
	model.Layers[0].Bias = []float32{0, 0}

	trainer := NewTrainer(model, Config{
		Epochs:      2,
		BatchSize:   1,
		BaseLR:      1e-1,
		MinLR:       1e-3,
		WarmupSteps: 1,
		LogEvery:    1 << 30,
	})
	_, err := trainer.Train(ds)
	if err == nil {
		t.Fatal("expected NaN guard to abort training, got nil error")
	}
}

// TestDatasetBounds verifies dataset validation rejects malformed samples.
func TestDatasetBounds(t *testing.T) {
	ds := &Dataset{Samples: []Sample{
		{Input: []float32{1, 2}, Label: 0},
		{Input: []float32{1}, Label: 1}, // inconsistent feature size
	}}
	if err := ds.Bounds(); err == nil {
		t.Fatal("expected bounds error for inconsistent feature size")
	}
}

// TestDenseModelLearnCommands verifies the dense MLP learns to separate the
// basic update-command corpus (social vs code_update) using the same
// architecture and hyperparameters as the dense_train / dense_llm CLIs.
func TestDenseModelLearnCommands(t *testing.T) {
	ds := CommandDataset()

	// Same architecture as dense_train: vocab size -> [16] -> 2 classes.
	model := NewDenseModel(ds.FeatureSize(), []int{16}, ds.NumClasses())
	trainer := NewTrainer(model, Config{
		Epochs:      300,
		BatchSize:   8,
		BaseLR:      5e-2,
		MinLR:       1e-3,
		WarmupSteps: 10,
		LogEvery:    1 << 30, // suppress per-step logging in tests
	})

	loss, err := trainer.Train(ds)
	if err != nil {
		t.Fatalf("Train: %v", err)
	}
	if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
		t.Fatalf("loss not finite: %v", loss)
	}

	pred := model.Predict(inputsForAll(ds))
	want := labelsForAll(ds)
	if acc := Accuracy(pred, want); acc < 1.0 {
		t.Fatalf("command train accuracy = %.2f (pred=%v want=%v)", acc, pred, want)
	}

	// Sanity-check a few held-back phrasings that share vocabulary with the
	// corpus: "hello" -> social, "add brace to if" -> code_update.
	socialPred := model.Predict([][]float32{BagOfWords("hello, how are you doing?", CommandVocab)})
	if len(socialPred) > 0 && CommandLabels[socialPred[0]] != "social" {
		t.Fatalf("expected social classification, got %q", CommandLabels[socialPred[0]])
	}
	codePred := model.Predict([][]float32{BagOfWords("add missing brace to the if statement", CommandVocab)})
	if len(codePred) > 0 && CommandLabels[codePred[0]] != "code_update" {
		t.Fatalf("expected code_update classification, got %q", CommandLabels[codePred[0]])
	}
}

func TestCommandTypeClassificationIncludesFileAndCodeActions(t *testing.T) {
	checks := map[string]string{
		"hello how are you":        "social",
		"add missing brace to if":  "code_update",
		"create file main.go":      "file_create",
		"modify file config.json":  "file_edit",
		"delete file tmp.log":      "file_delete",
		"what is folder jim":       "folder_query",
		"create folder jim":        "folder_create",
		"delete folder old":        "folder_delete",
		"fix file jim/jim.go":      "file_edit",
		"create function ping":     "code_update",
		"remove function cleanup":  "code_update",
		"edit function user setup": "code_update",
	}
	for prompt, want := range checks {
		if got := ClassifyCommandType(prompt); got != want {
			t.Fatalf("ClassifyCommandType(%q) = %q, want %q", prompt, got, want)
		}
	}
	if !containsLabel("file_create") || !containsLabel("file_edit") || !containsLabel("file_delete") ||
		!containsLabel("folder_create") || !containsLabel("folder_delete") || !containsLabel("folder_query") {
		t.Fatal("expected create/edit/delete and query file and folder labels in CommandLabels")
	}
}

func containsLabel(label string) bool {
	for _, l := range CommandLabels {
		if l == label {
			return true
		}
	}
	return false
}

func TestInferTargetPathFromPrompt(t *testing.T) {
	cases := map[string]string{
		"fix file jim/jim.go":  "jim/jim.go",
		"fix /file jim/jim.go": "jim/jim.go",
		"create file main.go":  "main.go",
		"what is folder jim":   "jim",
		"delete folder old":    "old",
	}
	for prompt, want := range cases {
		if got, _ := inferTargetFromPrompt(prompt); got != want {
			t.Fatalf("inferTargetFromPrompt(%q) = %q, want %q", prompt, got, want)
		}
	}
}

// TestCosineLR verifies warmup and cosine decay are deterministic and bounded.
func TestCosineLR(t *testing.T) {
	total := 1000
	warmup := 20
	base := float32(1e-2)
	min := float32(1e-4)

	// step 0: warmup start
	lr0 := cosineLR(0, warmup, base, min, total)
	if lr0 <= 0 || lr0 > base {
		t.Fatalf("warmup start out of range: %v", lr0)
	}
	// step warmup-1: peak
	lrPeak := cosineLR(warmup-1, warmup, base, min, total)
	if math.Abs(float64(lrPeak-base)) > 1e-5 {
		t.Fatalf("warmup peak != base: got %v want %v", lrPeak, base)
	}
	// step total: minimum
	lrEnd := cosineLR(total, warmup, base, min, total)
	if math.Abs(float64(lrEnd-min)) > 1e-5 {
		t.Fatalf("decay end != min: got %v want %v", lrEnd, min)
	}
	// monotonic non-increasing after warmup
	prev := cosineLR(warmup, warmup, base, min, total)
	for step := warmup + 1; step <= total; step += 10 {
		cur := cosineLR(step, warmup, base, min, total)
		if cur > prev+1e-6 {
			t.Fatalf("cosine decay not monotonic at step %d: %v > %v", step, cur, prev)
		}
		prev = cur
	}
}
