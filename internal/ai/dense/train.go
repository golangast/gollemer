package dense

import (
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
)

// Config holds the fixed, deterministic training hyperparameters. There is no
// dynamic reflection: the learning rate follows a static cosine schedule and
// no expert spawning/eviction occurs.
type Config struct {
	Epochs      int
	BatchSize   int
	BaseLR      float32
	MinLR       float32
	WarmupSteps int
	LogEvery    int // log a line every N steps to stdout
	// StagnantEpochLimit is the number of consecutive epochs whose average loss
	// fails to improve before training aborts. 0 means the default (100).
	StagnantEpochLimit int
}

// DefaultConfig returns a conservative, reproducible configuration.
func DefaultConfig() Config {
	return Config{
		Epochs:             200,
		BatchSize:          8,
		BaseLR:             1e-2,
		MinLR:              1e-4,
		WarmupSteps:        20,
		LogEvery:           10,
		StagnantEpochLimit: 100,
	}
}

// Trainer is a plain dense trainer. It owns no supervisor, no MoE router, and
// no autonomous hyperparameter reflection.
type Trainer struct {
	Model  *DenseModel
	Config Config
	Step   int
}

// NewTrainer wires a model to a config.
func NewTrainer(m *DenseModel, cfg Config) *Trainer {
	return &Trainer{Model: m, Config: cfg}
}

// lrAt returns the learning rate for a given global step using linear warmup
// followed by cosine decay to MinLR. This is fully deterministic.
func (t *Trainer) lrAt(step int) float32 {
	cfg := t.Config
	return cosineLR(step, cfg.WarmupSteps, cfg.BaseLR, cfg.MinLR, 1000)
}

// cosineLR computes warmup then cosine decay.
func cosineLR(step, warmup int, base, min float32, total int) float32 {
	if step < warmup {
		return base * float32(step+1) / float32(warmup)
	}
	progress := float64(step-warmup) / float64(maxInt(1, total-warmup))
	if progress > 1 {
		progress = 1
	}
	cos := 0.5 * (1 + math.Cos(math.Pi*progress))
	return min + (base-min)*float32(cos)
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// Train runs the full deterministic training loop. It logs metrics directly to
// stdout (no supervisor, no telemetry daemon) and returns the final loss.
func (t *Trainer) Train(ds *Dataset) (float32, error) {
	if err := ds.Bounds(); err != nil {
		return 0, err
	}
	cfg := t.Config
	if cfg.Epochs <= 0 {
		cfg.Epochs = 1
	}
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = 1
	}
	if cfg.LogEvery <= 0 {
		cfg.LogEvery = 1
	}
	if cfg.StagnantEpochLimit <= 0 {
		cfg.StagnantEpochLimit = 100
	}
	t.Config = cfg

	n := len(ds.Samples)
	totalSteps := cfg.Epochs * ((n + cfg.BatchSize - 1) / cfg.BatchSize)
	bestEpochAvg := float32(math.Inf(1))
	stagnantEpochs := 0

	fmt.Printf("[train] samples=%d features=%d classes=%d epochs=%d batch=%d total_steps=%d\n",
		n, ds.FeatureSize(), ds.NumClasses(), cfg.Epochs, cfg.BatchSize, totalSteps)

	var lastLoss float32
	for epoch := 0; epoch < cfg.Epochs; epoch++ {
		order := permute(n, int64(epoch)+ds.Seed)
		var epochLossSum float32
		var epochBatches int

		for start := 0; start < n; start += cfg.BatchSize {
			end := start + cfg.BatchSize
			if end > n {
				end = n
			}
			batch := order[start:end]

			inputs := make([][]float32, len(batch))
			targets := make([]int, len(batch))
			for i, idx := range batch {
				inputs[i] = ds.Samples[idx].Input
				targets[i] = ds.Samples[idx].Label
			}

			logits, acts, pre, err := t.Model.Forward(inputs)
			if err != nil {
				return 0, err
			}
			loss := CrossEntropy(logits, targets)
			if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
				return 0, fmt.Errorf("%w at step %d", ErrNaN, t.Step)
			}
			lastLoss = loss
			epochLossSum += loss
			epochBatches++

			grads, err := t.Model.Backward(inputs, targets, logits, acts, pre)
			if err != nil {
				return 0, err
			}
			lr := cosineLR(t.Step, cfg.WarmupSteps, cfg.BaseLR, cfg.MinLR, totalSteps)
			t.Model.Update(grads, lr)
			t.Step++

			if t.Step%cfg.LogEvery == 0 {
				fmt.Printf("[train] step=%d epoch=%d loss=%.6f lr=%.6f\n", t.Step, epoch, loss, lr)
			}
		}

		epochAvg := epochLossSum / float32(epochBatches)
		if epochAvg < bestEpochAvg {
			bestEpochAvg = epochAvg
			stagnantEpochs = 0
		} else {
			stagnantEpochs++
			if stagnantEpochs > cfg.StagnantEpochLimit {
				return 0, fmt.Errorf(
					"hard assertion failed: epoch-average loss not improving for %d epochs (best=%.6f current=%.6f)",
					stagnantEpochs, bestEpochAvg, epochAvg)
			}
		}
	}

	pred := t.Model.Predict(inputsForAll(ds))
	acc := Accuracy(pred, labelsForAll(ds))
	fmt.Printf("[train] done loss=%.6f train_acc=%.4f\n", lastLoss, acc)
	return lastLoss, nil
}

func inputsForAll(ds *Dataset) [][]float32 {
	out := make([][]float32, len(ds.Samples))
	for i, s := range ds.Samples {
		out[i] = s.Input
	}
	return out
}

func labelsForAll(ds *Dataset) []int {
	out := make([]int, len(ds.Samples))
	for i, s := range ds.Samples {
		out[i] = s.Label
	}
	return out
}

// permute returns a deterministic permutation of [0,n) using a simple LCG.
func permute(n int, seed int64) []int {
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	state := uint64(seed)
	for i := n - 1; i > 0; i-- {
		state = state*6364136223846793005 + 1442695040888963407
		j := int(state % uint64(i+1))
		idx[i], idx[j] = idx[j], idx[i]
	}
	return idx
}

// SaveWeights writes the flattened parameters to a file (plain text, one
// float per line) for inspection or checkpointing.
func (t *Trainer) SaveWeights(path string) error {
	params := t.Model.Parameters()
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	for _, v := range params {
		if _, err := fmt.Fprintf(f, "%.9f\n", v); err != nil {
			return err
		}
	}
	return nil
}

// LoadWeights restores a flattened parameter vector written by SaveWeights.
func (m *DenseModel) LoadWeights(path string) error {
	bs, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	fields := strings.Fields(string(bs))

	total := 0
	for _, l := range m.Layers {
		total += len(l.Weights) + len(l.Bias)
	}
	if len(fields) != total {
		return fmt.Errorf("weight count mismatch: file has %d, model needs %d", len(fields), total)
	}

	idx := 0
	next := func() (float32, error) {
		v, err := strconv.ParseFloat(fields[idx], 32)
		if err != nil {
			return 0, fmt.Errorf("invalid weight at index %d: %w", idx, err)
		}
		idx++
		return float32(v), nil
	}

	for _, l := range m.Layers {
		for i := range l.Weights {
			v, err := next()
			if err != nil {
				return err
			}
			l.Weights[i] = v
		}
		for i := range l.Bias {
			v, err := next()
			if err != nil {
				return err
			}
			l.Bias[i] = v
		}
	}
	return nil
}
