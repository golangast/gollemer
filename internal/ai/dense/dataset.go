package dense

import "strings"

// Sample is a plain, strongly-typed training example. No JSON, no CSV, no
// pipeline — just an explicit Go struct pair (Input -> Label).
//
// Input is a precomputed feature vector (e.g. bag-of-words counts or a tiny
// ASCII/character encoding). Label is the target class index.
type Sample struct {
	Input []float32
	Label int
}

// Dataset is an ordered slice of samples. Training walks it sequentially with
// a fixed, deterministic permutation so results are reproducible.
type Dataset struct {
	Samples []Sample
	Seed    int64
}

// NewDataset builds a dataset from already-encoded samples.
func NewDataset(seed int64, samples ...Sample) *Dataset {
	return &Dataset{Samples: samples, Seed: seed}
}

// FeatureSize returns the input vector dimension (or 0 when empty).
func (d *Dataset) FeatureSize() int {
	if len(d.Samples) == 0 {
		return 0
	}
	return len(d.Samples[0].Input)
}

// NumClasses returns the number of distinct labels.
func (d *Dataset) NumClasses() int {
	maxLabel := -1
	for _, s := range d.Samples {
		if s.Label > maxLabel {
			maxLabel = s.Label
		}
	}
	return maxLabel + 1
}

// Bounds guards against malformed samples.
func (d *Dataset) Bounds() error {
	if len(d.Samples) == 0 {
		return &errDim{index: -1, msg: "empty dataset"}
	}
	n := len(d.Samples[0].Input)
	for i, s := range d.Samples {
		if len(s.Input) != n {
			return &errDim{index: i, got: len(s.Input), want: n}
		}
		if s.Label < 0 {
			return &errDim{index: i, msg: "label must be >= 0"}
		}
	}
	return nil
}

type errDim struct {
	index int
	got   int
	want  int
	msg   string
}

func (e *errDim) Error() string {
	if e.msg != "" {
		return "dataset: " + e.msg + " at sample " + itoa(e.index)
	}
	return "dataset: inconsistent feature size at sample " + itoa(e.index) +
		" (got " + itoa(e.got) + ", want " + itoa(e.want) + ")"
}

func itoa(v int) string {
	if v == 0 {
		return "0"
	}
	neg := v < 0
	if neg {
		v = -v
	}
	var buf [20]byte
	i := len(buf)
	for v > 0 {
		i--
		buf[i] = byte('0' + v%10)
		v /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}

// ─── Encoders ────────────────────────────────────────────────────────────────

// CharEncode converts a fixed-alphabet token into a one-hot feature vector.
// Alphabet is ordered; unknown characters map to the final "UNK" slot.
func CharEncode(token string, alphabet string) []float32 {
	vec := make([]float32, len(alphabet)+1)
	for _, r := range token {
		idx := -1
		for i, c := range alphabet {
			if c == r {
				idx = i
				break
			}
		}
		if idx < 0 {
			idx = len(alphabet) // UNK
		}
		vec[idx] = 1
	}
	return vec
}

// BagOfWords encodes a short natural-language prompt as a binary presence
// vector over a fixed vocabulary. Words are lower-cased and split on
// whitespace; each vocabulary word present in the prompt sets its slot to 1.
// This is a simple, deterministic, first-principles feature encoder.
func BagOfWords(prompt string, vocab []string) []float32 {
	vec := make([]float32, len(vocab))
	for _, w := range strings.Fields(strings.ToLower(prompt)) {
		for i, v := range vocab {
			if w == v {
				vec[i] = 1
				break
			}
		}
	}
	return vec
}
