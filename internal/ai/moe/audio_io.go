package moe

import (
	"encoding/json"
	"os"
)

type SavedAudioModel struct {
	AudioWeights []float32
	GruWz        []float32
	GruUz        []float32
	GruBz        []float32
	GruWr        []float32
	GruUr        []float32
	GruBr        []float32
	GruWn        []float32
	GruUn        []float32
	GruBn        []float32
	HeadW        []float32
	HeadB        []float32
	ClassNames   []string
	// Prototypes stores the mean GRU embedding for each class.
	// This enables zero-shot command addition without retraining.
	Prototypes map[string][]float32
}

func SaveAudioModel(path string, ae *AudioEncoder, te *TemporalEncoder, headW, headB []float32, classNames []string, prototypes map[string][]float32) error {
	m := SavedAudioModel{
		AudioWeights: ae.Weights,
		GruWz:        te.Wz,
		GruUz:        te.Uz,
		GruBz:        te.Bz,
		GruWr:        te.Wr,
		GruUr:        te.Ur,
		GruBr:        te.Br,
		GruWn:        te.Wn,
		GruUn:        te.Un,
		GruBn:        te.Bn,
		HeadW:        headW,
		HeadB:        headB,
		ClassNames:   classNames,
		Prototypes:   prototypes,
	}
	b, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0644)
}

func LoadAudioModel(path string) (*AudioEncoder, *TemporalEncoder, []float32, []float32, []string, map[string][]float32, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, nil, nil, nil, nil, err
	}
	var m SavedAudioModel
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, nil, nil, nil, nil, nil, err
	}

	// AudioEncoder: InputDim=400, DModel=64
	// TemporalEncoder: InputDim=64, HiddenDim=32
	// We'll just hardcode dimensions for this example since they're fixed in the pipeline
	ae := NewAudioEncoder(400, 64)
	copy(ae.Weights, m.AudioWeights)

	te := NewTemporalEncoder(64, 32)
	copy(te.Wz, m.GruWz)
	copy(te.Uz, m.GruUz)
	copy(te.Bz, m.GruBz)
	copy(te.Wr, m.GruWr)
	copy(te.Ur, m.GruUr)
	copy(te.Br, m.GruBr)
	copy(te.Wn, m.GruWn)
	copy(te.Un, m.GruUn)
	copy(te.Bn, m.GruBn)

	return ae, te, m.HeadW, m.HeadB, m.ClassNames, m.Prototypes, nil
}
