package moe

import (
	"encoding/json"
	"os"
)

type SocialConfig struct {
	NumExperts          int     `json:"num_experts"`
	ModelDim            int     `json:"model_dim"`
	Epochs              int     `json:"epochs"`
	LearningRate        float32 `json:"learning_rate"`
	BatchSize           int     `json:"batch_size"`
	ContextMultiplier   float32 `json:"context_multiplier"`
	RouterNoise         float32 `json:"router_noise"`
	RouterTemperature   float32 `json:"router_temperature"`
	LoadBalancingWeight float32 `json:"load_balancing_weight"`
	ExpertDropout       float32 `json:"expert_dropout"`
	CollapseThreshold   float32 `json:"collapse_threshold"`
	LabelSmoothing      float32 `json:"label_smoothing"`
	AccumulateSteps     int     `json:"accumulate_steps"`
	WeightDecay         float32 `json:"weight_decay"`
	MaxGradNorm         float32 `json:"max_grad_norm"`
	AutoHeal            bool    `json:"auto_heal"`
	OverfitMode         bool    `json:"overfit_mode"`
	SamplingStart       int     `json:"sampling_start_epoch"`
	SamplingMax         float32 `json:"sampling_max_prob"`
	VerboseThinking     bool    `json:"verbose_thinking"`
	CapacityFactor      float32 `json:"capacity_factor"`
	K                   int     `json:"k"`
	RepetitionPenalty   float32 `json:"repetition_penalty"`
	EntropyWeight       float32 `json:"entropy_weight"`
}

func LoadSocialConfig(path string) SocialConfig {
	defaultConfig := SocialConfig{
		NumExperts:          16, // 8 GoffiExperts + 8 GrammarExperts (POS-role specialists)
		ModelDim:            256,
		Epochs:              1000,
		LearningRate:        1e-3,
		BatchSize:           1,
		ContextMultiplier:   15.0,
		RouterNoise:         1.2,
		RouterTemperature:   1.0,
		CapacityFactor:      1.25,
		LoadBalancingWeight: 0.05,
		ExpertDropout:       0.3,
		CollapseThreshold:   0.4,
		LabelSmoothing:      0.1,
		AccumulateSteps:     4,
		WeightDecay:         1e-4,
		MaxGradNorm:         1.0,
		AutoHeal:            true,
		OverfitMode:         false,
		SamplingStart:       5,
		SamplingMax:         0.5,
		VerboseThinking:     true,
		K:                   1,
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return defaultConfig
	}

	var config SocialConfig
	if err := json.Unmarshal(data, &config); err != nil {
		return defaultConfig
	}
	return config
}

func SaveSocialConfig(path string, config SocialConfig) error {
	data, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}
