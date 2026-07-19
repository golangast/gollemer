package moe

import (
	"encoding/json"
	"os"
)

type SocialConfig struct {
	NumExperts                 int                `json:"num_experts"`
	ModelDim                   int                `json:"model_dim"`
	Epochs                     int                `json:"epochs"`
	LearningRate               float32            `json:"learning_rate"`
	BatchSize                  int                `json:"batch_size"`
	ContextMultiplier          float32            `json:"context_multiplier"`
	RouterNoise                float32            `json:"router_noise"`
	RouterTemperature          float32            `json:"router_temperature"`
	LoadBalancingWeight        float32            `json:"load_balancing_weight"`
	ExpertDropout              float32            `json:"expert_dropout"`
	CollapseThreshold          float32            `json:"collapse_threshold"`
	LabelSmoothing             float32            `json:"label_smoothing"`
	AccumulateSteps            int                `json:"accumulate_steps"`
	WeightDecay                float32            `json:"weight_decay"`
	MaxGradNorm                float32            `json:"max_grad_norm"`
	AutoHeal                   bool               `json:"auto_heal"`
	OverfitMode                bool               `json:"overfit_mode"`
	SamplingStart              int                `json:"sampling_start_epoch"`
	SamplingMax                float32            `json:"sampling_max_prob"`
	VerboseThinking            bool               `json:"verbose_thinking"`
	CapacityFactor             float32            `json:"capacity_factor"`
	K                          int                `json:"k"`
	RepetitionPenalty          float32            `json:"repetition_penalty"`
	FrequencyPenalty           float32            `json:"frequency_penalty"`
	PresencePenalty            float32            `json:"presence_penalty"`
	EntropyWeight              float32            `json:"entropy_weight"`
	UnkPenalty                 float32            `json:"unk_penalty"`
	StructuralBiasIntensity    float32            `json:"structural_bias_intensity"`
	StructuralRoutingWeight    float32            `json:"structural_routing_weight"`
	ContextMultiplierDecay     float32            `json:"context_multiplier_decay"`
	ExpertRegularizationWeight float32            `json:"expert_regularization_weight"`
	ExpertSparsityWeight       float32            `json:"expert_sparsity_weight"`
	TokenWeights               map[string]float32 `json:"token_weights"`
	IntentBias                 float32            `json:"intent_bias"`
	TopP                       float32            `json:"top_p"`
	TopK                       int                `json:"top_k"`
	AutoTestSave               bool               `json:"auto_test_save"`
	TriggerTest                bool               `json:"trigger_test"`
	TriggerSave                bool               `json:"trigger_save"`
}

func LoadSocialConfig(path string) SocialConfig {
	defaultConfig := SocialConfig{
		NumExperts:                 16, // 8 GoffiExperts + 8 GrammarExperts (POS-role specialists)
		ModelDim:                   256,
		Epochs:                     1000,
		LearningRate:               1e-3,
		BatchSize:                  100,
		ContextMultiplier:          15.0,
		RouterNoise:                1.2,
		RouterTemperature:          1.0,
		CapacityFactor:             1.25,
		LoadBalancingWeight:        0.05,
		ExpertDropout:              0.3,
		CollapseThreshold:          0.4,
		LabelSmoothing:             0.1,
		AccumulateSteps:            4,
		WeightDecay:                1e-4,
		MaxGradNorm:                1.0,
		AutoHeal:                   true,
		OverfitMode:                false,
		SamplingStart:              5,
		SamplingMax:                0.5,
		VerboseThinking:            true,
		K:                          1,
		RepetitionPenalty:          1.2,
		FrequencyPenalty:           0.1,
		PresencePenalty:            0.1,
		ContextMultiplierDecay:     0.98,
		StructuralRoutingWeight:    1.5,
		StructuralBiasIntensity:    0.5,
		ExpertRegularizationWeight: 0.0001,
		ExpertSparsityWeight:       0.01,
		IntentBias:                 4.5,
		TopP:                       0.85,
		TopK:                       5,
		AutoTestSave:               true,
		TokenWeights:               make(map[string]float32),
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
