package orchestrator

import (
	"encoding/json"
	"log"
	"os"
	"sync"
	"time"
)

// PhaseConfig holds per-phase hyperparameters for the multi-phase curriculum.
type PhaseConfig struct {
	LearningRate            float32 `json:"learning_rate"`
	LoadBalancingWeight     float32 `json:"load_balancing_weight"`
	RouterTemperature       float32 `json:"router_temperature"`
	ExpertDropout           float32 `json:"expert_dropout"`
	BatchSize               int     `json:"batch_size"`
	MaxSeqLen               int     `json:"max_seq_len"`
	ForceSingleExpertEpochs int     `json:"force_single_expert_epochs"`
	FreezeExpertsStart      int     `json:"freeze_experts_start"`
	FreezeExpertsEnd        int     `json:"freeze_experts_end"`
	Dataset                 string  `json:"dataset"`
}

type TrainingConfig struct {
	NumExperts                 int                     `json:"num_experts"`
	ModelDim                   int                     `json:"model_dim"`
	Epochs                     int                     `json:"epochs"`
	LearningRate               float32                 `json:"learning_rate"`
	BatchSize                  int                     `json:"batch_size"`
	ContextMultiplier          float32                 `json:"context_multiplier"`
	RouterNoise                float32                 `json:"router_noise"`
	RouterTemperature          float32                 `json:"router_temperature"`
	LoadBalancingWeight        float32                 `json:"load_balancing_weight"`
	ExpertDropout              float32                 `json:"expert_dropout"`
	CollapseThreshold          float32                 `json:"collapse_threshold"`
	LabelSmoothing             float32                 `json:"label_smoothing"`
	AccumulateSteps            int                     `json:"accumulate_steps"`
	WeightDecay                float32                 `json:"weight_decay"`
	MaxGradNorm                float32                 `json:"max_grad_norm"`
	AutoHeal                   bool                    `json:"auto_heal"`
	OverfitMode                bool                    `json:"overfit_mode"`
	SamplingStart              int                     `json:"sampling_start_epoch"`
	SamplingMax                float32                 `json:"sampling_max_prob"`
	VerboseThinking            bool                    `json:"verbose_thinking"`
	CapacityFactor             float32                 `json:"capacity_factor"`
	K                          int                     `json:"k"`
	RepetitionPenalty          float32                 `json:"repetition_penalty"`
	EntropyWeight              float32                 `json:"entropy_weight"`
	GatingEntropyWeight        float32                 `json:"gating_entropy_weight"`
	ContextMultiplierDecay     float32                 `json:"context_multiplier_decay"`
	StructuralRoutingWeight    float32                 `json:"structural_routing_weight"`
	StructuralBiasIntensity    float32                 `json:"structural_bias_intensity"`
	UnkPenalty                 float32                 `json:"unk_penalty"`
	ExpertRegularizationWeight float32                 `json:"expert_regularization_weight"`
	ExpertSparsityWeight       float32                 `json:"expert_sparsity_weight"`
	TokenWeights               map[string]float32      `json:"token_weights"`
	FrequencyPenalty           float32                 `json:"frequency_penalty"`
	PresencePenalty            float32                 `json:"presence_penalty"`
	IntentBias                 float32                 `json:"intent_bias"`
	TopP                       float32                 `json:"top_p"`
	TopK                       int                     `json:"top_k"`
	AutoTestSave               bool                    `json:"auto_test_save"`
	TriggerTest                bool                    `json:"trigger_test"`
	TriggerSave                bool                    `json:"trigger_save"`
	MaxSeqLen                  int                     `json:"max_seq_len"`
	ForceSingleExpertEpochs    int                     `json:"force_single_expert_epochs"`
	Phases                     map[string]*PhaseConfig `json:"phases"`
	// Automated LR step-down
	AutoLREnabled   bool    `json:"auto_lr_enabled"`
	AutoLRThreshold float32 `json:"auto_lr_threshold"`
	AutoLRFactor    float32 `json:"auto_lr_factor"`
	AutoLRMaxSteps  int     `json:"auto_lr_max_steps"`
	// Early stopping: halt training as soon as loss drops to this value (0 = disabled)
	TargetLoss float32 `json:"target_loss"`
}

type SafeConfig struct {
	sync.RWMutex
	Config         TrainingConfig
	lastReloadTime time.Time
}

func NewSafeConfig(path string) (*SafeConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var cfg TrainingConfig
	cfg.AutoTestSave = true
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, err
	}
	return &SafeConfig{Config: cfg}, nil
}

func (s *SafeConfig) Get() TrainingConfig {
	s.RLock()
	defer s.RUnlock()
	return s.Config
}

func (s *SafeConfig) Update(fn func(*TrainingConfig)) {
	s.Lock()
	defer s.Unlock()
	fn(&s.Config)
}

func (s *SafeConfig) WatchConfig(path string) error {
	go func() {
		var lastSize int64
		var lastModTime time.Time

		// Initial state
		if info, err := os.Stat(path); err == nil {
			lastSize = info.Size()
			lastModTime = info.ModTime()
		}

		ticker := time.NewTicker(1 * time.Second)
		defer ticker.Stop()

		for range ticker.C {
			info, err := os.Stat(path)
			if err != nil {
				continue
			}

			// Check for changes in size or modification time
			if info.Size() != lastSize || !info.ModTime().Equal(lastModTime) {
				lastSize = info.Size()
				lastModTime = info.ModTime()

				// Debounce: prevent multiple reloads within a short window
				s.Lock()
				lastReload := s.lastReloadTime
				s.lastReloadTime = time.Now()
				s.Unlock()

				if time.Since(lastReload) < 500*time.Millisecond {
					continue
				}

				// Small delay to ensure the file is completely written/closed
				time.Sleep(200 * time.Millisecond)

				data, err := os.ReadFile(path)
				if err != nil {
					// On some systems, the file might temporarily not exist during an atomic save
					time.Sleep(100 * time.Millisecond)
					data, err = os.ReadFile(path)
					if err != nil {
						log.Printf("⚠️  Failed to reload config: %v", err)
						continue
					}
				}

				var cfg TrainingConfig
				cfg.AutoTestSave = true
				if err := json.Unmarshal(data, &cfg); err != nil {
					log.Printf("⚠️  Failed to parse reloaded config: %v", err)
					continue
				}

				s.Lock()
				s.Config = cfg
				s.Unlock()
				// log.Printf("🚀 Training variables updated via hot-reload (from %s)", path)
			}
		}
	}()

	return nil
}
