package orchestrator

import (
	"encoding/json"
	"log"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/fsnotify/fsnotify"
)

type TrainingConfig struct {
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
	GatingEntropyWeight    float32 `json:"gating_entropy_weight"`
	ContextMultiplierDecay float32 `json:"context_multiplier_decay"`
	StructuralRoutingWeight float32 `json:"structural_routing_weight"`
	StructuralBiasIntensity float32 `json:"structural_bias_intensity"`
	UnkPenalty             float32 `json:"unk_penalty"`
	ExpertRegularizationWeight float32 `json:"expert_regularization_weight"`
	ExpertSparsityWeight       float32 `json:"expert_sparsity_weight"`
	TokenWeights           map[string]float32 `json:"token_weights"`
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
	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		return err
	}

	dir := filepath.Dir(path)
	filename := filepath.Base(path)

	go func() {
		defer watcher.Close()
		for {
			select {
			case event, ok := <-watcher.Events:
				if !ok {
					return
				}
				// Watch for Write or Rename/Create (atomic saves often involve renames)
				if (event.Op&fsnotify.Write == fsnotify.Write || event.Op&fsnotify.Create == fsnotify.Create) && 
				   filepath.Base(event.Name) == filename {
					// Debounce: prevent multiple reloads within a short window
					s.Lock()
					lastReload := s.lastReloadTime
					s.lastReloadTime = time.Now()
					s.Unlock()
					
					if time.Since(lastReload) < 500 * time.Millisecond {
						continue
					}
					
					// Small delay to ensure the file is completely written/closed
					time.Sleep(200 * time.Millisecond)
					
					data, err := os.ReadFile(path)
					if err != nil {
						// On some systems, the file might temporarily not exist during a move
						time.Sleep(100 * time.Millisecond)
						data, err = os.ReadFile(path)
						if err != nil {
							log.Printf("⚠️  Failed to reload config: %v", err)
							continue
						}
					}
					
					var cfg TrainingConfig
					if err := json.Unmarshal(data, &cfg); err != nil {
						log.Printf("⚠️  Failed to parse reloaded config: %v", err)
						continue
					}
					
					s.Lock()
					s.Config = cfg
					s.Unlock()
					log.Printf("🚀 Training variables updated via hot-reload (from %s)", path)
				}
			case err, ok := <-watcher.Errors:
				if !ok {
					return
				}
				log.Printf("⚠️  Config watcher error: %v", err)
			}
		}
	}()

	return watcher.Add(dir)
}
