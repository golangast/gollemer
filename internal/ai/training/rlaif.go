// Package training implements Compiler-Driven Reinforcement Learning from AI Feedback
// (RLAIF) for Gollemer. This is the critical step that hooks the inference loop
// directly into Go's toolchain:
//
//  1. Gollemer generates a SEARCH/REPLACE patch
//  2. Patch is applied to an AST in-memory
//  3. go/parser, go/vet, go/build verify correctness
//  4. Compilation success → Reward signal (+1.0), save as valid pair
//  5. Compilation failure → Penalty signal (-1.0), feed error log back to LLM
//
// This creates a self-improving loop where the model learns from compiler feedback.
package training

import (
	"encoding/json"
	"fmt"
	"go/parser"
	"go/token"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// RLAIFConfig holds configuration for the RLAIF training loop.
type RLAIFConfig struct {
	MaxIterations     int     `json:"max_iterations"`      // Number of RL iterations
	SamplesPerPrompt  int     `json:"samples_per_prompt"`  // Patches to sample per prompt
	RewardSuccess     float64 `json:"reward_success"`      // +1.0 for compilation pass
	RewardFailure     float64 `json:"reward_failure"`      // -1.0 for compilation fail
	RewardVetWarn     float64 `json:"reward_vet_warn"`     // -0.5 for vet warnings
	GoModPath         string  `json:"go_mod_path"`         // Path to go.mod for module context
	TempDir           string  `json:"temp_dir"`            // Temp directory for sandbox
	KeepFailedPatches bool    `json:"keep_failed_patches"` // Save failed patches for analysis
	MaxRetries        int     `json:"max_retries"`         // Error ingestion retries
}

// DefaultRLAIFConfig returns sensible defaults for RLAIF training.
func DefaultRLAIFConfig() RLAIFConfig {
	return RLAIFConfig{
		MaxIterations:     1000,
		SamplesPerPrompt:  5,
		RewardSuccess:     1.0,
		RewardFailure:     -1.0,
		RewardVetWarn:     -0.5,
		GoModPath:         ".",
		TempDir:           "/tmp/gollemer_rlaif",
		KeepFailedPatches: true,
		MaxRetries:        3,
	}
}

// PatchOutcome captures the result of applying and verifying a patch.
type PatchOutcome struct {
	Patch          string  `json:"patch"`
	Reward         float64 `json:"reward"`
	ValidGo        bool    `json:"valid_go"`
	VetPassed      bool    `json:"vet_passed"`
	BuildPassed    bool    `json:"build_passed"`
	CompilerErrors string  `json:"compiler_errors,omitempty"`
	Duration       string  `json:"duration"`
}

// RLAIFTrainer orchestrates compiler-driven RL training.
type RLAIFTrainer struct {
	config   RLAIFConfig
	model    RLAIFTrainableModel
	outcomes []PatchOutcome
	mu       sync.Mutex
}

// RLAIFTrainableModel is the interface the MoE model must implement for RLAIF.
type RLAIFTrainableModel interface {
	// GeneratePatch generates a SEARCH/REPLACE patch for the given instruction and context.
	GeneratePatch(instruction, beforeCode string) (patch string, err error)

	// UpdateFromReward updates model weights based on the reward signal.
	UpdateFromReward(outcome PatchOutcome) error

	// SaveCheckpoint saves the current model state.
	SaveCheckpoint(path string) error
}

// NewRLAIFTrainer creates a new RLAIF trainer.
func NewRLAIFTrainer(config RLAIFConfig, model RLAIFTrainableModel) *RLAIFTrainer {
	return &RLAIFTrainer{
		config:   config,
		model:    model,
		outcomes: make([]PatchOutcome, 0),
	}
}

// RunTrainingLoop runs the main RLAIF training loop.
// It takes training examples, generates patches, verifies them with Go toolchain,
// and updates the model based on compiler reward signals.
func (r *RLAIFTrainer) RunTrainingLoop(examples []RLTrainingExample) error {
	if len(examples) == 0 {
		return fmt.Errorf("no training examples provided")
	}

	// Create temp directory for sandbox
	if err := os.MkdirAll(r.config.TempDir, 0755); err != nil {
		return fmt.Errorf("create temp dir: %w", err)
	}

	log.Printf("Starting RLAIF training loop:")
	log.Printf("  Examples: %d", len(examples))
	log.Printf("  Max iterations: %d", r.config.MaxIterations)
	log.Printf("  Samples per prompt: %d", r.config.SamplesPerPrompt)
	log.Printf("  Reward scheme: +%.1f pass, %.1f fail, %.1f vet warn",
		r.config.RewardSuccess, r.config.RewardFailure, r.config.RewardVetWarn)

	totalReward := 0.0
	successCount := 0
	failureCount := 0

	for iter := 0; iter < r.config.MaxIterations; iter++ {
		// Pick a random example
		ex := examples[iter%len(examples)]

		// Generate multiple patch samples
		for sample := 0; sample < r.config.SamplesPerPrompt; sample++ {
			start := time.Now()

			// Generate patch
			patch, err := r.model.GeneratePatch(ex.Instruction, ex.BeforeCode)
			if err != nil {
				log.Printf("  Sample %d: patch generation failed: %v", sample, err)
				continue
			}

			// Verify patch against Go toolchain
			outcome := r.verifyPatch(patch, ex.BeforeCode, ex.FilePath)
			outcome.Duration = time.Since(start).String()
			outcome.Patch = patch

			// Store outcome
			r.mu.Lock()
			r.outcomes = append(r.outcomes, outcome)
			r.mu.Unlock()

			// Update model with reward
			if err := r.model.UpdateFromReward(outcome); err != nil {
				log.Printf("  Model update failed: %v", err)
			}

			totalReward += outcome.Reward
			if outcome.Reward > 0 {
				successCount++
			} else {
				failureCount++
			}

			// Error ingestion: if compilation failed, generate corrective patch
			if !outcome.ValidGo || !outcome.BuildPassed {
				if r.config.MaxRetries > 0 {
					r.errorIngestion(ex, outcome)
				}
			}

			// Save failed patches for analysis
			if !outcome.ValidGo && r.config.KeepFailedPatches {
				r.saveFailedPatch(iter, sample, patch, outcome.CompilerErrors)
			}
		}

		// Log progress
		if (iter+1)%10 == 0 {
			avgReward := totalReward / float64(max(1, (iter+1)*r.config.SamplesPerPrompt))
			log.Printf("Iteration %d/%d: avg_reward=%.4f success=%d failure=%d",
				iter+1, r.config.MaxIterations, avgReward, successCount, failureCount)
		}

		// Save checkpoint periodically
		if (iter+1)%100 == 0 {
			r.saveCheckpoint(fmt.Sprintf("rlaif_iter_%d", iter+1))
		}
	}

	log.Printf("RLAIF training complete!")
	log.Printf("  Total reward: %.2f", totalReward)
	log.Printf("  Success rate: %.1f%%", float64(successCount)/float64(max(1, successCount+failureCount))*100)

	// Save final checkpoint
	r.saveCheckpoint("rlaif_final")

	return nil
}

// verifyPatch applies the patch and runs Go toolchain verification.
func (r *RLAIFTrainer) verifyPatch(patch, beforeCode, filePath string) PatchOutcome {
	outcome := PatchOutcome{
		Reward:  r.config.RewardFailure, // Default: failure
		ValidGo: true,
	}

	// Step 1: Verify the before code parses
	fset := token.NewFileSet()
	if _, err := parser.ParseFile(fset, "", beforeCode, parser.AllErrors); err != nil {
		outcome.ValidGo = false
		outcome.CompilerErrors = fmt.Sprintf("before code parse error: %v", err)
		return outcome
	}

	// Step 2: Extract the after code from the patch
	afterCode := extractAfterFromPatch(patch)
	if afterCode == "" {
		outcome.ValidGo = false
		outcome.CompilerErrors = "could not extract REPLACE section from patch"
		return outcome
	}

	// Step 3: Verify the after code parses
	fset = token.NewFileSet()
	if _, err := parser.ParseFile(fset, "", afterCode, parser.AllErrors); err != nil {
		outcome.ValidGo = false
		outcome.CompilerErrors = fmt.Sprintf("after code parse error: %v", err)
		return outcome
	}
	outcome.ValidGo = true

	// Step 4: Write to temp file and run go vet
	if filePath == "" {
		filePath = "patch_output.go"
	}
	tempFile := filepath.Join(r.config.TempDir, filepath.Base(filePath))

	if err := os.WriteFile(tempFile, []byte(afterCode), 0644); err != nil {
		outcome.CompilerErrors = fmt.Sprintf("write temp file: %v", err)
		return outcome
	}

	// Run go vet
	cmd := exec.Command("go", "vet", tempFile)
	if output, err := cmd.CombinedOutput(); err != nil {
		outcome.VetPassed = false
		outcome.CompilerErrors = string(output)
		outcome.Reward = r.config.RewardVetWarn
		return outcome
	}
	outcome.VetPassed = true

	// Step 5: Try go build
	// Create a temporary module if needed
	modDir := filepath.Dir(tempFile)
	if _, err := os.Stat(filepath.Join(modDir, "go.mod")); os.IsNotExist(err) {
		initCmd := exec.Command("go", "mod", "init", "gollemer_rlaif_sandbox")
		initCmd.Dir = modDir
		if err := initCmd.Run(); err != nil {
			log.Printf("Warning: go mod init failed: %v", err)
		}
	}

	buildCmd := exec.Command("go", "build", "-o", "/dev/null", tempFile)
	buildCmd.Dir = modDir
	if output, err := buildCmd.CombinedOutput(); err != nil {
		outcome.BuildPassed = false
		outcome.CompilerErrors = string(output)
		outcome.Reward = r.config.RewardFailure
		return outcome
	}
	outcome.BuildPassed = true

	// All checks passed!
	outcome.Reward = r.config.RewardSuccess

	return outcome
}

// errorIngestion feeds compiler errors back to generate a corrected patch.
func (r *RLAIFTrainer) errorIngestion(ex RLTrainingExample, outcome PatchOutcome) {
	// Build error context
	errorContext := fmt.Sprintf(
		"Compiler Error:\n%s\n\n"+
			"Original instruction: %s\n"+
			"Generate a corrected patch that fixes the compilation error.",
		outcome.CompilerErrors,
		ex.Instruction,
	)

	// Generate corrective patch with error context
	for attempt := 0; attempt < r.config.MaxRetries; attempt++ {
		correctivePatch, err := r.model.GeneratePatch(errorContext, ex.BeforeCode)
		if err != nil {
			continue
		}

		// Verify the corrective patch
		correctiveOutcome := r.verifyPatch(correctivePatch, ex.BeforeCode, ex.FilePath)
		if correctiveOutcome.ValidGo && correctiveOutcome.BuildPassed {
			// The corrective patch succeeded - reward it
			correctiveOutcome.Reward = r.config.RewardSuccess * 0.8 // Slightly less than original
			_ = r.model.UpdateFromReward(correctiveOutcome)

			r.mu.Lock()
			r.outcomes = append(r.outcomes, correctiveOutcome)
			r.mu.Unlock()

			log.Printf("  Error ingestion: corrected patch found (attempt %d)", attempt+1)
			return
		}
	}
}

// saveFailedPatch stores a failed patch for later analysis.
func (r *RLAIFTrainer) saveFailedPatch(iter, sample int, patch, errors string) {
	failedDir := filepath.Join(r.config.TempDir, "failed_patches")
	if err := os.MkdirAll(failedDir, 0755); err != nil {
		return
	}

	record := struct {
		Iteration int    `json:"iteration"`
		Sample    int    `json:"sample"`
		Patch     string `json:"patch"`
		Errors    string `json:"errors"`
	}{
		Iteration: iter,
		Sample:    sample,
		Patch:     patch,
		Errors:    errors,
	}

	data, err := json.MarshalIndent(record, "", "  ")
	if err != nil {
		return
	}

	path := filepath.Join(failedDir, fmt.Sprintf("failed_%d_%d.json", iter, sample))
	_ = os.WriteFile(path, data, 0644)
}

// saveCheckpoint saves model state.
func (r *RLAIFTrainer) saveCheckpoint(name string) {
	dir := filepath.Join(r.config.TempDir, "checkpoints")
	if err := os.MkdirAll(dir, 0755); err != nil {
		log.Printf("Warning: cannot create checkpoint dir: %v", err)
		return
	}
	path := filepath.Join(dir, fmt.Sprintf("%s.bin", name))
	if err := r.model.SaveCheckpoint(path); err != nil {
		log.Printf("Warning: save checkpoint failed: %v", err)
	}
	log.Printf("Checkpoint saved: %s", path)
}

// GetStats returns training statistics.
func (r *RLAIFTrainer) GetStats() RLAIFStats {
	r.mu.Lock()
	defer r.mu.Unlock()

	stats := RLAIFStats{
		TotalPatches: len(r.outcomes),
		Outcomes:     r.outcomes,
	}

	for _, o := range r.outcomes {
		if o.ValidGo {
			stats.ValidPatches++
		}
		if o.VetPassed {
			stats.VetPassed++
		}
		if o.BuildPassed {
			stats.BuildPassed++
		}
		if o.Reward > 0 {
			stats.SuccessfulPatches++
		}
	}

	return stats
}

// RLAIFStats tracks RLAIF training statistics.
type RLAIFStats struct {
	TotalPatches      int            `json:"total_patches"`
	ValidPatches      int            `json:"valid_patches"`
	VetPassed         int            `json:"vet_passed"`
	BuildPassed       int            `json:"build_passed"`
	SuccessfulPatches int            `json:"successful_patches"`
	Outcomes          []PatchOutcome `json:"outcomes,omitempty"`
}

// RLTrainingExample is a single training example for RLAIF.
type RLTrainingExample struct {
	Instruction string `json:"instruction"`
	BeforeCode  string `json:"before_code"`
	FilePath    string `json:"file_path,omitempty"`
	TargetPatch string `json:"target_patch,omitempty"`
}

// extractAfterFromPatch extracts the REPLACE section from a SEARCH/REPLACE patch.
func extractAfterFromPatch(patch string) string {
	// Format: <<<<<<< SEARCH\n...\n=======\n...\n>>>>>>> REPLACE
	parts := strings.Split(patch, "=======\n")
	if len(parts) != 2 {
		return ""
	}
	after := strings.TrimSuffix(parts[1], "\n>>>>>>> REPLACE")
	after = strings.TrimSuffix(after, ">>>>>>> REPLACE")
	return after
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
