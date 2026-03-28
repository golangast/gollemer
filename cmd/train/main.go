package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/golangast/gollemer/internal/ai/moe"
)

func main() {
	inputDim := flag.Int("dim", 128, "Input dimension")
	numExperts := flag.Int("experts", 8, "Number of experts")
	topK := flag.Int("k", 2, "Number of experts to activate")
	lr := flag.Float64("lr", 0.01, "Learning rate")
	flag.Parse()

	log.Printf("Initializing MoE Model [Dim=%d, Experts=%d, K=%d]", *inputDim, *numExperts, *topK)
	
	// Create Model
	gater := moe.NewSparseGater(*inputDim, *numExperts, *topK)
	experts := make([]*moe.SparseExpert, *numExperts)
	for i := 0; i < *numExperts; i++ {
		experts[i] = moe.NewSparseExpert(i, *inputDim, *inputDim)
	}
	model := &moe.SparseModel{Gater: gater, Experts: experts}

	// Setup Signal Handling for Graceful Shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	stopTraining := false
	go func() {
		<-sigChan
		fmt.Println("\n[!] Interrupt received. Finishing current batch and saving...")
		stopTraining = true
	}()

	// Training Loop (Sample Data)
	log.Println("--- 🚀 Starting Gollemer Training ---")
	for epoch := 0; !stopTraining; epoch++ {
		// Sample data for demonstration
		input := make([]float32, *inputDim)
		target := make([]float32, *inputDim)
		for i := range input {
			input[i] = rand.Float32()
			target[i] = input[i] * 0.5 // Linear task: halving
		}

		// 1. Forward Pass
		prediction, indices := model.Predict(input)

		// 2. Training Step (Manual Backprop for MoE)
		errors := make([]float32, *inputDim)
		totalLoss := float32(0)
		for i := range target {
			errors[i] = prediction[i] - target[i]
			totalLoss += errors[i] * errors[i]
		}
		avgLoss := totalLoss / float32(*inputDim)

		// 3. Update only active experts (Strict Sparsity)
		for _, idx := range indices {
			model.Experts[idx].UpdateWeights(input, errors, float32(*lr))
		}

		// 4. Update Gater (Simplified Importance Update)
		gaterGrad := make([]float32, *numExperts)
		for _, idx := range indices {
			// Proxy: Expert ID i receives gradient proportional to its slot's error mean
			for _, e := range errors {
				gaterGrad[idx] += e
			}
			gaterGrad[idx] /= float32(*inputDim)
		}
		model.Gater.UpdateGaterWeights(input, gaterGrad, float32(*lr))

		if epoch%100 == 0 {
			fmt.Printf("Epoch %d | Loss: %.6f | Active Experts: %v\r", epoch, avgLoss, indices)
		}

		// 🗃️ Periodic Checkpointing
		if epoch > 0 && epoch%5000 == 0 {
			err := model.SaveCheckpoint(epoch, avgLoss)
			if err != nil {
				log.Printf("Failed to save checkpoint: %v", err)
			}
		}

		if stopTraining {
			break
		}
		
		// Wait a tiny bit to not overwhelm CPU if sample is too simple
		time.Sleep(1 * time.Millisecond)
	}

	// Final Save
	fmt.Println("\n[✓] Finalizing weights and exiting.")
	model.SaveCheckpoint(-1, 0)
}
