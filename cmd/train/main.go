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
	batchSize := flag.Int("batch", 1, "Batch size for training")
	flag.Parse()

	log.Printf("Initializing MoE Model [Dim=%d, Experts=%d, K=%d, Batch=%d]", *inputDim, *numExperts, *topK, *batchSize)

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
	startTime := time.Now()
	for epoch := 0; !stopTraining; epoch++ {
		// Process batch of samples
		var batchLoss float32 = 0
		batchExpertGrads := make(map[int]float32)

		for b := 0; b < *batchSize; b++ {
			// Sample data for demonstration
			input := make([]float32, *inputDim)
			target := make([]float32, *inputDim)
			for i := range input {
				input[i] = rand.Float32()
				target[i] = input[i] * 0.5 // Linear task: halving
			}

			// 1. Forward Pass
			prediction, indices := model.Predict(input)

			// 2. Compute error
			errors := make([]float32, *inputDim)
			sampleLoss := float32(0)
			for i := range target {
				errors[i] = prediction[i] - target[i]
				sampleLoss += errors[i] * errors[i]
			}
			batchLoss += sampleLoss / float32(*inputDim)

			// 3. Accumulate gradients from active experts
			for _, idx := range indices {
				model.Experts[idx].UpdateWeights(input, errors, float32(*lr))

				// Track gradient for gater
				for _, e := range errors {
					batchExpertGrads[idx] += e
				}
			}
		}

		// Average loss across batch
		avgLoss := batchLoss / float32(*batchSize)

		// 4. Update Gater with accumulated gradients
		gaterGrad := make([]float32, *numExperts)
		for idx, grad := range batchExpertGrads {
			gaterGrad[idx] = grad / float32(*inputDim**batchSize)
		}
		// Create dummy input for gater update (first sample of batch)
		dummyInput := make([]float32, *inputDim)
		for i := range dummyInput {
			dummyInput[i] = rand.Float32()
		}
		model.Gater.UpdateGaterWeights(dummyInput, gaterGrad, float32(*lr))

		if epoch%100 == 0 {
			elapsed := time.Since(startTime).Seconds()
			throughput := float64((epoch+1)*(*batchSize)) / elapsed
			fmt.Printf("Epoch %d | Loss: %.6f | Batch Size: %d | Throughput: %.0f samples/sec\r", epoch, avgLoss, *batchSize, throughput)
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
