package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"os/signal"
	"sync"
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
	prefetch := flag.Int("prefetch", 1, "GPU prefetch batches (for pipelining)")
	flag.Parse()

	log.Printf("Initializing MoE Model [Dim=%d, Experts=%d, K=%d, Batch=%d, Prefetch=%d]", *inputDim, *numExperts, *topK, *batchSize, *prefetch)

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

	// Training Loop with Goroutine Pipelining
	// Data preparation and GPU compute run in parallel via go channels
	log.Println("--- 🚀 Starting Gollemer Training (Goroutine-pipelined) ---")
	startTime := time.Now()

	// Channel for pipelined data (CPU prepares while GPU computes)
	type SampleBatch struct {
		inputs  [][]float32
		targets [][]float32
	}
	dataCh := make(chan *SampleBatch, *prefetch) // Buffer prefetch batches

	// Data generator goroutine (runs in parallel, generates batches continuously)
	go func() {
		defer close(dataCh)
		for {
			if stopTraining {
				return
			}

			batch := &SampleBatch{
				inputs:  make([][]float32, *batchSize),
				targets: make([][]float32, *batchSize),
			}

			// Pre-generate batch data (CPU task, parallel with GPU)
			for b := 0; b < *batchSize; b++ {
				batch.inputs[b] = make([]float32, *inputDim)
				batch.targets[b] = make([]float32, *inputDim)
				for i := range batch.inputs[b] {
					batch.inputs[b][i] = rand.Float32()
					batch.targets[b][i] = batch.inputs[b][i] * 0.5
				}
			}

			// Blocking send - wait for main training loop to consume batch
			dataCh <- batch
		}
	}()

	for epoch := 0; !stopTraining; epoch++ {
		batch, ok := <-dataCh
		if !ok || batch == nil {
			break
		}

		var batchLoss float32 = 0
		batchExpertGrads := make(map[int]float32)

		// Process pre-computed batch samples (GPU + expert dispatch)
		for b := 0; b < len(batch.inputs); b++ {
			input := batch.inputs[b]
			target := batch.targets[b]

			// 1. Forward Pass through MoE with expert routing
			prediction, indices := model.Predict(input)

			// 2. Compute error
			errors := make([]float32, *inputDim)
			sampleLoss := float32(0)
			for i := range target {
				errors[i] = prediction[i] - target[i]
				sampleLoss += errors[i] * errors[i]
			}
			batchLoss += sampleLoss / float32(*inputDim)

			// 3. Parallel expert gradient updates via goroutines
			// Each active expert processes gradients independently
			var wg sync.WaitGroup
			var gradMutex sync.Mutex

			for _, expertIdx := range indices {
				wg.Add(1)
				go func(idx int) {
					defer wg.Done()
					// Expert computation (GPU lock acquired inside)
					model.Experts[idx].UpdateWeights(input, errors, float32(*lr))

					// Track gradient for gater
					gradMutex.Lock()
					for _, e := range errors {
						batchExpertGrads[idx] += e
					}
					gradMutex.Unlock()
				}(expertIdx)
			}
			wg.Wait() // Wait for all active experts to finish
		}

		// Average loss across batch
		avgLoss := batchLoss / float32(*batchSize)

		// 4. Update Gater with accumulated gradients
		gaterGrad := make([]float32, *numExperts)
		for idx, grad := range batchExpertGrads {
			gaterGrad[idx] = grad / float32(*inputDim**batchSize)
		}
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
