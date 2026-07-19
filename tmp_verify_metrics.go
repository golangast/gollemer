//go:build ignore
// +build ignore

package main

import (
	"fmt"

	"github.com/golangast/gollemer/internal/ai/moe"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

func main() {
	v := vocab.NewVocabulary()
	trainer := &moe.Trainer{}
	trainer.InitializeObservability(8, 100, v, nil)
	agg := moe.NewMetricsAggregator(trainer, v)
	for step := 0; step < 5; step++ {
		trainer.RecordTrainingStep([]int{step % 8}, []int{step, step + 1}, float32(0.5+float32(step)*0.01))
		if step%2 == 0 {
			data := make([]float32, 8*64)
			for i := range data {
				data[i] = float32((step+i)%15) * 0.02
			}
			t := tensor.NewTensor([]int{8, 64}, data, false)
			trainer.RecordWeightSnapshot("layer_0", make([]float32, 128))
			trainer.UpdateWeightVelocity("layer_0", make([]float32, 128), t)
			trainer.UpdateEmbeddingGalaxy(v, t, 20)
		}
		m := agg.CollectMetrics()
		fmt.Printf("step=%d drift=%v health=%v\n", step, m["semantic_drift"], m["health_indicators"])
	}
}
