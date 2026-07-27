package main

import (
	"fmt"
	"github.com/golangast/gollemer/internal/ai/moe"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
)

func main() {
	v := vocab.NewVocabulary()
	t := &moe.Trainer{}
	t.InitializeObservability(8, 500, v, nil)
	expertIDs := make([]int, 32)
	tokenIDs := []int{}
	for i := 0; i < 32; i++ {
		expertIDs[i] = i % 8
		for j := 0; j < 10; j++ {
			tokenIDs = append(tokenIDs, (i+j)%v.Size())
		}
	}
	t.RecordTrainingStep(expertIDs, tokenIDs, 1.0)
	m := t.Observability.GetDashboardMetrics(v)
	fmt.Printf("expert_lexicon=%#v\n", m["expert_lexicon"])
}
