package main

import (
	"fmt"
	"log"

	"github.com/golangast/gollemer/internal/ai/moe"
	"github.com/golangast/gollemer/internal/ai/tensor"
)

func main() {
	modelPath := "data/models/gob_models/moe_social_model.gob"
	ckpt, err := moe.LoadIntentMoECheckpoint(modelPath)
	if err != nil {
		log.Fatalf("Failed to load: %v", err)
	}

	targetExperts := 16

	layers := append(ckpt.Model.Encoder.GetMoELayers(), ckpt.Model.Decoder.OutputMoE)
	for i, l := range layers {
		if len(l.Experts) > targetExperts {
			fmt.Printf("Pruning Layer %d from %d to %d experts\n", i, len(l.Experts), targetExperts)
			l.Experts = l.Experts[:targetExperts]
			if l.ExpertOutputScale != nil && len(l.ExpertOutputScale) > targetExperts {
				l.ExpertOutputScale = l.ExpertOutputScale[:targetExperts]
			}
			
			if l.AccumulatedUtilization != nil && len(l.AccumulatedUtilization) > targetExperts {
				newMap := make(map[int]int)
				for k, v := range l.AccumulatedUtilization {
					if k < targetExperts {
						newMap[k] = v
					}
				}
				l.AccumulatedUtilization = newMap
			}

			if l.StepRoutingBias != nil {
				for k, arr := range l.StepRoutingBias {
					if len(arr) > targetExperts {
						l.StepRoutingBias[k] = arr[:targetExperts]
					}
				}
			}
			
			// Resize GatingNetwork weights
			if l.GatingNetwork != nil {
				oldWeights := l.GatingNetwork.Linear.Weights
				inputDim := oldWeights.Shape[0]
				oldNumExperts := oldWeights.Shape[1]
				
				newWeights := tensor.NewTensor([]int{inputDim, targetExperts}, make([]float32, inputDim*targetExperts), true)
				for d := 0; d < inputDim; d++ {
					for e := 0; e < targetExperts; e++ {
						newWeights.Data[d*targetExperts + e] = oldWeights.Data[d*oldNumExperts + e]
					}
				}
				l.GatingNetwork.Linear.Weights = newWeights
			}
		}
	}
	
	ckpt.Model.RebuildActiveLayers()

	// Save the model
	if err := moe.SaveIntentMoECheckpoint(ckpt, modelPath); err != nil {
		log.Fatalf("Failed to save: %v", err)
	}
	fmt.Println("Model pruned successfully!")
}
