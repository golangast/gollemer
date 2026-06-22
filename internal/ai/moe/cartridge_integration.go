package moe

import (
	"fmt"
	"log"

	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

// MountCartridgeToLayer adds a loaded expert to a specific MoE layer for hot-swapping.
func (s *Supervisor) MountCartridgeToLayer(model *IntentMoE, layerIdx int, expert Expert) (int, error) {
	layers := model.Encoder.GetMoELayers()
	if model.Decoder.OutputMoE != nil {
		layers = append(layers, model.Decoder.OutputMoE)
	}

	if layerIdx < 0 || layerIdx >= len(layers) {
		return -1, fmt.Errorf("MountCartridgeToLayer: layerIdx %d out of range", layerIdx)
	}
	layer := layers[layerIdx]

	s.mu.Lock()
	defer s.mu.Unlock()

	newID := len(layer.Experts)
	layer.Experts = append(layer.Experts, expert)
	layer.NumExperts++

	// Extend slices
	layer.ExpertOutputScale = append(layer.ExpertOutputScale, 1.0)
	layer.AccumulatedUtilization = append(layer.AccumulatedUtilization, 0)
	layer.ExpertFrozen = append(layer.ExpertFrozen, true) // Cartridges are read-only
	layer.StagnationCounters = append(layer.StagnationCounters, 0)
	layer.ExpertGradMultiplier = append(layer.ExpertGradMultiplier, 1.0)
	layer.ExpertHealth = append(layer.ExpertHealth, 1.0)
	layer.ExpertRole = append(layer.ExpertRole, "CARTRIDGE")

	// Extend gating network
	if layer.GatingNetwork != nil && layer.GatingNetwork.Linear != nil {
		gw := layer.GatingNetwork.Linear.Weights
		oldNumExperts := gw.Shape[1]
		newNumExperts := oldNumExperts + 1
		oldData := gw.Data
		newData := make([]float32, gw.Shape[0]*newNumExperts)
		for row := 0; row < gw.Shape[0]; row++ {
			copy(newData[row*newNumExperts:row*newNumExperts+oldNumExperts],
				oldData[row*oldNumExperts:row*oldNumExperts+oldNumExperts])
			// Give cartridge a strong initial routing bias so it actually gets used
			newData[row*newNumExperts+oldNumExperts] = 0.5 
		}
		layer.GatingNetwork.Linear.Weights = tensor.NewTensor(
			[]int{gw.Shape[0], newNumExperts}, newData, true)

		if layer.GatingNetwork.Linear.Biases != nil {
			oldBias := layer.GatingNetwork.Linear.Biases.Data
			newBias := make([]float32, newNumExperts)
			copy(newBias, oldBias)
			newBias[oldNumExperts] = 2.0 // Strong bias
			layer.GatingNetwork.Linear.Biases = tensor.NewTensor(
				[]int{newNumExperts}, newBias, true)
		}
		layer.GatingNetwork.RepairArchitecture()
	}

	log.Printf("🔌 Cartridge Mounted: Expert %d added to Layer %d", newID, layerIdx)
	return newID, nil
}

// UnmountCartridgeFromLayer removes the dynamically added expert from a layer.
func (s *Supervisor) UnmountCartridgeFromLayer(model *IntentMoE, layerIdx int, expertIdx int) error {
	layers := model.Encoder.GetMoELayers()
	if model.Decoder.OutputMoE != nil {
		layers = append(layers, model.Decoder.OutputMoE)
	}

	if layerIdx < 0 || layerIdx >= len(layers) {
		return fmt.Errorf("UnmountCartridgeFromLayer: layerIdx out of range")
	}
	layer := layers[layerIdx]

	s.mu.Lock()
	defer s.mu.Unlock()

	if expertIdx < 0 || expertIdx >= len(layer.Experts) {
		return fmt.Errorf("UnmountCartridgeFromLayer: expertIdx out of range")
	}

	// Remove expert from slice
	layer.Experts = append(layer.Experts[:expertIdx], layer.Experts[expertIdx+1:]...)
	layer.NumExperts--

	layer.ExpertOutputScale = append(layer.ExpertOutputScale[:expertIdx], layer.ExpertOutputScale[expertIdx+1:]...)
	layer.AccumulatedUtilization = append(layer.AccumulatedUtilization[:expertIdx], layer.AccumulatedUtilization[expertIdx+1:]...)
	layer.ExpertFrozen = append(layer.ExpertFrozen[:expertIdx], layer.ExpertFrozen[expertIdx+1:]...)
	layer.StagnationCounters = append(layer.StagnationCounters[:expertIdx], layer.StagnationCounters[expertIdx+1:]...)
	layer.ExpertGradMultiplier = append(layer.ExpertGradMultiplier[:expertIdx], layer.ExpertGradMultiplier[expertIdx+1:]...)
	layer.ExpertHealth = append(layer.ExpertHealth[:expertIdx], layer.ExpertHealth[expertIdx+1:]...)
	layer.ExpertRole = append(layer.ExpertRole[:expertIdx], layer.ExpertRole[expertIdx+1:]...)

	// Shrink gating network
	if layer.GatingNetwork != nil && layer.GatingNetwork.Linear != nil {
		gw := layer.GatingNetwork.Linear.Weights
		oldNumExperts := gw.Shape[1]
		newNumExperts := oldNumExperts - 1
		oldData := gw.Data
		newData := make([]float32, gw.Shape[0]*newNumExperts)
		for row := 0; row < gw.Shape[0]; row++ {
			copy(newData[row*newNumExperts:row*newNumExperts+expertIdx], oldData[row*oldNumExperts:row*oldNumExperts+expertIdx])
			copy(newData[row*newNumExperts+expertIdx:row*newNumExperts+newNumExperts], oldData[row*oldNumExperts+expertIdx+1:row*oldNumExperts+oldNumExperts])
		}
		layer.GatingNetwork.Linear.Weights = tensor.NewTensor(
			[]int{gw.Shape[0], newNumExperts}, newData, true)

		if layer.GatingNetwork.Linear.Biases != nil {
			oldBias := layer.GatingNetwork.Linear.Biases.Data
			newBias := make([]float32, newNumExperts)
			copy(newBias[:expertIdx], oldBias[:expertIdx])
			copy(newBias[expertIdx:], oldBias[expertIdx+1:])
			layer.GatingNetwork.Linear.Biases = tensor.NewTensor(
				[]int{newNumExperts}, newBias, true)
		}
		layer.GatingNetwork.RepairArchitecture()
	}

	log.Printf("🔌 Cartridge Unmounted: Expert %d removed from Layer %d", expertIdx, layerIdx)
	return nil
}
