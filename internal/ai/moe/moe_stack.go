package moe

import (
	"fmt"
	"github.com/golangast/gollemer/internal/ai/neural/nn"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

// MoEStack represents a sequence of MoE layers with residual connections.
// Forward: x = x + Layer0(x); x = x + Layer1(x); ...
type MoEStack struct {
	Layers []*MoELayer
	Norms  []*nn.LayerNormalization
}

// NewMoEStack creates a new MoEStack.
func NewMoEStack(layers ...*MoELayer) *MoEStack {
	norms := make([]*nn.LayerNormalization, len(layers))
	for i, l := range layers {
		norms[i] = nn.NewLayerNormalization(l.InputDim)
	}
	return &MoEStack{Layers: layers, Norms: norms}
}

// Forward performs the forward pass of the MoEStack with residual connections.
func (s *MoEStack) Forward(inputs ...*tensor.Tensor) (*tensor.Tensor, error) {
	if len(inputs) == 0 {
		return nil, fmt.Errorf("MoEStack.Forward expects at least 1 input")
	}
	
	x := inputs[0]
	for i, layer := range s.Layers {
		out, err := layer.Forward(x)
		if err != nil {
			return nil, err
		}
		
		// Residual: x = x + layer(x) * scale
		var outScaled *tensor.Tensor
		if layer.ResidualScale != nil {
			var err error
			outScaled, err = out.MulScalar(layer.ResidualScale.Data[0])
			if err != nil {
				return nil, fmt.Errorf("MoEStack residual scale failed: %w", err)
			}
			outScaled.Creator = out.Creator // Preserve creator for backprop
		} else {
			outScaled = out
		}

		res, err := x.Add(outScaled)
		if err != nil {
			return nil, fmt.Errorf("MoEStack residual add failed: %w", err)
		}
		
		// Apply LayerNorm
		if i < len(s.Norms) {
			normed, err := s.Norms[i].Forward(res)
			if err != nil {
				return nil, fmt.Errorf("MoEStack layer norm forward failed: %w", err)
			}
			x = normed
		} else {
			x = res
		}
	}
	
	return x, nil
}

// Backward performs the backward pass for the MoEStack.
func (s *MoEStack) Backward(grad *tensor.Tensor) error {
	// Backpropagate in reverse order
	currGrad := grad
	for i := len(s.Layers) - 1; i >= 0; i-- {
		layer := s.Layers[i]
		
		// Backpropagate through LayerNorm first
		if i < len(s.Norms) {
			if err := s.Norms[i].Backward(currGrad); err != nil {
				return fmt.Errorf("MoEStack layer norm backward failed: %w", err)
			}
			currGrad = s.Norms[i].Inputs()[0].Grad
		}

		// Backpropagate through the layer.
		layerGrad := currGrad
		if layer.ResidualScale != nil {
			layerGrad, _ = currGrad.MulScalar(layer.ResidualScale.Data[0])
			
			// Gradient w.r.t. ResidualScale: dot(currGrad, layerOutput)
			if layer.ResidualScale.RequiresGrad {
				if layer.ResidualScale.Grad == nil {
					layer.ResidualScale.Grad = tensor.NewTensor(layer.ResidualScale.Shape, make([]float32, 1), false)
				}
				
				stack := layer.GetStateStack()
				if len(stack) > 0 {
					lastOutput := stack[len(stack)-1].lastOutput
					normFactor := 1.0 / float32(len(currGrad.Data))
					var dScale float32
					for j := range currGrad.Data {
						dScale += currGrad.Data[j] * lastOutput.Data[j]
					}
					layer.ResidualScale.Grad.Data[0] += dScale * normFactor
				}
			}
		}

		if err := layer.Backward(layerGrad); err != nil {
			return err
		}
		
		if len(layer.Inputs()) == 0 {
			continue
		}
		
		input := layer.Inputs()[0]
		if input.Grad == nil {
			input.Grad = tensor.NewTensor(input.Shape, make([]float32, len(input.Data)), false)
		}
		
		// Accumulate residual gradient
		tensor.AddAccumulate(input.Grad.Data, currGrad.Data)
		
		currGrad = input.Grad
	}
	return nil
}

// Parameters returns all learnable parameters of the MoEStack.
func (s *MoEStack) Parameters() []*tensor.Tensor {
	params := []*tensor.Tensor{}
	for _, layer := range s.Layers {
		params = append(params, layer.Parameters()...)
	}
	for _, norm := range s.Norms {
		params = append(params, norm.Parameters()...)
	}
	return params
}

// Inputs returns the input tensors of the MoEStack.
func (s *MoEStack) Inputs() []*tensor.Tensor {
	if len(s.Layers) > 0 {
		return s.Layers[0].Inputs()
	}
	return []*tensor.Tensor{}
}

// SetMode sets training/inference mode for all layers.
func (s *MoEStack) SetMode(training bool) {
	for _, l := range s.Layers {
		l.SetMode(training)
	}
}

// ClearState clears internal states for BPTT.
func (s *MoEStack) ClearState() {
	for _, l := range s.Layers {
		l.ClearState()
	}
}

func (s *MoEStack) GetMoELayers() []*MoELayer {
	return s.Layers
}

// SetGateTemperature updates the temperature for all layers in the stack.
func (s *MoEStack) SetGateTemperature(temp float32) {
	for _, l := range s.Layers {
		l.SetGateTemperature(temp)
	}
}

// ToGPU moves all layers in the stack to the GPU.
func (s *MoEStack) ToGPU() {
	for _, l := range s.Layers {
		if l != nil {
			l.ToGPU()
		}
	}
}

func (s *MoEStack) SyncParameters() error {
	for _, l := range s.Layers {
		if l != nil {
			if err := l.SyncParameters(); err != nil {
				return err
			}
		}
	}
	return nil
}

func (s *MoEStack) RepairArchitecture() {
	if s.Norms == nil || len(s.Norms) != len(s.Layers) {
		s.Norms = make([]*nn.LayerNormalization, len(s.Layers))
		for i, l := range s.Layers {
			s.Norms[i] = nn.NewLayerNormalization(l.InputDim)
		}
	}
	for _, l := range s.Layers {
		if l != nil {
			l.RepairArchitecture()
		}
	}
}
