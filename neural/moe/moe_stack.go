package moe

import (
	"fmt"
	"github.com/golangast/gollemer/neural/tensor"
)

// MoEStack represents a sequence of MoE layers with residual connections.
// Forward: x = x + Layer0(x); x = x + Layer1(x); ...
type MoEStack struct {
	Layers []*MoELayer
}

// NewMoEStack creates a new MoEStack.
func NewMoEStack(layers ...*MoELayer) *MoEStack {
	return &MoEStack{Layers: layers}
}

// Forward performs the forward pass of the MoEStack with residual connections.
func (s *MoEStack) Forward(inputs ...*tensor.Tensor) (*tensor.Tensor, error) {
	if len(inputs) == 0 {
		return nil, fmt.Errorf("MoEStack.Forward expects at least 1 input")
	}
	
	x := inputs[0]
	for _, layer := range s.Layers {
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
		x = res
	}
	
	return x, nil
}

// Backward performs the backward pass for the MoEStack.
func (s *MoEStack) Backward(grad *tensor.Tensor) error {
	// Backpropagate in reverse order
	currGrad := grad
	for i := len(s.Layers) - 1; i >= 0; i-- {
		layer := s.Layers[i]
		
		// Backpropagate through the layer.
		// The incoming gradient to 'layer' should be multiplied by ResidualScale.
		layerGrad := currGrad
		if layer.ResidualScale != nil {
			layerGrad, _ = currGrad.MulScalar(layer.ResidualScale.Data[0])
			
			// Gradient w.r.t. ResidualScale: dot(currGrad, layerOutput)
			if layer.ResidualScale.RequiresGrad {
				if layer.ResidualScale.Grad == nil {
					layer.ResidualScale.Grad = tensor.NewTensor(layer.ResidualScale.Shape, make([]float64, 1), false)
				}
				
				// Retrieve the output from the last forward pass stored in stateStack
				stack := layer.GetStateStack()
				if len(stack) > 0 {
					lastOutput := stack[len(stack)-1].lastOutput
					
					// dL/dScale = sum(currGrad * lastOutput)
					var dScale float64
					for j := range currGrad.Data {
						dScale += currGrad.Data[j] * lastOutput.Data[j]
					}
					layer.ResidualScale.Grad.Data[0] += dScale
				}
			}
		}

		if err := layer.Backward(layerGrad); err != nil {
			return err
		}
		
		// Input to this layer is the output of the previous layer (or the stack input)
		if len(layer.Inputs()) == 0 {
			continue
		}
		
		input := layer.Inputs()[0]
		if input.Grad == nil {
			input.Grad = tensor.NewTensor(input.Shape, make([]float64, len(input.Data)), false)
		}
		
		// Accumulate residual gradient
		tensor.AddAccumulate(input.Grad.Data, currGrad.Data)
		
		// Move to the next gradient in the stack
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
