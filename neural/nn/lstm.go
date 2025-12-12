package nn

import (
	"fmt"
	"log"
	"math/rand"
	"os"

	. "github.com/zendrulat/nlptagger/neural/tensor"
)

func init() {
	log.SetOutput(os.Stderr)
	log.SetFlags(log.LstdFlags | log.Lshortfile)
}

// applyDropout applies dropout to a tensor during training.
// During training, randomly sets dropoutRate fraction of values to 0 and scales remaining by 1/(1-dropoutRate).
// During inference (training=false), returns the tensor unchanged.
func applyDropout(tensor *Tensor, dropoutRate float64, training bool) *Tensor {
	if !training || dropoutRate == 0.0 {
		return tensor
	}

	// Create dropout mask
	mask := NewTensor(tensor.Shape, make([]float64, len(tensor.Data)), false)
	scale := 1.0 / (1.0 - dropoutRate)

	for i := range mask.Data {
		if rand.Float64() < dropoutRate {
			mask.Data[i] = 0.0
		} else {
			mask.Data[i] = scale
		}
	}

	// Apply mask
	output := NewTensor(tensor.Shape, make([]float64, len(tensor.Data)), tensor.RequiresGrad)
	for i := range output.Data {
		output.Data[i] = tensor.Data[i] * mask.Data[i]
	}

	return output
}

// LSTMCell represents a single LSTM cell.
type LSTMCell struct {
	InputSize  int
	HiddenSize int

	// Weight matrices
	Wf, Wi, Wc, Wo *Tensor
	// Bias vectors
	Bf, Bi, Bc, Bo *Tensor

	// Stored for backward pass
	InputTensor    *Tensor
	PrevHidden     *Tensor
	PrevCell       *Tensor
	ft, it, ct, ot *Tensor
	cct            *Tensor
}

// NewLSTMCell creates a new LSTMCell.
func NewLSTMCell(inputSize, hiddenSize int) (*LSTMCell, error) {
	// Initialize weights
	wf, err := NewLinear(inputSize+hiddenSize, hiddenSize)
	if err != nil {
		return nil, err
	}
	wi, err := NewLinear(inputSize+hiddenSize, hiddenSize)
	if err != nil {
		return nil, err
	}
	wc, err := NewLinear(inputSize+hiddenSize, hiddenSize)
	if err != nil {
		return nil, err
	}
	wo, err := NewLinear(inputSize+hiddenSize, hiddenSize)
	if err != nil {
		return nil, err
	}

	return &LSTMCell{
		InputSize:  inputSize,
		HiddenSize: hiddenSize,
		Wf:         wf.Weights,
		Wi:         wi.Weights,
		Wc:         wc.Weights,
		Wo:         wo.Weights,
		Bf:         wf.Biases,
		Bi:         wi.Biases,
		Bc:         wc.Biases,
		Bo:         wo.Biases,
	}, nil
}

// Parameters returns all learnable parameters of the LSTMCell.
func (c *LSTMCell) Parameters() []*Tensor {
	return []*Tensor{c.Wf, c.Wi, c.Wc, c.Wo, c.Bf, c.Bi, c.Bc, c.Bo}
}

// Forward performs the forward pass of the LSTMCell.
func (c *LSTMCell) Forward(inputs ...*Tensor) (*Tensor, *Tensor, error) {
	if len(inputs) != 3 {
		return nil, nil, fmt.Errorf("LSTMCell.Forward expects 3 inputs (input, prevHidden, prevCell), got %d", len(inputs))
	}
	input, prevHidden, prevCell := inputs[0], inputs[1], inputs[2]

	// Store inputs for backward pass
	c.InputTensor = input
	c.PrevHidden = prevHidden
	c.PrevCell = prevCell

	// Concatenate input and previous hidden state
	combined, err := Concat([]*Tensor{input, prevHidden}, 1)
	if err != nil {
		return nil, nil, err
	}

	// Forget gate
	ft, err := combined.MatMul(c.Wf)
	if err != nil {
		return nil, nil, err
	}
	ft, err = ft.AddWithBroadcast(c.Bf)
	if err != nil {
		return nil, nil, err
	}
	ft, err = ft.Sigmoid()
	if err != nil {
		return nil, nil, err
	}
	c.ft = ft

	// Input gate
	it, err := combined.MatMul(c.Wi)
	if err != nil {
		return nil, nil, err
	}
	it, err = it.AddWithBroadcast(c.Bi)
	if err != nil {
		return nil, nil, err
	}
	it, err = it.Sigmoid()
	if err != nil {
		return nil, nil, err
	}
	c.it = it

	// Candidate cell state
	cct, err := combined.MatMul(c.Wc)
	if err != nil {
		return nil, nil, err
	}
	cct, err = cct.AddWithBroadcast(c.Bc)
	if err != nil {
		return nil, nil, err
	}
	cct, err = cct.Tanh()
	if err != nil {
		return nil, nil, err
	}
	c.cct = cct

	// New cell state
	ct, err := ft.Mul(prevCell)
	if err != nil {
		return nil, nil, err
	}
	it_cct, err := it.Mul(cct)
	if err != nil {
		return nil, nil, err
	}
	ct, err = ct.Add(it_cct)
	if err != nil {
		return nil, nil, err
	}
	c.ct = ct

	// Output gate
	ot, err := combined.MatMul(c.Wo)
	if err != nil {
		return nil, nil, err
	}
	ot, err = ot.AddWithBroadcast(c.Bo)
	if err != nil {
		return nil, nil, err
	}
	ot, err = ot.Sigmoid()
	if err != nil {
		return nil, nil, err
	}
	c.ot = ot

	// New hidden state
	ct_tanh, err := ct.Tanh()
	if err != nil {
		return nil, nil, fmt.Errorf("LSTMCell.Forward: Tanh operation failed: %w", err)
	}
	ht, err := ot.Mul(ct_tanh)
	if err != nil {
		return nil, nil, fmt.Errorf("LSTMCell.Forward: Mul operation failed for hidden state: %w", err)
	}

	return ht, ct, nil
}

// Backward performs the backward pass for the LSTMCell.
func (c *LSTMCell) Backward(gradHt, gradCt *Tensor) error {
	// gradHt is dL/dht, gradCt is dL/dct from next timestep

	// 1. dL/dot and dL/d(tanh(ct))
	// ht = ot * tanh(ct)
	ct_tanh, err := c.ct.Tanh()
	if err != nil {
		return err
	}
	gradOt, err := gradHt.Mul(ct_tanh)
	if err != nil {
		return err
	}
	grad_ct_tanh, err := gradHt.Mul(c.ot)
	if err != nil {
		return err
	}

	// 2. dL/dct (total)
	// tanh'(x) = 1 - tanh(x)^2
	grad_ct_from_ht, err := grad_ct_tanh.OneMinusSquareTanh(c.ct)
	if err != nil {
		return err
	}
	gradCt, err = gradCt.Add(grad_ct_from_ht)
	if err != nil {
		return err
	}

	// 3. dL/d(prev_c), dL/dft, dL/dit, dL/dcct
	// ct = ft * prev_c + it * cct
	gradPrevCell, err := gradCt.Mul(c.ft)
	if err != nil {
		return err
	}
	gradFt, err := gradCt.Mul(c.PrevCell)
	if err != nil {
		return err
	}
	gradIt, err := gradCt.Mul(c.cct)
	if err != nil {
		return err
	}
	gradCct, err := gradCt.Mul(c.it)
	if err != nil {
		return err
	}

	// 4. Backprop through activations for gates
	// sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
	gradOt_linear, err := gradOt.SigmoidBackward(c.ot)
	if err != nil {
		return err
	}
	gradFt_linear, err := gradFt.SigmoidBackward(c.ft)
	if err != nil {
		return err
	}
	gradIt_linear, err := gradIt.SigmoidBackward(c.it)
	if err != nil {
		return err
	}
	// tanh'(x) = 1 - tanh(x)^2
	gradCct_linear, err := gradCct.OneMinusSquareTanh(c.cct)
	if err != nil {
		return err
	}

	// 5. Gradients for weights and biases
	combined, err := Concat([]*Tensor{c.InputTensor, c.PrevHidden}, 1)
	if err != nil {
		return err
	}
	combinedT, err := combined.Transpose(0, 1)
	if err != nil {
		return err
	}

	gradWf, err := combinedT.MatMul(gradFt_linear)
	if err != nil {
		return err
	}
	gradWi, err := combinedT.MatMul(gradIt_linear)
	if err != nil {
		return err
	}
	gradWc, err := combinedT.MatMul(gradCct_linear)
	if err != nil {
		return err
	}
	gradWo, err := combinedT.MatMul(gradOt_linear)
	if err != nil {
		return err
	}

	gradBf, err := gradFt_linear.Sum(0)
	if err != nil {
		return err
	}
	gradBi, err := gradIt_linear.Sum(0)
	if err != nil {
		return err
	}
	gradBc, err := gradCct_linear.Sum(0)
	if err != nil {
		return err
	}
	gradBo, err := gradOt_linear.Sum(0)
	if err != nil {
		return err
	}

	// 6. Accumulate gradients for weights and biases
	if c.Wf.Grad == nil { c.Wf.Grad = NewTensor(c.Wf.Shape, nil, false) }
	c.Wf.Grad, err = c.Wf.Grad.Add(gradWf)
	if err != nil { return err }

	if c.Wi.Grad == nil { c.Wi.Grad = NewTensor(c.Wi.Shape, nil, false) }
	c.Wi.Grad, err = c.Wi.Grad.Add(gradWi)
	if err != nil { return err }

	if c.Wc.Grad == nil { c.Wc.Grad = NewTensor(c.Wc.Shape, nil, false) }
	c.Wc.Grad, err = c.Wc.Grad.Add(gradWc)
	if err != nil { return err }

	if c.Wo.Grad == nil { c.Wo.Grad = NewTensor(c.Wo.Shape, nil, false) }
	c.Wo.Grad, err = c.Wo.Grad.Add(gradWo)
	if err != nil { return err }

	if c.Bf.Grad == nil { c.Bf.Grad = NewTensor(c.Bf.Shape, nil, false) }
	c.Bf.Grad, err = c.Bf.Grad.Add(gradBf)
	if err != nil { return err }

	if c.Bi.Grad == nil { c.Bi.Grad = NewTensor(c.Bi.Shape, nil, false) }
	c.Bi.Grad, err = c.Bi.Grad.Add(gradBi)
	if err != nil { return err }

	if c.Bc.Grad == nil { c.Bc.Grad = NewTensor(c.Bc.Shape, nil, false) }
	c.Bc.Grad, err = c.Bc.Grad.Add(gradBc)
	if err != nil { return err }

	if c.Bo.Grad == nil { c.Bo.Grad = NewTensor(c.Bo.Shape, nil, false) }
	c.Bo.Grad, err = c.Bo.Grad.Add(gradBo)
	if err != nil { return err }


	// 7. Gradients for combined input
	transposedWf, err := c.Wf.Transpose(0, 1)
	if err != nil {
		return err
	}
	gradCombined_f, err := gradFt_linear.MatMul(transposedWf)
	if err != nil {
		return err
	}
	transposedWi, err := c.Wi.Transpose(0, 1)
	if err != nil {
		return err
	}
	gradCombined_i, err := gradIt_linear.MatMul(transposedWi)
	if err != nil {
		return err
	}
	transposedWc, err := c.Wc.Transpose(0, 1)
	if err != nil {
		return err
	}
	gradCombined_c, err := gradCct_linear.MatMul(transposedWc)
	if err != nil {
		return err
	}
	transposedWo, err := c.Wo.Transpose(0, 1)
	if err != nil {
		return err
	}
	gradCombined_o, err := gradOt_linear.MatMul(transposedWo)
	if err != nil {
		return err
	}

	gradCombined, err := gradCombined_f.Add(gradCombined_i)
	if err != nil {
		return err
	}
	gradCombined, err = gradCombined.Add(gradCombined_c)
	if err != nil {
		return err
	}
	gradCombined, err = gradCombined.Add(gradCombined_o)
	if err != nil {
		return err
	}

	// 8. Split gradCombined into gradInput and gradPrevHidden
	gradInput, err := gradCombined.Slice(1, 0, c.InputSize)
	if err != nil {
		return err
	}
	gradPrevHidden, err := gradCombined.Slice(1, c.InputSize, c.InputSize+c.HiddenSize)
	if err != nil {
		return err
	}

	// 9. Accumulate gradients for inputs
	if c.InputTensor.RequiresGrad {
		if c.InputTensor.Grad == nil {
			c.InputTensor.Grad = NewTensor(c.InputTensor.Shape, make([]float64, len(c.InputTensor.Data)), false)
		}
		c.InputTensor.Grad, err = c.InputTensor.Grad.Add(gradInput)
		if err != nil {
			return err
		}
	}
	if c.PrevHidden.RequiresGrad {
		if c.PrevHidden.Grad == nil {
			c.PrevHidden.Grad = NewTensor(c.PrevHidden.Shape, make([]float64, len(c.PrevHidden.Data)), false)
		}
		c.PrevHidden.Grad, err = c.PrevHidden.Grad.Add(gradPrevHidden)
		if err != nil {
			return err
		}
	}
	if c.PrevCell.RequiresGrad {
		if c.PrevCell.Grad == nil {
			c.PrevCell.Grad = NewTensor(c.PrevCell.Shape, make([]float64, len(c.PrevCell.Data)), false)
		}
		c.PrevCell.Grad, err = c.PrevCell.Grad.Add(gradPrevCell)
		if err != nil {
			return err
		}
	}

	return nil
}

// LSTM represents a multi-layer LSTM.
type LSTM struct {
	InputSize     int
	HiddenSize    int
	NumLayers     int
	Cells         [][]*LSTMCell
	timeStepCells [][]*LSTMCell // Stores cells for each timestep for BPTT
	DropoutRate   float64       // Dropout rate between layers (0.0 = no dropout)
	Training      bool          // Whether model is in training mode (dropout active)
}

// NewLSTM creates a new LSTM.
func NewLSTM(inputSize, hiddenSize, numLayers int) (*LSTM, error) {
	cells := make([][]*LSTMCell, numLayers)
	for i := 0; i < numLayers; i++ {
		layerInputSize := inputSize
		if i > 0 {
			layerInputSize = hiddenSize
		}
		cells[i] = make([]*LSTMCell, 1) // Assuming single cell per layer for now
		cell, err := NewLSTMCell(layerInputSize, hiddenSize)
		if err != nil {
			return nil, err
		}
		cells[i][0] = cell
	}
	return &LSTM{
		InputSize:  inputSize,
		HiddenSize: hiddenSize,
		NumLayers:  numLayers,
		Cells:      cells,
	}, nil
}

// Parameters returns all learnable parameters of the LSTM.
func (l *LSTM) Parameters() []*Tensor {
	params := []*Tensor{}
	for _, layer := range l.Cells {
		for _, cell := range layer {
			params = append(params, cell.Parameters()...)
		}
	}
	return params
}

// Forward performs the forward pass of the LSTM.
func (l *LSTM) Forward(inputs ...*Tensor) (*Tensor, *Tensor, error) {
	if len(inputs) != 3 {
		return nil, nil, fmt.Errorf("LSTM.Forward expects 3 inputs (input, initialHidden, initialCell), got %d", len(inputs))
	}
	input, initialHidden, initialCell := inputs[0], inputs[1], inputs[2]

	if len(input.Shape) == 3 { // Sequence input
		batchSize := input.Shape[0]
		sequenceLength := input.Shape[1]

		// Initialize timeStepCells for BPTT
		l.timeStepCells = make([][]*LSTMCell, l.NumLayers)
		for i := range l.timeStepCells {
			l.timeStepCells[i] = make([]*LSTMCell, sequenceLength)
		}

		layerInput := input
		var lastCellState *Tensor

		for i := 0; i < l.NumLayers; i++ {
			h, c := initialHidden, initialCell
			if i > 0 {
				h = NewTensor([]int{batchSize, l.HiddenSize}, make([]float64, batchSize*l.HiddenSize), false)
				c = NewTensor([]int{batchSize, l.HiddenSize}, make([]float64, batchSize*l.HiddenSize), false)
			}

			outputs := make([]*Tensor, sequenceLength)
			for t := 0; t < sequenceLength; t++ {
				timeStepInput, err := layerInput.Slice(1, t, t+1)
				if err != nil {
					return nil, nil, fmt.Errorf("slicing input failed: %w", err)
				}
				timeStepInput, err = timeStepInput.Squeeze(1)
				if err != nil {
					return nil, nil, fmt.Errorf("squeezing input failed: %w", err)
				}

				// Create a new cell for this timestep with shared weights
				cellForTimeStep := *l.Cells[i][0] // Copy struct, pointers to weights are shared
				l.timeStepCells[i][t] = &cellForTimeStep

				h, c, err = l.timeStepCells[i][t].Forward(timeStepInput, h, c)
				if err != nil {
					return nil, nil, fmt.Errorf("LSTMCell forward failed: %w", err)
				}
				outputs[t] = h
			}

			lastCellState = c

			// Manual stack since Stack function is not available
			stackedOutputData := make([]float64, batchSize*sequenceLength*l.HiddenSize)
			for t, ht := range outputs {
				for b := 0; b < batchSize; b++ {
					copy(stackedOutputData[(b*sequenceLength+t)*l.HiddenSize:(b*sequenceLength+t+1)*l.HiddenSize], ht.Data[b*l.HiddenSize:(b+1)*l.HiddenSize])
				}
			}
			stackedOutput := NewTensor([]int{batchSize, sequenceLength, l.HiddenSize}, stackedOutputData, true)

			if i < l.NumLayers-1 {
				layerInput = applyDropout(stackedOutput, l.DropoutRate, l.Training)
			} else {
				// Also store the full layerInput in the cell for the backward pass to find it.
				for t := 0; t < sequenceLength; t++ {
					l.timeStepCells[i][t].InputTensor.Creator = layerInput
				}
				return stackedOutput, lastCellState, nil
			}
		}
		return nil, nil, fmt.Errorf("LSTM forward loop finished without returning") // Should not happen
	} else if len(input.Shape) == 2 { // Single time step
		var currentHidden, currentCell *Tensor = initialHidden, initialCell
		var layerOutput *Tensor = input

		for i := 0; i < l.NumLayers; i++ {
			if i > 0 {
				layerOutput = currentHidden
				if i < l.NumLayers {
					layerOutput = applyDropout(layerOutput, l.DropoutRate, l.Training)
				}
			}

			ht, ct, err := l.Cells[i][0].Forward(layerOutput, currentHidden, currentCell)
			if err != nil {
				log.Printf("LSTMCell.Forward in LSTM.Forward failed: %+v", err)
				return nil, nil, err
			}
			currentHidden = ht
			currentCell = ct
		}
		return currentHidden, currentCell, nil
	} else {
		return nil, nil, fmt.Errorf("LSTM.Forward expects a 2D or 3D input tensor, got %d dimensions", len(input.Shape))
	}
}

// Backward performs the backward pass for the entire LSTM layer.
func (l *LSTM) Backward(gradNextHidden, gradNextCell *Tensor) error {
	// If timeStepCells is not populated, it means forward was not called on a sequence.
	// Fallback to the simple, single-step backward pass.
	if len(l.timeStepCells) == 0 {
		gradH := gradNextHidden
		gradC := gradNextCell
		var err error
		for i := l.NumLayers - 1; i >= 0; i-- {
			cell := l.Cells[i][0]
			err = cell.Backward(gradH, gradC)
			if err != nil {
				return fmt.Errorf("failed to backpropagate through LSTM cell in layer %d: %w", i, err)
			}
			if i > 0 {
				if cell.InputTensor.Grad == nil || cell.PrevHidden.Grad == nil {
					return fmt.Errorf("gradient not computed for input or hidden state in layer %d", i)
				}
				gradH, err = cell.InputTensor.Grad.Add(cell.PrevHidden.Grad)
				if err != nil {
					return err
				}
				gradC = cell.PrevCell.Grad
			}
		}
		return nil
	}

	// --- Backpropagation Through Time (BPTT) ---
	gradH := gradNextHidden
	gradC := gradNextCell

	for i := l.NumLayers - 1; i >= 0; i-- {
		sequenceLength := len(l.timeStepCells[i])
		layerInputTensor := l.timeStepCells[i][0].InputTensor.Creator.(*Tensor)

		if layerInputTensor.Grad == nil {
			layerInputTensor.Grad = NewTensor(layerInputTensor.Shape, make([]float64, len(layerInputTensor.Data)), false)
		}

		// Initialize gradients from future (t+1)
		gradHFromFuture := NewTensor(gradH.Shape, make([]float64, len(gradH.Data)), false)
		gradCFromFuture := NewTensor(gradC.Shape, make([]float64, len(gradC.Data)), false)

		for t := sequenceLength - 1; t >= 0; t-- {
			cell := l.timeStepCells[i][t]

			// Total gradient for ht = (grad from layer above) + (grad from h_{t+1})
			var totalGradH, totalGradC *Tensor
			var err error
			if t == sequenceLength-1 {
				totalGradH, err = gradH.Add(gradHFromFuture)
				if err != nil {
					return err
				}
				totalGradC, err = gradC.Add(gradCFromFuture)
				if err != nil {
					return err
				}
			} else {
				totalGradH = gradHFromFuture
				totalGradC = gradCFromFuture
			}

			err = cell.Backward(totalGradH, totalGradC)
			if err != nil {
				return fmt.Errorf("BPTT: cell.Backward at t=%d, layer=%d failed: %w", t, i, err)
			}

			// Gradients for the previous time step are now available
			gradHFromFuture = cell.PrevHidden.Grad
			gradCFromFuture = cell.PrevCell.Grad

			// Accumulate gradient for the input of this layer
			// The input to the cell was a slice, so its gradient must be manually added
			// to the correct position in the full input gradient tensor for this layer.
			if cell.InputTensor.Grad != nil {
				offset := t * cell.InputTensor.Shape[0] * cell.InputTensor.Shape[1]
				for j, gradVal := range cell.InputTensor.Grad.Data {
					layerInputTensor.Grad.Data[offset+j] += gradVal
				}
			}
		}

		// The gradient for the input of this layer becomes the gradH for the layer below.
		gradH = layerInputTensor.Grad
		// There is no cell state gradient between layers.
		gradC = NewTensor(gradC.Shape, make([]float64, len(gradC.Data)), false)
	}
	return nil
}
