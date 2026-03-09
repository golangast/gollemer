package nn

import (
	"fmt"
	"log"
	"math/rand"
	"os"

	. "github.com/golangast/gollemer/neural/tensor"
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
	MulVectors(tensor.Data, mask.Data, output.Data)

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
	if inputs[0] == nil || inputs[1] == nil || inputs[2] == nil {
		return nil, nil, fmt.Errorf("LSTMCell.Forward received nil input tensor(s)")
	}
	input, prevHidden, prevCell := inputs[0], inputs[1], inputs[2]

	// Store inputs for backward pass
	c.InputTensor = input
	c.PrevHidden = prevHidden
	c.PrevCell = prevCell

	// Instead of Concat, we perform two MatMuls or slice the weight matrix.
	// W = [W_input; W_hidden]
	// ft = sigmoid(input @ Wf_in + prevHidden @ Wf_hid + Bf)

	inputSize := c.InputSize

	// Slice weights for input and hidden parts without copying (using Reshape might be tricky here,
	// but since we know the layout, we can just use the MatMul on slices if MatMul supported it).
	// For now, to keep it simple and safe, we'll continue using the combined MatMul but
	// we'll optimize the Concat out by using a pre-allocated buffer or just doing two MatMuls.
	// Two MatMuls is cleaner for now.

	wf_in, err := c.Wf.Slice(0, 0, inputSize)
	if err != nil {
		return nil, nil, err
	}
	wf_hid, err := c.Wf.Slice(0, inputSize, c.Wf.Shape[0])
	if err != nil {
		return nil, nil, err
	}

	wi_in, err := c.Wi.Slice(0, 0, inputSize)
	if err != nil {
		return nil, nil, err
	}
	wi_hid, err := c.Wi.Slice(0, inputSize, c.Wi.Shape[0])
	if err != nil {
		return nil, nil, err
	}

	wc_in, err := c.Wc.Slice(0, 0, inputSize)
	if err != nil {
		return nil, nil, err
	}
	wc_hid, err := c.Wc.Slice(0, inputSize, c.Wc.Shape[0])
	if err != nil {
		return nil, nil, err
	}

	wo_in, err := c.Wo.Slice(0, 0, inputSize)
	if err != nil {
		return nil, nil, err
	}
	wo_hid, err := c.Wo.Slice(0, inputSize, c.Wo.Shape[0])
	if err != nil {
		return nil, nil, err
	}

	// Forget gate
	f_in, err := input.MatMul(wf_in)
	if err != nil {
		return nil, nil, err
	}
	f_hid, err := prevHidden.MatMul(wf_hid)
	if err != nil {
		return nil, nil, err
	}
	ft, err := f_in.Add(f_hid)
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
	i_in, err := input.MatMul(wi_in)
	if err != nil {
		return nil, nil, err
	}
	i_hid, err := prevHidden.MatMul(wi_hid)
	if err != nil {
		return nil, nil, err
	}
	it, err := i_in.Add(i_hid)
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
	cc_in, err := input.MatMul(wc_in)
	if err != nil {
		return nil, nil, err
	}
	cc_hid, err := prevHidden.MatMul(wc_hid)
	if err != nil {
		return nil, nil, err
	}
	cct, err := cc_in.Add(cc_hid)
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

	// Output gate
	o_in, err := input.MatMul(wo_in)
	if err != nil {
		return nil, nil, err
	}
	o_hid, err := prevHidden.MatMul(wo_hid)
	if err != nil {
		return nil, nil, err
	}
	ot, err := o_in.Add(o_hid)
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

	// Cell state
	// ct = ft * prev_c + it * cct
	term1, err := ft.Mul(prevCell)
	if err != nil {
		return nil, nil, err
	}
	term2, err := it.Mul(cct)
	if err != nil {
		return nil, nil, err
	}
	ct, err := term1.Add(term2)
	if err != nil {
		return nil, nil, err
	}
	c.ct = ct

	// Hidden state
	// ht = ot * tanh(ct)
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
	if c.Wf.Grad == nil {
		c.Wf.Grad = NewTensor(c.Wf.Shape, make([]float64, len(c.Wf.Data)), false)
	}
	safeAccumulate(c.Wf.Grad.Data, gradWf.Data)

	if c.Wi.Grad == nil {
		c.Wi.Grad = NewTensor(c.Wi.Shape, make([]float64, len(c.Wi.Data)), false)
	}
	safeAccumulate(c.Wi.Grad.Data, gradWi.Data)

	if c.Wc.Grad == nil {
		c.Wc.Grad = NewTensor(c.Wc.Shape, make([]float64, len(c.Wc.Data)), false)
	}
	safeAccumulate(c.Wc.Grad.Data, gradWc.Data)

	if c.Wo.Grad == nil {
		c.Wo.Grad = NewTensor(c.Wo.Shape, make([]float64, len(c.Wo.Data)), false)
	}
	safeAccumulate(c.Wo.Grad.Data, gradWo.Data)

	if c.Bf.Grad == nil {
		c.Bf.Grad = NewTensor(c.Bf.Shape, make([]float64, len(c.Bf.Data)), false)
	}
	safeAccumulate(c.Bf.Grad.Data, gradBf.Data)

	if c.Bi.Grad == nil {
		c.Bi.Grad = NewTensor(c.Bi.Shape, make([]float64, len(c.Bi.Data)), false)
	}
	safeAccumulate(c.Bi.Grad.Data, gradBi.Data)

	if c.Bc.Grad == nil {
		c.Bc.Grad = NewTensor(c.Bc.Shape, make([]float64, len(c.Bc.Data)), false)
	}
	safeAccumulate(c.Bc.Grad.Data, gradBc.Data)

	if c.Bo.Grad == nil {
		c.Bo.Grad = NewTensor(c.Bo.Shape, make([]float64, len(c.Bo.Data)), false)
	}
	safeAccumulate(c.Bo.Grad.Data, gradBo.Data)

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
	for i := range numLayers {
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

			// Pre-projection optimization:
			// Combine input-part weights for all 4 gates into one large matrix
			cell0 := l.Cells[i][0]
			inSize := layerInput.Shape[2]

			wf_in, _ := cell0.Wf.Slice(0, 0, inSize)
			wi_in, _ := cell0.Wi.Slice(0, 0, inSize)
			wc_in, _ := cell0.Wc.Slice(0, 0, inSize)
			wo_in, _ := cell0.Wo.Slice(0, 0, inSize)
			w_in_all, _ := Concat([]*Tensor{wf_in, wi_in, wc_in, wo_in}, 1)

			b_all, _ := Concat([]*Tensor{cell0.Bf, cell0.Bi, cell0.Bc, cell0.Bo}, 0)

			// Combine hidden-part weights once per layer
			wf_hid, _ := cell0.Wf.Slice(0, inSize, cell0.Wf.Shape[0])
			wi_hid, _ := cell0.Wi.Slice(0, inSize, cell0.Wi.Shape[0])
			wc_hid, _ := cell0.Wc.Slice(0, inSize, cell0.Wc.Shape[0])
			wo_hid, _ := cell0.Wo.Slice(0, inSize, cell0.Wo.Shape[0])
			w_hid_all, _ := Concat([]*Tensor{wf_hid, wi_hid, wc_hid, wo_hid}, 1)

			// Project the entire input sequence once
			reshapedInput, _ := layerInput.Reshape([]int{batchSize * sequenceLength, inSize})
			inputProjAll, err := reshapedInput.MatMul(w_in_all)
			if err != nil {
				return nil, nil, fmt.Errorf("failed to pre-project input: %w", err)
			}
			inputProjAll, _ = inputProjAll.AddWithBroadcast(b_all)

			// Transpose inputProjAll to [sequenceLength, batchSize, 4 * hiddenSize]
			inputProjAll3D, _ := inputProjAll.Reshape([]int{batchSize, sequenceLength, 4 * l.HiddenSize})
			inputProjTrans, _ := inputProjAll3D.Transpose(0, 1)

			outputs := make([]*Tensor, sequenceLength)
			for t := range sequenceLength {
				// Use pre-computed input projection view (zero-copy)
				projTData := inputProjTrans.Data[t*batchSize*4*l.HiddenSize : (t+1)*batchSize*4*l.HiddenSize]
				projT := NewTensor([]int{batchSize, 4 * l.HiddenSize}, projTData, false)

				// Create a new cell for this timestep with shared weights for BPTT
				cellForTimeStep := *cell0
				l.timeStepCells[i][t] = &cellForTimeStep

				// Compute hidden projection
				hidProjAll, _ := h.MatMul(w_hid_all)

				// total gates = input_precomputed + hidden_projection
				gatesAll, _ := projT.Add(hidProjAll)

				// Split gatesAll into ft, it, cct, ot
				ft, _ := gatesAll.Slice(1, 0, l.HiddenSize)
				it, _ := gatesAll.Slice(1, l.HiddenSize, 2*l.HiddenSize)
				cct, _ := gatesAll.Slice(1, 2*l.HiddenSize, 3*l.HiddenSize)
				ot, _ := gatesAll.Slice(1, 3*l.HiddenSize, 4*l.HiddenSize)

				ft, _ = ft.Sigmoid()
				it, _ = it.Sigmoid()
				cct, _ = cct.Tanh()
				ot, _ = ot.Sigmoid()

				cellForTimeStep.ft = ft
				cellForTimeStep.it = it
				cellForTimeStep.cct = cct
				cellForTimeStep.ot = ot
				// Extract input for this timestep to allow Backward to compute weight gradients
				inputSlice, _ := layerInput.Slice(1, t, t+1)
				inputT, _ := inputSlice.Reshape([]int{batchSize, inSize})
				cellForTimeStep.InputTensor = inputT
				cellForTimeStep.PrevHidden = h
				cellForTimeStep.PrevCell = c

				// ct = ft * c + it * cct
				term1, _ := ft.Mul(c)
				term2, _ := it.Mul(cct)
				c, _ = term1.Add(term2)
				cellForTimeStep.ct = c

				// ht = ot * tanh(ct)
				ct_tanh, _ := c.Tanh()
				h, _ = ot.Mul(ct_tanh)

				outputs[t] = h
			}

			lastCellState = c

			// Manual stack along dimension 1 to create [batchSize, sequenceLength, l.HiddenSize]
			stackedOutputData := make([]float64, batchSize*sequenceLength*l.HiddenSize)
			for t, ht := range outputs {
				for b := 0; b < batchSize; b++ {
					copy(stackedOutputData[(b*sequenceLength+t)*l.HiddenSize:(b*sequenceLength+t+1)*l.HiddenSize], ht.Data[b*l.HiddenSize:(b+1)*l.HiddenSize])
				}
			}
			stackedOutput := NewTensor([]int{batchSize, sequenceLength, l.HiddenSize}, stackedOutputData, true)

			// Store the full layerInput in the cell for the backward pass to find it
			for t := range sequenceLength {
				l.timeStepCells[i][t].InputTensor.Creator = layerInput
			}

			if i < l.NumLayers-1 {
				layerInput = applyDropout(stackedOutput, l.DropoutRate, l.Training)
			} else {
				return stackedOutput, lastCellState, nil
			}
		}
		return nil, nil, fmt.Errorf("LSTM forward loop finished without returning") // Should not happen
	} else if len(input.Shape) == 2 { // Single time step
		// Clear timeStepCells to ensure Backward uses the single-step path
		l.timeStepCells = nil

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

		// Initialize gradients from future (t+1) as 2D tensors [batchSize, hiddenSize]
		batchSize := gradH.Shape[0]
		gradHFromFuture := NewTensor([]int{batchSize, l.HiddenSize}, make([]float64, batchSize*l.HiddenSize), false)
		gradCFromFuture := NewTensor([]int{batchSize, l.HiddenSize}, make([]float64, batchSize*l.HiddenSize), false)

		for t := sequenceLength - 1; t >= 0; t-- {
			cell := l.timeStepCells[i][t]

			// Total gradient for ht = (grad from layer above) + (grad from h_{t+1})
			// Get incoming gradient for this time step
			var currentStepGradH *Tensor
			if len(gradH.Shape) == 3 {
				// Input gradH is [batchSize, sequenceLength, hiddenSize]
				shSlice, _ := gradH.Slice(1, t, t+1)
				currentStepGradH, _ = shSlice.Squeeze(1)
			} else if t == sequenceLength-1 {
				// Input gradH is [batchSize, hiddenSize], only applies to last step
				currentStepGradH = gradH
			} else {
				// No incoming gradient for this middle step (e.g. if only last step used)
				currentStepGradH = NewTensor(gradHFromFuture.Shape, make([]float64, len(gradHFromFuture.Data)), false)
			}

			totalGradH, err := currentStepGradH.Add(gradHFromFuture)
			if err != nil {
				return err
			}
			
			// For cell state, normally only gradNextCell (last step) is provided
			totalGradC := gradCFromFuture
			if t == sequenceLength-1 {
				totalGradC, err = gradC.Add(gradCFromFuture)
				if err != nil {
					return err
				}
			}

			err = cell.Backward(totalGradH, totalGradC)
			if err != nil {
				return fmt.Errorf("BPTT: cell.Backward at t=%d, layer=%d failed: %w", t, i, err)
			}

			// Gradients for the previous time step
			gradHFromFuture = cell.PrevHidden.Grad
			gradCFromFuture = cell.PrevCell.Grad

			// Accumulate gradient for the input of this layer
			if cell.InputTensor.Grad != nil {
				batchSize := cell.InputTensor.Shape[0]
				inputSize := cell.InputTensor.Shape[1]
				for b := 0; b < batchSize; b++ {
					outOffset := (b*sequenceLength + t) * inputSize
					inOffset := b * inputSize
					if outOffset >= 0 && outOffset+inputSize <= len(layerInputTensor.Grad.Data) && inOffset >= 0 && inOffset+inputSize <= len(cell.InputTensor.Grad.Data) {
						safeAccumulate(layerInputTensor.Grad.Data[outOffset:outOffset+inputSize], cell.InputTensor.Grad.Data[inOffset:inOffset+inputSize])
					} else {
						// Skip invalid indices to prevent panic
					}
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

func (l *LSTM) GetInputGrad() *Tensor {
	if len(l.timeStepCells) > 0 && len(l.timeStepCells[0]) > 0 {
		creator := l.timeStepCells[0][0].InputTensor.Creator
		if creator != nil {
			if t, ok := creator.(*Tensor); ok {
				return t.Grad
			}
		}
	}
	return nil
}

func (l *LSTM) GetCellState(layer, timestep int) *Tensor {
	if layer < 0 || layer >= l.NumLayers || l.timeStepCells == nil || timestep < 0 || timestep >= len(l.timeStepCells[0]) {
		return nil
	}
	return l.timeStepCells[layer][timestep].ct
}

func (l *LSTM) SetCellState(hidden, cell *Tensor) {
	for i := 0; i < l.NumLayers; i++ {
		l.Cells[i][0].PrevHidden = hidden
		l.Cells[i][0].PrevCell = cell
	}
}

func (l *LSTM) BackwardStep(gradH, gradC *Tensor, timestep int) error {
	if l.timeStepCells == nil || timestep < 0 || timestep >= len(l.timeStepCells[0]) {
		return fmt.Errorf("invalid timestep for BackwardStep")
	}

	currentGradH := gradH
	currentGradC := gradC

	// Backprop through layers at this timestep
	for layer := l.NumLayers - 1; layer >= 0; layer-- {
		cell := l.timeStepCells[layer][timestep]
		err := cell.Backward(currentGradH, currentGradC)
		if err != nil {
			return err
		}
		currentGradH = cell.InputTensor.Grad
		currentGradC = cell.PrevCell.Grad
	}
	return nil
}

func (l *LSTM) GetPrevHiddenGrad() *Tensor {
	if l.timeStepCells == nil || len(l.timeStepCells[0]) == 0 {
		return nil
	}
	return l.timeStepCells[0][0].PrevHidden.Grad
}

func (l *LSTM) GetPrevCellGrad() *Tensor {
	if l.timeStepCells == nil || len(l.timeStepCells[0]) == 0 {
		return nil
	}
	return l.timeStepCells[0][0].PrevCell.Grad
}

func (l *LSTM) GetInputGradStep(t int) *Tensor {
	if l.timeStepCells == nil || t < 0 || t >= len(l.timeStepCells[0]) {
		return nil
	}
	return l.timeStepCells[0][t].InputTensor.Grad
}

// ClearState clears the internal state of the LSTM.
func (l *LSTM) ClearState() {
	l.timeStepCells = nil
}
