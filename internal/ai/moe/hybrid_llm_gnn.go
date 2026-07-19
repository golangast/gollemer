package moe

import (
	"encoding/gob"
	"fmt"
	"github.com/golangast/gollemer/internal/ai/neural/nn"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
	"log"
)

func init() {
	gob.Register(&HybridLLMGNNEncoder{})
}

// HybridLLMGNNEncoder combines an LLM-based encoder (like MoE or BERT) with a GNN layer.
type HybridLLMGNNEncoder struct {
	LLMEncoder   Encoder // Can be MoELayer or simple Encoder
	GNNLayer     *nn.GCNLayer
	LayerNorm    *nn.LayerNorm
	EmbeddingDim int

	// Intermediate tensors for backward
	llmOutput   *tensor.Tensor
	gnnOutput   *tensor.Tensor
	adj         *tensor.Tensor
	residualOut *tensor.Tensor
}

func NewHybridLLMGNNEncoder(llmEncoder Encoder, embeddingDim int) (*HybridLLMGNNEncoder, error) {
	gnn, err := nn.NewGCNLayer(embeddingDim, embeddingDim)
	if err != nil {
		return nil, err
	}
	ln := nn.NewLayerNorm(embeddingDim)
	return &HybridLLMGNNEncoder{
		LLMEncoder:   llmEncoder,
		GNNLayer:     gnn,
		LayerNorm:    ln,
		EmbeddingDim: embeddingDim,
	}, nil
}

func (e *HybridLLMGNNEncoder) Forward(inputs ...*tensor.Tensor) (*tensor.Tensor, error) {
	// 1. Run LLM Encoder
	llmOut, err := e.LLMEncoder.Forward(inputs...)
	if err != nil {
		return nil, fmt.Errorf("HybridLLMGNNEncoder LLM failed: %w", err)
	}

	// 2. Build Graph (Window-based Chain)
	// Assume llmOut is [batch, seq_len, dim]
	// If 2D (batch=1), reshape to 3D for consistency or handle separately.
	// MoE usually returns [batch, seq_len, hidden].

	if len(llmOut.Shape) != 3 {
		// If 2D [seq_len, dim], treat as batch=1
		if len(llmOut.Shape) == 2 {
			llmOut, _ = llmOut.Reshape([]int{1, llmOut.Shape[0], llmOut.Shape[1]})
		} else {
			return nil, fmt.Errorf("HybridLLMGNNEncoder expects 3D output from LLM, got %v", llmOut.Shape)
		}
	}

	batchSize := llmOut.Shape[0]
	seqLen := llmOut.Shape[1]

	// Create Adjacency Matrix: [batch, seq_len, seq_len]
	// Using Global Receptive Field with Distance Decay to solve "no context" issue.
	// Every token now sees every other token, but closer neighbors have higher weight.
	adjData := make([]float32, batchSize*seqLen*seqLen)

	for b := 0; b < batchSize; b++ {
		batchOffset := b * seqLen * seqLen
		for i := 0; i < seqLen; i++ {
			rowOffset := batchOffset + i*seqLen

			var rowSum float32
			for j := 0; j < seqLen; j++ {
				dist := float64(i - j)
				if dist < 0 {
					dist = -dist
				}

				// Weight = 1.0 for local (dist <= 2), 0.2 for global (dist > 2)
				// This allows global context flow while preserving local grammar.
				weight := float32(0.2)
				if dist <= 2 {
					weight = 1.0
				}
				adjData[rowOffset+j] = weight
				rowSum += weight
			}

			// Normalize row to sum to 1.0
			if rowSum > 0 {
				invSum := 1.0 / rowSum
				for j := 0; j < seqLen; j++ {
					adjData[rowOffset+j] *= invSum
				}
			}
		}
	}

	adj := tensor.NewTensor([]int{batchSize, seqLen, seqLen}, adjData, false)

	// 3. Run GNN Layer
	gnnOut, err := e.GNNLayer.Forward(llmOut, adj)
	if err != nil {
		return nil, fmt.Errorf("HybridLLMGNNEncoder GNN failed: %w", err)
	}

	// 4. Residual Connection: X = X + GNN(X)
	// This preserves the original token identity while adding neighborhood context.
	residualOut, err := llmOut.Add(gnnOut)
	if err != nil {
		// If shape mismatch (e.g. projection happened in GNN), fallback to GNN output
		log.Printf("⚠️ Residual connection failed (shape mismatch?): %v. Using GNN output only.", err)
		residualOut = gnnOut
	}

	// 5. Final Normalization for context stability
	normed, err := e.LayerNorm.Forward(residualOut)
	if err != nil {
		return residualOut, nil
	}

	// Store for backward
	e.llmOutput = llmOut
	e.gnnOutput = gnnOut
	e.adj = adj
	e.residualOut = normed

	return normed, nil
}

func (e *HybridLLMGNNEncoder) Backward(grad *tensor.Tensor) error {
	if e.llmOutput == nil {
		return fmt.Errorf("HybridLLMGNNEncoder.Backward: llmOutput is nil (forget to call Forward?)")
	}

	if grad == nil {
		return nil // Nothing to backprop
	}

	// 1. Backward through LayerNorm
	err := e.LayerNorm.Backward(grad)
	if err != nil {
		return fmt.Errorf("failed LayerNorm branch backward: %w", err)
	}

	gradToPropagate := e.LayerNorm.Input().Grad
	if gradToPropagate == nil {
		gradToPropagate = grad
	}

	// 2. Backward through GNN branch
	err = e.GNNLayer.Backward(gradToPropagate)
	if err != nil {
		return fmt.Errorf("failed GNN branch backward: %w", err)
	}

	// 3. Backward through Residual branch
	if e.llmOutput.RequiresGrad {
		if e.llmOutput.Grad == nil {
			e.llmOutput.Grad = tensor.NewTensor(e.llmOutput.Shape, make([]float32, len(e.llmOutput.Data)), false)
		}
		for i := range gradToPropagate.Data {
			if i < len(e.llmOutput.Grad.Data) {
				e.llmOutput.Grad.Data[i] += gradToPropagate.Data[i]
			}
		}
	}

	// 4. Backward through LLM Encoder
	return e.LLMEncoder.Backward(e.llmOutput.Grad)
}

func (e *HybridLLMGNNEncoder) ClearState() {
	e.llmOutput = nil
	e.gnnOutput = nil
	e.adj = nil
	e.residualOut = nil
	if e.GNNLayer != nil {
		e.GNNLayer.ClearState()
	}
	if e.LayerNorm != nil {
		e.LayerNorm.ClearState()
	}
	if e.LLMEncoder != nil {
		e.LLMEncoder.ClearState()
	}
}

func (e *HybridLLMGNNEncoder) Parameters() []*tensor.Tensor {
	params := e.LLMEncoder.Parameters()
	params = append(params, e.GNNLayer.Parameters()...)
	params = append(params, e.LayerNorm.Parameters()...)
	return params
}

func (e *HybridLLMGNNEncoder) Inputs() []*tensor.Tensor {
	return e.LLMEncoder.Inputs()
}

func (e *HybridLLMGNNEncoder) SetMode(training bool) {
	e.LLMEncoder.SetMode(training)
}

func (e *HybridLLMGNNEncoder) GetMoELayers() []*MoELayer {
	return e.LLMEncoder.GetMoELayers()
}

// SetGateTemperature updates the temperature for the inner LLM encoder.
func (e *HybridLLMGNNEncoder) SetGateTemperature(temp float32) {
	if e.LLMEncoder != nil {
		e.LLMEncoder.SetGateTemperature(temp)
	}
}

// ToGPU moves the parameters to the GPU.
func (e *HybridLLMGNNEncoder) ToGPU() {
	if e.LLMEncoder != nil {
		e.LLMEncoder.ToGPU()
	}
	if e.GNNLayer != nil {
		e.GNNLayer.ToGPU()
	}
	if e.LayerNorm != nil {
		e.LayerNorm.ToGPU()
	}
}

func (e *HybridLLMGNNEncoder) SyncParameters() error {
	if e.LLMEncoder != nil {
		if err := e.LLMEncoder.SyncParameters(); err != nil {
			return err
		}
	}
	// GNN and LayerNorm are currently CPU-resident or handle their own sync during ToGPU/Forward
	return nil
}

func (e *HybridLLMGNNEncoder) RepairArchitecture() {
	if e.LLMEncoder != nil {
		e.LLMEncoder.RepairArchitecture()
	}
}
