package moe

import (
	"encoding/gob"
	"fmt"
	"log"
	"github.com/golangast/gollemer/neural/nn"
	"github.com/golangast/gollemer/neural/tensor"
)

func init() {
	gob.Register(&HybridLLMGNNEncoder{})
}

// HybridLLMGNNEncoder combines an LLM-based encoder (like MoE or BERT) with a GNN layer.
type HybridLLMGNNEncoder struct {
	LLMEncoder   Encoder // Can be MoELayer or simple Encoder
	GNNLayer     *nn.GCNLayer
	EmbeddingDim int

	// Intermediate tensors for backward
	llmOutput *tensor.Tensor
	gnnOutput *tensor.Tensor
	adj       *tensor.Tensor
}

func NewHybridLLMGNNEncoder(llmEncoder Encoder, embeddingDim int) (*HybridLLMGNNEncoder, error) {
	gnn, err := nn.NewGCNLayer(embeddingDim, embeddingDim)
	if err != nil {
		return nil, err
	}
	return &HybridLLMGNNEncoder{
		LLMEncoder:   llmEncoder,
		GNNLayer:     gnn,
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
	// A[i][j] = 1 if |i-j| <= 1 else 0 (Window size 1) = Chain graph
	adjData := make([]float64, batchSize*seqLen*seqLen)
	for b := range batchSize {
		for i := range seqLen {
			for j := range seqLen {
				// Self-loop + neighbors
				if i == j || i == j-1 || i == j+1 {
					// Normalize by degree (approx 3 or 2)
					degree := 3.0
					if i == 0 || i == seqLen-1 {
						degree = 2.0
					}
					if seqLen == 1 {
						degree = 1.0
					}
					// Row-normalize D^-1 A
					adjData[b*seqLen*seqLen+i*seqLen+j] = 1.0 / degree
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

	// Store for backward
	e.llmOutput = llmOut
	e.gnnOutput = gnnOut
	e.adj = adj

	return residualOut, nil
}

func (e *HybridLLMGNNEncoder) Backward(grad *tensor.Tensor) error {
	// grad is dL/dGNNOutput

	// Initialize input gradient if needed
	if e.llmOutput.Grad == nil {
		e.llmOutput.Grad = tensor.NewTensor(e.llmOutput.Shape, make([]float64, len(e.llmOutput.Data)), false)
	}

	// 1. Backward through GNN branch
	// This updates e.llmOutput.Grad via the inner GCNLayer
	err := e.GNNLayer.Backward(grad)
	if err != nil {
		return fmt.Errorf("failed GNN branch backward: %w", err)
	}

	// 2. Backward through Residual branch
	// d(X + GNN(X))/dX = 1 + dGNN(X)/dX
	// So we add the incoming gradient directly to llmOutput.Grad
	if e.llmOutput.RequiresGrad {
		if e.llmOutput.Grad == nil {
			e.llmOutput.Grad = tensor.NewTensor(e.llmOutput.Shape, make([]float64, len(e.llmOutput.Data)), false)
		}
		// Explicitly accumulate the residual gradient
		for i := range grad.Data {
			e.llmOutput.Grad.Data[i] += grad.Data[i]
		}
	}

	// 3. Backward through LLM Encoder using the total gradient on its output
	return e.LLMEncoder.Backward(e.llmOutput.Grad)
}

func (e *HybridLLMGNNEncoder) ClearState() {
	e.llmOutput = nil
	e.gnnOutput = nil
	e.adj = nil
	if moeEnc, ok := e.LLMEncoder.(*MoELayer); ok {
		moeEnc.ClearState()
	}
}


func (e *HybridLLMGNNEncoder) Parameters() []*tensor.Tensor {
	params := e.LLMEncoder.Parameters()
	params = append(params, e.GNNLayer.Parameters()...)
	return params
}

func (e *HybridLLMGNNEncoder) Inputs() []*tensor.Tensor {
	return e.LLMEncoder.Inputs()
}

func (e *HybridLLMGNNEncoder) SetMode(training bool) {
	e.LLMEncoder.SetMode(training)
}
