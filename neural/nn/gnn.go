package nn

import (
	"encoding/gob"
	"fmt"
	"github.com/golangast/gollemer/neural/tensor"
)

func init() {
	gob.Register(&GCNLayer{})
}

// GCNLayer implements a Graph Convolutional Network layer.
// Output = A * X * W
type GCNLayer struct {
	Linear *Linear
	// Cache for backward pass
	adj       *tensor.Tensor
	linearOut *tensor.Tensor
}

func NewGCNLayer(inFeatures, outFeatures int) (*GCNLayer, error) {
	l, err := NewLinear(inFeatures, outFeatures)
	if err != nil {
		return nil, err
	}
	return &GCNLayer{Linear: l}, nil
}

func (l *GCNLayer) Parameters() []*tensor.Tensor {
	return l.Linear.Parameters()
}

func (l *GCNLayer) ClearState() {
	l.adj = nil
	l.linearOut = nil
	if l.Linear != nil {
		l.Linear.ClearState()
	}
}

// Forward performs the forward pass.
// inputs: [batch_size, num_nodes, in_features] or [num_nodes, in_features]
// adj: [num_nodes, num_nodes] (Adjacency matrix, usually normalized)
func (l *GCNLayer) Forward(inputs, adj *tensor.Tensor) (*tensor.Tensor, error) {
	// 1. Linear Transform: XW
	linearOut, err := l.Linear.Forward(inputs)
	if err != nil {
		return nil, fmt.Errorf("GCN linear forward failed: %w", err)
	}

	// 2. Aggregate: A * (XW)
	// MatMul expects [rows, cols] for 2D.
	// We need to handle batch dimension if present.
	// If inputs is 3D [b, n, f], linearOut is [b, n, f].
	// Adj is likely 2D [n, n] (shared) or 3D [b, n, n].
	
	// Let's assume input is 3D [batch, nodes, feats] and Adj is 3D [batch, nodes, nodes]
	// If Adj is 2D, we might need to broadcast or repeat it?
	// The tensor library `MatMul` handles batches if both are 4D (batch, heads, r, c).
	// It doesn't seem to natively support 3D batch matmul [b, n, m] * [b, m, p] based on my reading of `tensor.go`.
	
	// WORKAROUND: Process per batch element if 3D.
	// Or reshape to 4D with heads=1?
	
	var out *tensor.Tensor
	
	if len(linearOut.Shape) == 3 && len(adj.Shape) == 3 {
		// [b, n, f]
		batchSize := linearOut.Shape[0]
		nodes := linearOut.Shape[1]
		feats := linearOut.Shape[2]
		
		// Reshape to 4D: [b, 1, n, f]
		lin4D, err := linearOut.Reshape([]int{batchSize, 1, nodes, feats})
		if err != nil { return nil, err }
		
		// Adj 4D: [b, 1, n, n]
		adj4D, err := adj.Reshape([]int{batchSize, 1, nodes, nodes})
		if err != nil { return nil, err }
		
		out4D, err := adj4D.MatMul(lin4D)
		if err != nil { return nil, err }
		
		// Reshape back to 3D
		out, err = out4D.Reshape([]int{batchSize, nodes, l.Linear.Weights.Shape[1]})
		if err != nil { return nil, err }
		
	} else if len(linearOut.Shape) == 2 && len(adj.Shape) == 2 {
		var err error
		out, err = adj.MatMul(linearOut)
		if err != nil { return nil, err }
	} else {
		return nil, fmt.Errorf("GCN dimension mismatch/unsupported: adj %v, input %v", adj.Shape, inputs.Shape)
	}
	
	l.adj = adj
	l.linearOut = linearOut
	
	return out, nil
}

func (l *GCNLayer) Backward(grad *tensor.Tensor) error {
	// out = Adj * LinearOut
	// dL/dLinearOut = Adj^T * grad
	
	// Transpose Adj
	var adjT *tensor.Tensor
	var err error
	
	if len(l.adj.Shape) == 3 {
		adjT, err = l.adj.Transpose(1, 2)
	} else {
		adjT, err = l.adj.Transpose(0, 1)
	}
	if err != nil { return fmt.Errorf("GCN backward transpose failed: %w", err) }
	
	// Handle 3D/4D conversion for MatMul if needed
	var dLinearOut *tensor.Tensor
	if len(grad.Shape) == 3 {
		// reshape to 4D for matmul
		b, n, f := grad.Shape[0], grad.Shape[1], grad.Shape[2]
		grad4D, _ := grad.Reshape([]int{b, 1, n, f})
		adjT4D, _ := adjT.Reshape([]int{b, 1, n, n})
		
		res4D, err := adjT4D.MatMul(grad4D)
		if err != nil { return err }
		dLinearOut, _ = res4D.Reshape([]int{b, n, f})
	} else {
		dLinearOut, err = adjT.MatMul(grad)
		if err != nil { return err }
	}
	
	// Pass to Linear
	// We need to accumulate this into linearOut.Grad because other branches might have used it? 
	// Or we can just pass it directly to Backward if it's the only consumer.
	// Linear.Backward expects `l.input` to be set (it is) and `grad` to be dLoss/dOutput.
	
	return l.Linear.Backward(dLinearOut)
}
