package moe

import (
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

// Expert is an interface for an expert network in a Mixture of Experts model.
type Expert interface {
	// Forward performs the forward pass of the expert network.
	// It takes a tensor of shape (batch_size, input_dim) and returns a tensor of shape (batch_size, output_dim).
	Forward(input *tensor.Tensor) (*tensor.Tensor, error)

	// Backward performs the backward pass of the expert network.
	Backward(grad *tensor.Tensor) error

	// Parameters returns all learnable parameters of the expert.
	Parameters() []*tensor.Tensor

	// Inputs returns the input tensors of the expert's last forward operation.
	Inputs() []*tensor.Tensor

	// Description returns a string description of the expert.
	Description() string

	// SetMode sets the mode of the expert (training or inference).
	SetMode(training bool)

	// ClearState clears the expert's internal states.
	ClearState()

	// ClipWeights bounds the expert's learnable parameters.
	ClipWeights(maxVal float32)

	// EvolutionaryReset re-initializes the expert based on a winner's weights (Genetic Mutation).
	EvolutionaryReset(winner Expert, jitterScale float32)

	// Shake performs an in-place noise injection to break loops.
	Shake(intensity float32)

	// IsStagnant returns true if the expert is not contributing significantly.
	IsStagnant() bool

	// UpdateHealth updates the expert's relevance metric (EMA).
	UpdateHealth(wasUsed bool)

	// ToGPU moves the expert's parameters to the GPU.
	ToGPU()

	// Resize updates the output dimension of the expert.
	Resize(newOutputDim int)

	// SyncParameters synchronizes parameters from CPU to GPU.
	SyncParameters() error

	// GetID returns the expert's unique ID.
	GetID() int

	// GetContext extracts the expert's temporal/hidden state for paging out.
	GetContext() []float32

	// RestoreContext re-inflates the expert's temporal/hidden state after paging in.
	RestoreContext(ctx []float32)
}
