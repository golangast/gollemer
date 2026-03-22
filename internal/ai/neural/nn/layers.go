package nn

import (
	"encoding/gob"
	"errors"
	"fmt"
	"math"
	"log"
	"math/rand"

	. "github.com/golangast/gollemer/internal/ai/neural/tensor"
)

// weightIterator efficiently iterates over a flat weight array as a 2D matrix.
type weightIterator struct {
	data      []float64
	inputDim  int
	outputDim int
	row       int
	col       int
	totalSize int
}

func (wi *weightIterator) Next() bool {
	wi.col++
	if wi.col >= wi.outputDim {
		wi.col = 0
		wi.row++
	}
	return wi.row < wi.inputDim
}

func (wi *weightIterator) GetIndex() int {
	return wi.row*wi.outputDim + wi.col
}

func newWeightIterator(data []float64, inputDim, outputDim int) *weightIterator {
	return &weightIterator{data: data, inputDim: inputDim, outputDim: outputDim, row: 0, col: -1, totalSize: inputDim * outputDim}
}

func init() {
	gob.Register(&Linear{})
	gob.Register(&MultiHeadCrossAttention{})

}

// safeAccumulate adds src to dst, ensuring we don't go out of bounds.
func safeAccumulate(dst, src []float64) {
	n := len(dst)
	if len(src) < n {
		n = len(src)
	}
	for i := 0; i < n; i++ {
		dst[i] += src[i]
	}
}

// Linear represents a linear layer (fully connected layer).
type Linear struct {
	Weights *Tensor
	Biases  *Tensor
	input   *Tensor // Store input for backward pass
}

// NewLinear creates a new Linear layer with random weights and zero biases.
func NewLinear(inputDim, outputDim int) (*Linear, error) {
	// He initialization
	stdDev := math.Sqrt(2.0 / float64(inputDim))
	weightsData := make([]float64, inputDim*outputDim)
	wi := newWeightIterator(weightsData, inputDim, outputDim)
	for wi.Next() {
		idx := wi.GetIndex()
		weightsData[idx] = rand.NormFloat64() * stdDev
	}

	weights := NewTensor([]int{inputDim, outputDim}, weightsData, true)
	weights.RequiresGrad = true

	// Biases are usually initialized to zero
	biasesData := make([]float64, outputDim)
	biases := NewTensor([]int{outputDim}, biasesData, true)
	biases.RequiresGrad = true

	return &Linear{Weights: weights, Biases: biases}, nil
}

func (l *Linear) UpdateWeightsUnrolled(updateFn func(idx int, val float64) float64, unroll int) {
	wi := newWeightIterator(l.Weights.Data, l.Weights.Shape[0], l.Weights.Shape[1])
	for wi.Next() {
		idx := wi.GetIndex()
		l.Weights.Data[idx] = updateFn(idx, l.Weights.Data[idx])
	}
}

// Parameters returns all learnable parameters of the layer.
func (l *Linear) Parameters() []*Tensor {
	params := []*Tensor{l.Weights}
	if l.Biases != nil {
		params = append(params, l.Biases)
	}
	return params
}

// Input returns the input tensor of the Linear operation.
func (l *Linear) Input() *Tensor {
	return l.input
}

// ClearState clears the intermediate states to free memory.
func (l *Linear) ClearState() {
	l.input = nil
}

// Forward performs the forward pass of the Linear layer.
func (l *Linear) Forward(inputs ...*Tensor) (*Tensor, error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("Linear.Forward expects 1 input, got %d", len(inputs))
	}
	input := inputs[0]
	if input == nil { // Add this check
		return nil, fmt.Errorf("Linear.Forward received a nil input tensor")
	}
	// Store input tensor for backward pass
	l.input = input

	// Assuming input is 2D [batch_size, input_dim] or 3D [batch_size, sequence_length, input_dim]
	var output *Tensor
	var err error

	switch len(input.Shape) {
	case 2:
		// Handle 2D input: [batch_size, input_dim]
		// Perform matrix multiplication: [batch_size, input_dim] @ [input_dim, output_dim]
		output, err = input.MatMul(l.Weights)
		if err != nil {
			return nil, fmt.Errorf("linear layer 2D matrix multiplication failed: %w", err)
		}

	case 3:
		// Handle 3D input: [batch_size, sequence_length, input_dim]
		batchSize := input.Shape[0]
		seqLength := input.Shape[1]
		inputDim := input.Shape[2]
		outputDim := l.Weights.Shape[1]

		// Reshape input for batch matrix multiplication without copying data
		reshapedInput, err := input.Reshape([]int{batchSize * seqLength, inputDim})
		if err != nil {
			return nil, fmt.Errorf("linear layer 3D reshape failed: %w", err)
		}

		// Perform matrix multiplication: [batch_size * sequence_length, input_dim] @ [input_dim, output_dim]
		output2D, err := reshapedInput.MatMul(l.Weights)
		if err != nil {
			return nil, fmt.Errorf("linear layer 3D matrix multiplication failed: %w", err)
		}

		// Reshape output back to 3D without copying data
		output, err = output2D.Reshape([]int{batchSize, seqLength, outputDim})
		if err != nil {
			return nil, fmt.Errorf("linear layer 3D output reshape failed: %w", err)
		}

	default:
		return nil, fmt.Errorf("linear layer only supports 2D or 3D input, got %d dimensions", len(input.Shape))
	}

	// Add biases if they exist
	if l.Biases != nil {
		// AddWithBroadcast handles broadcasting biases
		output, err = output.AddWithBroadcast(l.Biases)
		if err != nil {
			return nil, fmt.Errorf("linear layer bias addition failed: %w", err)
		}
	}

	// Set creator and RequiresGrad for the output tensor
	output.RequiresGrad = input.RequiresGrad || l.Weights.RequiresGrad || (l.Biases != nil && l.Biases.RequiresGrad)
	if output.RequiresGrad {
		output.Creator = l
	}

	return output, nil
}

// Backward performs the backward pass for the Linear layer.
// grad is the gradient from the output (dLoss/dOutput).
func (l *Linear) Backward(grad *Tensor) error {
	if grad == nil || grad.Data == nil {
		// No gradient to propagate
		return nil
	}
	if l.input == nil {
		return errors.New("linear layer backward called before forward (input is nil)")
	}

	// Ensure gradients are initialized for parameters that require them
	if l.Weights.RequiresGrad && l.Weights.Grad == nil {
		l.Weights.Grad = NewTensor(l.Weights.Shape, make([]float64, len(l.Weights.Data)), false)
	}
	if l.Biases != nil && l.Biases.RequiresGrad {
		if l.Biases.Grad == nil {
			l.Biases.Grad = NewTensor(l.Biases.Shape, make([]float64, len(l.Biases.Data)), false)
		}
	}

	// --- Calculate Gradient with respect to Weights (dLoss/dWeights) ---
	// dLoss/dWeights = Input^T @ dLoss/dOutput (grad)
	var inputTranspose *Tensor
	var err error

	switch len(l.input.Shape) {
	case 2:
		// Input is 2D [batch_size, input_dim]
		inputTranspose, err = l.input.Transpose(0, 1)
		if err != nil {
			return fmt.Errorf("linear layer backward: failed to transpose input (2D): %w", err)
		}
		if l.Weights.RequiresGrad {
			dWeights, err := inputTranspose.MatMul(grad)
			if err != nil {
				return fmt.Errorf("linear layer backward: failed to calculate dLoss/dWeights (2D): %w", err)
			}
			AddAccumulate(l.Weights.Grad.Data, dWeights.Data)
		}

	case 3:
		// Input is 3D [batch_size, sequence_length, input_dim]
		batchSize := l.input.Shape[0]
		seqLength := l.input.Shape[1]
		inputDim := l.input.Shape[2]
		outputDim := l.Weights.Shape[1]

		reshapedInput := NewTensor([]int{batchSize * seqLength, inputDim}, l.input.Data, false)
		// Safety check: ensure grad.Data has enough elements for the reshaped shape
		if len(grad.Data) < batchSize*seqLength*outputDim {
			return fmt.Errorf("linear layer backward (3D): grad data length %d is too small for shape [%d, %d]", len(grad.Data), batchSize*seqLength, outputDim)
		}
		reshapedGrad := NewTensor([]int{batchSize * seqLength, outputDim}, grad.Data, false)

		inputTranspose, err = reshapedInput.Transpose(0, 1)
		if err != nil {
			return fmt.Errorf("linear layer backward: failed to transpose reshaped input (3D): %w", err)
		}

		if l.Weights.RequiresGrad {
			dWeights, err := inputTranspose.MatMul(reshapedGrad)
			if err != nil {
				return fmt.Errorf("linear layer backward: failed to calculate dLoss/dWeights (3D): %w", err)
			}
			safeAccumulate(l.Weights.Grad.Data, dWeights.Data)
		}

	default:
		return fmt.Errorf("linear layer backward only supports 2D or 3D input, got %d dimensions", len(l.input.Shape))
	}

	// --- Calculate Gradient with respect to Bias (dLoss/dBias) ---
	if l.Biases != nil && l.Biases.RequiresGrad {
		outputDim := l.Biases.Shape[0]
		for i := 0; i < len(grad.Data); i += outputDim {
			end := i + outputDim
			if end > len(grad.Data) {
				end = len(grad.Data)
			}
			if i < end {
				chunk := grad.Data[i:end]
				for j, v := range chunk {
					if j < len(l.Biases.Grad.Data) {
						l.Biases.Grad.Data[j] += v
					}
				}
			}
		}
	}

	// --- Calculate Gradient with respect to Input (dLoss/dInput) ---
	if l.input.RequiresGrad {
		if l.input.Grad == nil {
			l.input.Grad = NewTensor(l.input.Shape, make([]float64, len(l.input.Data)), false)
		}

		weightsTranspose, err := l.Weights.Transpose(0, 1)
		if err != nil {
			return fmt.Errorf("linear layer backward: failed to transpose weights: %w", err)
		}

		var dInput *Tensor
		switch len(grad.Shape) {
		case 2:
			dInput, err = grad.MatMul(weightsTranspose)
			if err != nil {
				return fmt.Errorf("linear layer backward: failed to calculate dLoss/dInput (2D): %w", err)
			}
		case 3:
			batchSize := grad.Shape[0]
			seqLength := grad.Shape[1]
			outputDim := grad.Shape[2]
			inputDim := l.Weights.Shape[0]

			// Safety check: ensure grad.Data has enough elements for the reshaped shape
			if len(grad.Data) < batchSize*seqLength*outputDim {
				return fmt.Errorf("linear layer backward (3D res): grad data length %d is too small for shape [%d, %d]", len(grad.Data), batchSize*seqLength, outputDim)
			}
			reshapedGrad := NewTensor([]int{batchSize * seqLength, outputDim}, grad.Data, false)

			dInput2D, err := reshapedGrad.MatMul(weightsTranspose)
			if err != nil {
				return fmt.Errorf("linear layer backward: failed to calculate dLoss/dInput (3D): %w", err)
			}
			dInput = NewTensor([]int{batchSize, seqLength, inputDim}, dInput2D.Data, false)

		default:
			return fmt.Errorf("linear layer backward only supports 2D or 3D gradient, got %d dimensions", len(grad.Shape))
		}

		safeAccumulate(l.input.Grad.Data, dInput.Data)
	}

	return nil
}

// Inputs returns the input tensors of the Linear operation.
func (l *Linear) Inputs() []*Tensor {
	if l.input != nil {
		return []*Tensor{l.input}
	}
	return []*Tensor{}
}

// LayerNormalization represents a layer normalization layer.
type LayerNormalization struct {
	Gamma           *Tensor // Scale parameter
	Beta            *Tensor // Shift parameter
	Epsilon         float64 // Small value to prevent division by zero
	mean            *Tensor
	invStdDev       *Tensor // Inverse standard deviation (1 / sqrt(variance + epsilon))
	normalizedInput *Tensor // Input after normalization, before scaling and shifting
	inputTensor     *Tensor // Add this field
	inputShape      []int   // Store input shape for backward
}

// NewLayerNormalization creates a new LayerNormalization layer.
func NewLayerNormalization(dimModel int) *LayerNormalization {
	// Initialize gamma to ones and beta to zeros
	gammaData := make([]float64, dimModel)
	betaData := make([]float64, dimModel)
	for i := range gammaData {
		gammaData[i] = 1.0 // Initialize gamma to 1s
		betaData[i] = 0.0  // Initialize beta to 0s
	}
	gamma := NewTensor([]int{dimModel}, gammaData, true)
	beta := NewTensor([]int{dimModel}, betaData, true)

	return &LayerNormalization{
		Gamma:   gamma,
		Beta:    beta,
		Epsilon: 1e-5, // Small epsilon value
	}
}

// Parameters returns all learnable parameters of the layer.
func (l *LayerNormalization) Parameters() []*Tensor {
	return []*Tensor{l.Gamma, l.Beta}
}

// Backward performs the backward pass for layer normalization.
// grad is the gradient from the output (dLoss/dOutput).
func (l *LayerNormalization) Backward(grad *Tensor) error {
	if grad == nil || grad.Data == nil {
		// No gradient to propagate
		return nil
	}

	if l.normalizedInput == nil || l.invStdDev == nil || l.mean == nil {
		panic("LayerNormalization backward called before forward (intermediate values are nil)")
	}
	if l.Gamma == nil || l.Beta == nil {
		panic("LayerNormalization scale or bias is nil in backward")
	}

	lastDimSize := l.inputShape[len(l.inputShape)-1]
	numElementsToNormalize := len(l.normalizedInput.Data) / lastDimSize

	// Ensure gradients are initialized
	if l.inputTensor != nil && l.inputTensor.RequiresGrad {
		if l.inputTensor.Grad == nil {
			l.inputTensor.Grad = NewTensor(l.inputTensor.Shape, make([]float64, len(l.inputTensor.Data)), false)
		} else if len(l.inputTensor.Grad.Data) != len(l.inputTensor.Data) {
			// Fix for zombie/mismatched gradients from previous batches
			l.inputTensor.Grad = NewTensor(l.inputTensor.Shape, make([]float64, len(l.inputTensor.Data)), false)
		}
	}
	if l.Gamma.RequiresGrad {
		if l.Gamma.Grad == nil {
			l.Gamma.Grad = NewTensor(l.Gamma.Shape, make([]float64, len(l.Gamma.Data)), false)
		}
	}
	if l.Beta.RequiresGrad {
		if l.Beta.Grad == nil {
			l.Beta.Grad = NewTensor(l.Beta.Shape, make([]float64, len(l.Beta.Data)), false)
		}
	}

	// --- Calculate Gradient with respect to Beta (Bias) ---
	// dLoss/dBeta = Sum(grad) over all dims except the last one.
	if l.Beta.RequiresGrad {
		for i := 0; i < len(grad.Data); i++ {
			biasIndex := i % lastDimSize
			if biasIndex < len(l.Beta.Grad.Data) {
				l.Beta.Grad.Data[biasIndex] += grad.Data[i]
			}
		}
	}

	// --- Calculate Gradient with respect to Gamma (Scale) ---
	// dLoss/dGamma = Sum(grad * normalized_input) over all dims except the last one.
	if l.Gamma.RequiresGrad {
		if l.Gamma.Grad == nil {
			l.Gamma.Grad = NewTensor(l.Gamma.Shape, make([]float64, len(l.Gamma.Data)), false)
		}
		// Sum (grad * normalizedInput) over all dimensions except the last one
		for i := range numElementsToNormalize {
			for j := range lastDimSize {
				// Index in flattened data
				flatIndex := i*lastDimSize + j
				if flatIndex < len(grad.Data) && flatIndex < len(l.normalizedInput.Data) && j < len(l.Gamma.Grad.Data) {
					l.Gamma.Grad.Data[j] += grad.Data[flatIndex] * l.normalizedInput.Data[flatIndex]
				}
			}
		}
	}

	// Calculate gradient with respect to normalized input
	// dLoss/dNormalizedInput = grad * gamma (Scale)
	dLoss_dNormalizedInputData := make([]float64, len(grad.Data))
	for i := range numElementsToNormalize {
		for j := range lastDimSize {
			flatIndex := i*lastDimSize + j
			if flatIndex < len(grad.Data) && j < len(l.Gamma.Data) {
				dLoss_dNormalizedInputData[flatIndex] = grad.Data[flatIndex] * l.Gamma.Data[j]
			}
		}
	}

	// Propagate gradient backward through normalization (mean and variance)
	// This is the most complex part. The formula for the gradient with respect to the input 'x' is:
	// dLoss/dx = dLoss/dNormalizedInput * (1 / std_dev)
	//           + dLoss/dStdDev * (x - mean) / (std_dev^2 * N)
	//           + dLoss/dMean / N
	// where N is the size of the last dimension (lastDimSize).

	// dLoss/dStdDev = Sum(dLoss/dNormalizedInput * (x - mean)) over the last dimension.
	// dLoss/dMean = Sum(dLoss/dNormalizedInput * (-1 / std_dev)) over the last dimension.

	// Let's calculate dLoss/dStdDev and dLoss/dMean first.

	dLoss_dStdDevData := make([]float64, numElementsToNormalize) // Gradients for std dev of each feature set
	dLoss_dMeanData := make([]float64, numElementsToNormalize)   // Gradients for mean of each feature set

	for i := range numElementsToNormalize {
		sum_dL_dNorm_x_minus_mean := 0.0
		sum_dL_dNorm := 0.0 // Needed for dLoss/dMean calculation

		for j := range lastDimSize {
			flatIndex := i*lastDimSize + j
			if flatIndex < len(l.inputTensor.Data) && i < len(l.mean.Data) && flatIndex < len(dLoss_dNormalizedInputData) {
				x_minus_mean := l.inputTensor.Data[flatIndex] - l.mean.Data[i] // Assuming inputTensor is stored
				sum_dL_dNorm_x_minus_mean += dLoss_dNormalizedInputData[flatIndex] * x_minus_mean
				sum_dL_dNorm += dLoss_dNormalizedInputData[flatIndex]
			}
		}

		// dLoss/dStdDev
		stdDev := 1.0 / l.invStdDev.Data[i]
		if math.IsNaN(stdDev) || math.IsInf(stdDev, 0) {
		}
		dLoss_dStdDevData[i] = sum_dL_dNorm_x_minus_mean * (-1.0 / (stdDev * stdDev)) // Derivative of 1/std_dev is -1/std_dev^2
		dLoss_dMeanData[i] = sum_dL_dNorm * (-l.invStdDev.Data[i])

		if math.IsNaN(dLoss_dStdDevData[i]) || math.IsInf(dLoss_dStdDevData[i], 0) {
		}
		if math.IsNaN(dLoss_dMeanData[i]) || math.IsInf(dLoss_dMeanData[i], 0) {
		}
	}

	if l.inputTensor.RequiresGrad {
		if l.inputTensor.Grad == nil {
			l.inputTensor.Grad = NewTensor(l.inputTensor.Shape, make([]float64, len(l.inputTensor.Data)), false)
		}
		// Iterate over each feature vector (e.g., each token embedding in a sequence)
		for i := range numElementsToNormalize {
			// Pre-calculate sums for the current feature vector to avoid redundant computation.
			sum_dL_dNorm := 0.0
			sum_dL_dNorm_x_minus_mean := 0.0
			for k := range lastDimSize {
				flatIndex_k := i*lastDimSize + k
				if flatIndex_k < len(dLoss_dNormalizedInputData) && flatIndex_k < len(l.inputTensor.Data) && i < len(l.mean.Data) {
					sum_dL_dNorm += dLoss_dNormalizedInputData[flatIndex_k]
					sum_dL_dNorm_x_minus_mean += dLoss_dNormalizedInputData[flatIndex_k] * (l.inputTensor.Data[flatIndex_k] - l.mean.Data[i])
				}
			}

			invStdDev_i := l.invStdDev.Data[i]

			// Now, calculate the gradient for each element within the feature vector using the pre-calculated sums.
			for j := range lastDimSize {
				flatIndex := i*lastDimSize + j
				if flatIndex < len(l.inputTensor.Data) && i < len(l.mean.Data) && flatIndex < len(dLoss_dNormalizedInputData) && flatIndex < len(l.inputTensor.Grad.Data) {
					x_j_minus_mean_i := l.inputTensor.Data[flatIndex] - l.mean.Data[i]

					// Calculate dLoss/dx_j
					dL_dx_j := invStdDev_i * (dLoss_dNormalizedInputData[flatIndex] - sum_dL_dNorm/float64(lastDimSize) - x_j_minus_mean_i*invStdDev_i*invStdDev_i*sum_dL_dNorm_x_minus_mean/float64(lastDimSize))
					if math.IsNaN(dL_dx_j) || math.IsInf(dL_dx_j, 0) {
					}

					l.inputTensor.Grad.Data[flatIndex] += dL_dx_j // Accumulate gradient
				}
			}
		}
	}
	return nil
}

// Inputs returns the input tensors of the LayerNormalization operation.
// Assuming the input tensor is stored in the struct.
func (l *LayerNormalization) Inputs() []*Tensor {
	if l.inputTensor != nil {
		return []*Tensor{l.inputTensor}
	}
	return []*Tensor{} // Return empty slice if inputTensor is nil
}

// Forward performs the forward pass of layer normalization.
func (l *LayerNormalization) Forward(inputs ...*Tensor) (*Tensor, error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("LayerNormalization.Forward expects 1 input, got %d", len(inputs))
	}
	input := inputs[0]
	l.inputTensor = input      // Store input for backward pass
	l.inputShape = input.Shape // Store input shape

	for i, val := range input.Data {
		if math.IsNaN(val) || math.IsInf(val, 0) {
			return nil, fmt.Errorf("input to LayerNormalization contains NaN or Inf at index %d", i)
		}
	}

	l.inputTensor = input // Add this line
	if input == nil || input.Data == nil {
		return nil, errors.New("input tensor is nil or has no data")
	}
	if len(input.Shape) == 0 {
		return nil, errors.New("input tensor cannot be a scalar for layer normalization")
	}
	if l.Gamma == nil || l.Beta == nil {
		return nil, errors.New("layer normalization Gamma or Beta is nil")
	}
	if len(l.Gamma.Shape) != 1 || l.Gamma.Shape[0] != input.Shape[len(input.Shape)-1] ||
		len(l.Beta.Shape) != 1 || l.Beta.Shape[0] != input.Shape[len(input.Shape)-1] {
		return nil, fmt.Errorf("layer normalization Gamma/bias shape mismatch with input last dimension: %v vs %d", l.Gamma.Shape, input.Shape[len(input.Shape)-1])
	}

	// Store input shape for backward pass
	l.inputShape = input.Shape

	// Calculate mean and variance across the last dimension
	lastDimSize := input.Shape[len(input.Shape)-1]
	numElementsToNormalize := len(input.Data) / lastDimSize // Number of elements to calculate mean/variance over for each feature

	meanData := make([]float64, numElementsToNormalize)
	varianceData := make([]float64, numElementsToNormalize)
	normalizedInputData := make([]float64, len(input.Data))
	meanShape := make([]int, len(input.Shape)-1)
	copy(meanShape, input.Shape[:len(input.Shape)-1])

	// Create the Tensor structs for intermediate values
	l.mean = NewTensor(meanShape, meanData, false)                                     // Pass the data slice
	l.invStdDev = NewTensor(meanShape, make([]float64, numElementsToNormalize), false) // Create data slice here
	l.normalizedInput = NewTensor(input.Shape, normalizedInputData, false)             // Pass the data slice
	for i := range numElementsToNormalize {
		// Calculate mean
		sum := 0.0
		for j := range lastDimSize {
			sum += input.Data[i*lastDimSize+j]
		}
		l.mean.Data[i] = sum / float64(lastDimSize) // Store mean in the tensor's data

		// Calculate variance
		sumSqDiff := 0.0
		for j := range lastDimSize {
			diff := input.Data[i*lastDimSize+j] - l.mean.Data[i] // Use l.mean.Data
			sumSqDiff += diff * diff
		}
		varianceData[i] = sumSqDiff / float64(lastDimSize)

		// Calculate normalized input
		variance := math.Max(0, varianceData[i]) // Ensure variance is non-negative
		invStdDev := 1.0 / math.Sqrt(variance+l.Epsilon)
		l.invStdDev.Data[i] = invStdDev // Store inverse standard deviation in the tensor's data

		for j := range lastDimSize {
			l.normalizedInput.Data[i*lastDimSize+j] = (input.Data[i*lastDimSize+j] - l.mean.Data[i]) * invStdDev // Store normalized input
		}
	}

	// Scale and shift
	outputData := make([]float64, len(input.Data))
	for i := range numElementsToNormalize {
		for j := range lastDimSize {
			outputData[i*lastDimSize+j] = l.Gamma.Data[j]*l.normalizedInput.Data[i*lastDimSize+j] + l.Beta.Data[j] // Use l.normalizedInput.Data
		}
	}

	outputTensor := NewTensor(input.Shape, outputData, false)
	outputTensor.RequiresGrad = input.RequiresGrad || l.Gamma.RequiresGrad || l.Beta.RequiresGrad
	if outputTensor.RequiresGrad {
		outputTensor.Creator = l
	}

	return outputTensor, nil
}

// MultiHeadAttention represents a multi-head attention layer.
// MultiHeadAttention represents a multi-head attention layer.
type MultiHeadAttention struct {
	NumHeads        int
	DimModel        int
	HeadDim         int // dimModel / numHeads
	Depth           int
	attentionOutput *Tensor
	// Stored intermediate tensors for backward pass
	inputTensor                 *Tensor // Original input (Q, K, V are the same for self-attention)
	q, k, v                     *Tensor // Q, K, V after linear projection and splitting heads
	attentionScores             *Tensor // Q @ K^T
	attentionWeights            *Tensor // Softmax(attentionScores) + Mask
	attentionOutputBeforeConcat *Tensor // attentionWeights @ V (before concatenating heads)
	QueryLinear                 *Linear
	KeyLinear                   *Linear
	ValueLinear                 *Linear
	OutputLinear                *Linear
}

func NewMultiHeadAttention(dimModel, numHeads, numKVHeads int) (*MultiHeadAttention, error) {
	if dimModel%numHeads != 0 {
		return nil, fmt.Errorf("dimModel (%d) must be divisible by numHeads (%d)", dimModel, numHeads)
	}
	headDim := dimModel / numHeads

	queryLinear, err := NewLinear(dimModel, dimModel) // Output dim is dimModel (numHeads * headDim)
	if err != nil {
		return nil, fmt.Errorf("failed to create query linear layer: %w", err)
	}
	keyLinear, err := NewLinear(dimModel, dimModel) // Output dim is dimModel (numKVHeads * headDim in general, but here assuming numHeads == numKVHeads)
	if err != nil {
		return nil, fmt.Errorf("failed to create key linear layer: %w", err)
	}
	valueLinear, err := NewLinear(dimModel, dimModel) // Output dim is dimModel (numKVHeads * headDim in general, but here assuming numHeads == numKVHeads)
	if err != nil {
		return nil, fmt.Errorf("failed to create value linear layer: %w", err)
	}
	outputLinear, err := NewLinear(dimModel, dimModel) // Output dim is dimModel
	if err != nil {
		return nil, fmt.Errorf("failed to create output linear layer: %w", err)
	}

	return &MultiHeadAttention{
		NumHeads:     numHeads,
		DimModel:     dimModel,
		HeadDim:      headDim, // Store headDim
		Depth:        headDim, // Initialize Depth
		QueryLinear:  queryLinear,
		KeyLinear:    keyLinear,
		ValueLinear:  valueLinear,
		OutputLinear: outputLinear,
	}, nil
}

// Parameters returns all learnable parameters of the layer.
func (mha *MultiHeadAttention) Parameters() []*Tensor {
	params := mha.QueryLinear.Parameters()
	params = append(params, mha.KeyLinear.Parameters()...)
	params = append(params, mha.ValueLinear.Parameters()...)
	params = append(params, mha.OutputLinear.Parameters()...)
	return params
}

// ClearState clears the intermediate states to free memory.
func (mha *MultiHeadAttention) ClearState() {
	mha.attentionOutput = nil
	mha.inputTensor = nil
	mha.q = nil
	mha.k = nil
	mha.v = nil
	mha.attentionScores = nil
	mha.attentionWeights = nil
	mha.attentionOutputBeforeConcat = nil
	
	if mha.QueryLinear != nil { mha.QueryLinear.ClearState() }
	if mha.KeyLinear != nil { mha.KeyLinear.ClearState() }
	if mha.ValueLinear != nil { mha.ValueLinear.ClearState() }
	if mha.OutputLinear != nil { mha.OutputLinear.ClearState() }
}

// Backward performs the backward pass for multi-head self-attention.
// grad is the gradient from the output of the attention layer (after the final linear layer).
func (mha *MultiHeadAttention) Backward(grad *Tensor) error {
	if grad == nil || grad.Data == nil {
		return nil
	}

	if mha.attentionOutput == nil {
		return errors.New("mha.attentionOutput is nil in backward pass")
	}
	if mha.attentionOutput.Grad == nil {
		mha.attentionOutput.Grad = NewTensor(grad.Shape, make([]float64, len(grad.Data)), false)
	}
	safeAccumulate(mha.attentionOutput.Grad.Data, grad.Data)

	if mha.OutputLinear != nil {
		err := mha.OutputLinear.Backward(mha.attentionOutput.Grad)
		if err != nil {
			return err
		}
	}

	batchSize := mha.attentionOutput.Grad.Shape[0]
	seqLength := mha.attentionOutput.Grad.Shape[1]
	dimModel := mha.attentionOutput.Grad.Shape[2]
	numHeads := mha.NumHeads
	depth := mha.Depth

	// 1. Compute gradient w.r.t. input of OutputLinear:
	//    grad_context_reshaped = grad @ OutputLinear.Weights^T  ([b,s,dim] @ [dim,dim] -> [b,s,dim])
	//    Then reshape [b, s, dim_model] -> [b, s, num_heads, depth] -> [b, num_heads, s, depth]
	weightsT, wtErr := mha.OutputLinear.Weights.Transpose(0, 1)
	if wtErr != nil {
		return fmt.Errorf("MHA backward: transpose output weights failed: %w", wtErr)
	}
	// grad shape is [b, s, dim_model]; weightsT is [dim_model, dim_model]
	batchSeq := batchSize * seqLength
	gradFlat := NewTensor([]int{batchSeq, dimModel}, mha.attentionOutput.Grad.Data, false)
	gradContextFlat, matErr := gradFlat.MatMul(weightsT)
	if matErr != nil {
		return fmt.Errorf("MHA backward: grad @ OutputWeights^T failed: %w", matErr)
	}
	gradContext3D := NewTensor([]int{batchSize, seqLength, dimModel}, gradContextFlat.Data, false)
	gradReshaped, reshErr := gradContext3D.Reshape([]int{batchSize, seqLength, numHeads, depth})
	if reshErr != nil {
		return fmt.Errorf("MHA backward: reshape grad failed: %w", reshErr)
	}
	gradBeforeConcat, transpErr := gradReshaped.Transpose(1, 2)
	if transpErr != nil {
		return fmt.Errorf("MHA backward: transpose grad failed: %w", transpErr)
	}

	// 2. Backprop through MatMul(attentionWeights @ V)
	vTransposed, _ := mha.v.Transpose(2, 3)
	gradAttentionWeights, _ := gradBeforeConcat.MatMul(vTransposed)
	if len(gradAttentionWeights.Data) != len(mha.attentionWeights.Data) {
		return fmt.Errorf("MHA backward: gradAttentionWeights size mismatch: %d vs %d", len(gradAttentionWeights.Data), len(mha.attentionWeights.Data))
	}

	if mha.attentionWeights.RequiresGrad {
		if mha.attentionWeights.Grad == nil {
			mha.attentionWeights.Grad = NewTensor(mha.attentionWeights.Shape, make([]float64, len(mha.attentionWeights.Data)), false)
		}
		safeAccumulate(mha.attentionWeights.Grad.Data, gradAttentionWeights.Data)
	}

	attentionWeightsTransposed, _ := mha.attentionWeights.Transpose(2, 3)
	gradV_per_head, _ := attentionWeightsTransposed.MatMul(gradBeforeConcat)

	if mha.v.RequiresGrad {
		if mha.v.Grad == nil {
			mha.v.Grad = NewTensor(mha.v.Shape, make([]float64, len(mha.v.Data)), false)
		}
		safeAccumulate(mha.v.Grad.Data, gradV_per_head.Data)
	}

	// 3. Backprop through Softmax: dL/dS[i] = P[i]*(dL/dP[i] - dot(dL/dP, P))
	// Uses SIMD-accelerated SoftmaxBackwardRow from the tensor package.
	var gradAttentionScores *Tensor
	{
		attScoresShape := mha.attentionWeights.Shape // [b, h, s, s]
		b0 := attScoresShape[0]
		h0 := attScoresShape[1]
		s0 := attScoresShape[2]
		s1 := attScoresShape[3]
		gradScoresData := make([]float64, len(mha.attentionWeights.Data))
		for b := range b0 {
			for h := range h0 {
				for i := range s0 {
					base := (b*h0*s0+h*s0+i)*s1
					p := mha.attentionWeights.Data[base : base+s1]
					dp := gradAttentionWeights.Data[base : base+s1]
					out := gradScoresData[base : base+s1]
					SoftmaxBackwardRow(p, dp, out)
				}
			}
		}
		gradAttentionScores = NewTensor(attScoresShape, gradScoresData, false)
	}

	// 4. Backprop through scaling
	scale := 1.0 / math.Sqrt(float64(mha.HeadDim))
	scaledGradScoresData := make([]float64, len(gradAttentionScores.Data))
	MulScalar(gradAttentionScores.Data, scale, scaledGradScoresData)
	gradAttentionScores = NewTensor(gradAttentionScores.Shape, scaledGradScoresData, false)

	// 5. Backprop through MatMul(Q @ K^T)
	kForMatMul := mha.k
	// Fix for shape mismatch: grad [1, N, N] or [1, 1, N, N] vs K [1, 64]
	if (len(gradAttentionScores.Shape) == 3 || len(gradAttentionScores.Shape) == 4) && len(mha.k.Shape) == 2 {
		b := gradAttentionScores.Shape[0]
		var kSeq int
		if len(gradAttentionScores.Shape) == 3 {
			kSeq = gradAttentionScores.Shape[2]
		} else {
			kSeq = gradAttentionScores.Shape[3]
		}
		dim := mha.k.Shape[1]

		if kSeq > 1 {
			newData := make([]float64, b*kSeq*dim)
			for i := 0; i < b; i++ {
				srcStart := i * dim
				if srcStart+dim > len(mha.k.Data) {
					continue
				}
				src := mha.k.Data[srcStart : srcStart+dim] // [dim]
				for s := 0; s < kSeq; s++ {
					dstStart := (i*kSeq + s) * dim
					copy(newData[dstStart:dstStart+dim], src)
				}
			}
			
			if len(gradAttentionScores.Shape) == 4 {
				// Broadcast over heads: [b, h, kSeq, dim]
				h := gradAttentionScores.Shape[1]
				fullData := make([]float64, b*h*kSeq*dim)
				chunkSize := kSeq * dim
				for i := 0; i < b; i++ {
					srcChunk := newData[i*chunkSize : (i+1)*chunkSize]
					for j := 0; j < h; j++ {
						dstStart := (i*h + j) * chunkSize
						copy(fullData[dstStart:dstStart+chunkSize], srcChunk)
					}
				}
				kForMatMul = NewTensor([]int{b, h, kSeq, dim}, fullData, mha.k.RequiresGrad)
			} else {
				kForMatMul = NewTensor([]int{b, kSeq, dim}, newData, mha.k.RequiresGrad)
			}
		}
	}
	gradQ_per_head, err := gradAttentionScores.MatMul(kForMatMul)
	if err != nil {
		return fmt.Errorf("MHA backward: gradQ MatMul failed: %w", err)
	}
	if mha.q.RequiresGrad {
		if mha.q.Grad == nil {
			mha.q.Grad = NewTensor(mha.q.Shape, make([]float64, len(mha.q.Data)), false)
		}
		safeAccumulate(mha.q.Grad.Data, gradQ_per_head.Data)
	}

	gradScoresTransposed, _ := gradAttentionScores.Transpose(2, 3)
	
	qForMatMul := mha.q
	// Fix for shape mismatch: grad^T [1, N, N] or [1, 1, N, N] vs Q [1, 64]
	if (len(gradScoresTransposed.Shape) == 3 || len(gradScoresTransposed.Shape) == 4) && len(mha.q.Shape) == 2 {
		b := gradScoresTransposed.Shape[0]
		var qSeq int
		if len(gradScoresTransposed.Shape) == 3 {
			qSeq = gradScoresTransposed.Shape[2]
		} else {
			qSeq = gradScoresTransposed.Shape[3]
		}
		dim := mha.q.Shape[1]

		if qSeq > 1 {
			newData := make([]float64, b*qSeq*dim)
			for i := 0; i < b; i++ {
				srcStart := i * dim
				if srcStart+dim > len(mha.q.Data) {
					continue
				}
				src := mha.q.Data[srcStart : srcStart+dim] // [dim]
				for s := 0; s < qSeq; s++ {
					dstStart := (i*qSeq + s) * dim
					copy(newData[dstStart:dstStart+dim], src)
				}
			}
			
			if len(gradScoresTransposed.Shape) == 4 {
				// Broadcast over heads: [b, h, qSeq, dim]
				h := gradScoresTransposed.Shape[1]
				fullData := make([]float64, b*h*qSeq*dim)
				chunkSize := qSeq * dim
				for i := 0; i < b; i++ {
					srcChunk := newData[i*chunkSize : (i+1)*chunkSize]
					for j := 0; j < h; j++ {
						dstStart := (i*h + j) * chunkSize
						copy(fullData[dstStart:dstStart+chunkSize], srcChunk)
					}
				}
				qForMatMul = NewTensor([]int{b, h, qSeq, dim}, fullData, mha.q.RequiresGrad)
			} else {
				qForMatMul = NewTensor([]int{b, qSeq, dim}, newData, mha.q.RequiresGrad)
			}
		}
	}
	gradK_per_head, err := gradScoresTransposed.MatMul(qForMatMul)
	if err != nil {
		return fmt.Errorf("MHA backward: gradK MatMul failed: %w", err)
	}
	if mha.k.RequiresGrad {
		if mha.k.Grad == nil {
			mha.k.Grad = NewTensor(mha.k.Shape, make([]float64, len(mha.k.Data)), false)
		}
		safeAccumulate(mha.k.Grad.Data, gradK_per_head.Data)
	}

	// 6. Combine gradients and backprop to Linear layers
	// Combine Q gradients: [b, h, s, d] -> [b, s, h, d] -> [b, s, dim_model]
	if mha.q.Grad != nil {
		qGradTransposed, _ := mha.q.Grad.Transpose(1, 2)
		gradQCombined, _ := qGradTransposed.Reshape([]int{batchSize, seqLength, dimModel})
		err = mha.QueryLinear.Backward(gradQCombined)
		if err != nil {
			return err
		}
	}

	if mha.k.Grad != nil {
		kGradTransposed, _ := mha.k.Grad.Transpose(1, 2)
		gradKCombined, _ := kGradTransposed.Reshape([]int{batchSize, seqLength, dimModel})
		err = mha.KeyLinear.Backward(gradKCombined)
		if err != nil {
			return err
		}
	}

	if mha.v.Grad != nil {
		vGradTransposed, _ := mha.v.Grad.Transpose(1, 2)
		gradVCombined, _ := vGradTransposed.Reshape([]int{batchSize, seqLength, dimModel})
		err = mha.ValueLinear.Backward(gradVCombined)
		if err != nil {
			return err
		}
	}

	return nil
}

// Forward performs the forward pass of the MultiHeadAttention layer.
// This is a simplified version without caching or masks.
func (mha *MultiHeadAttention) Forward(inputs ...*Tensor) (*Tensor, error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("MultiHeadAttention.Forward expects 1 input, got %d", len(inputs))
	}
	value := inputs[0] // Extract the single input

	query := value
	key := value
	mask := value.Mask // Assuming mask is a field of the input tensor

	// Store the original input tensor (for self-attention, Q, K, V are the same)
	mha.inputTensor = query

	// Assume input shapes are [batch_size, sequence_length, dim_model]

	batchSize := query.Shape[0]
	qSeqLength := query.Shape[1]
	kvSeqLength := key.Shape[1] // Key and Value should have the same sequence length

	// Apply linear transformations to get Q, K, V
	q, err := mha.QueryLinear.Forward(query)
	if err != nil {
		return nil, fmt.Errorf("multihead attention query linear failed: %w", err)
	}
	k, err := mha.KeyLinear.Forward(key)
	if err != nil {
		return nil, fmt.Errorf("multihead attention key linear failed: %w", err)
	}
	v, err := mha.ValueLinear.Forward(value)
	if err != nil {
		return nil, fmt.Errorf("multihead attention value linear failed: %w", err)
	}

	// Reshape Q, K, V for multi-head attention
	// [batch_size, sequence_length, dim_model] -> [batch_size, num_heads, sequence_length, head_dim]
	qReshaped, err := q.Reshape([]int{batchSize, qSeqLength, mha.NumHeads, mha.HeadDim})
	if err != nil {
		return nil, fmt.Errorf("failed to reshape query tensor: %w", err)
	}

	qTransposed, err := qReshaped.Transpose(1, 2) // Transpose to [batch_size, num_heads, q_seq_length, head_dim]
	if err != nil {
		return nil, fmt.Errorf("failed to transpose query tensor: %w", err)
	}
	mha.q = qTransposed // Store q after transpose

	kReshaped, err := k.Reshape([]int{batchSize, kvSeqLength, mha.NumHeads, mha.HeadDim})
	if err != nil {
		return nil, fmt.Errorf("failed to reshape key tensor: %w", err)
	}

	kTransposed, err := kReshaped.Transpose(1, 2) // Transpose to [batch_size, num_heads, kv_seq_length, head_dim]
	if err != nil {
		return nil, fmt.Errorf("failed to transpose key tensor: %w", err)
	}
	mha.k = kTransposed // Store k after transpose

	vReshaped, err := v.Reshape([]int{batchSize, kvSeqLength, mha.NumHeads, mha.HeadDim})
	if err != nil {
		return nil, fmt.Errorf("failed to reshape value tensor: %w", err)
	}

	vTransposed, err := vReshaped.Transpose(1, 2) // Transpose to [batch_size, num_heads, kv_seq_length, head_dim]
	if err != nil {
		return nil, fmt.Errorf("failed to transpose value tensor: %w", err)
	}
	mha.v = vTransposed // Store v after transpose

	// Calculate attention scores: Q @ K^T
	// K^T will have shape [batch_size, num_heads, head_dim, kv_seq_length]
	kT_Transposed, err := kTransposed.Transpose(2, 3)
	if err != nil {
		return nil, fmt.Errorf("failed to transpose key tensor for multiplication: %w", err)
	}

	attentionScores, err := qTransposed.MatMul(kT_Transposed)
	if err != nil {
		return nil, fmt.Errorf("failed to calculate attention scores (Q@K^T): %w", err)
	}
	for i, val := range attentionScores.Data {
		if math.IsNaN(val) || math.IsInf(val, 0) {
			return nil, fmt.Errorf("attentionScores contains NaN or Inf at index %d", i)
		}
	}
	mha.attentionScores = attentionScores // Store attention scores

	// Gamma attention scores
	scale := 1.0 / math.Sqrt(float64(mha.HeadDim))
	scaledAttentionScores, err := attentionScores.MulScalar(scale)
	if err != nil {
		return nil, fmt.Errorf("failed to scale attention scores: %w", err)
	}
	for i, val := range scaledAttentionScores.Data {
		if math.IsNaN(val) || math.IsInf(val, 0) {
			return nil, fmt.Errorf("scaledAttentionScores contains NaN or Inf at index %d", i)
		}
	}

	// Apply mask (if provided) - Simplified, just add large negative to masked positions
	if mask != nil {
		maskedAttentionScores, err := scaledAttentionScores.AddWithBroadcast(mask)
		if err != nil {
			return nil, fmt.Errorf("failed to apply mask to attention scores: %w", err)
		}
		scaledAttentionScores = maskedAttentionScores
	}

	// Apply Softmax to get attention weights
	attentionWeights, err := scaledAttentionScores.Softmax(len(scaledAttentionScores.Shape) - 1) // Softmax along the last dimension
	if err != nil {
		return nil, fmt.Errorf("failed to apply softmax to attention scores: %w", err)
	}
	mha.attentionWeights = attentionWeights // Store attention weights

	// Apply attention weights to Value: Attention Weights @ V
	contextLayer, err := attentionWeights.MatMul(vTransposed)
	if err != nil {
		return nil, fmt.Errorf("failed to calculate context layer (Attention@V): %w", err)
	}

	mha.attentionOutputBeforeConcat = contextLayer // Store attention output before concatenation

	// Concatenate heads and apply final linear layer
	// [batch_size, num_heads, q_seq_length, head_dim] -> [batch_size, q_seq_length, num_heads * head_dim] (which is dim_model)
	contextLayerTransposed, err := contextLayer.Transpose(1, 2) // Transpose back to [batch_size, q_seq_length, num_heads, head_dim]
	if err != nil {
		return nil, fmt.Errorf("failed to transpose context layer: %w", err)
	}

	// Reshape to [batch_size, q_seq_length, dim_model]
	outputShape := []int{batchSize, qSeqLength, mha.DimModel}
	contextLayerReshaped, err := contextLayerTransposed.Reshape(outputShape)
	if err != nil {
		return nil, fmt.Errorf("failed to reshape context layer: %w", err)
	}

	// Apply the final linear layer to the reshaped context layer
	output, err := mha.OutputLinear.Forward(contextLayerReshaped) // Use contextLayerReshaped here
	if err != nil {
		return nil, fmt.Errorf("multihead attention output linear failed: %w", err)
	}

	// Set creator and RequiresGrad for the output tensor
	outputRequiresGrad := query.RequiresGrad || key.RequiresGrad || value.RequiresGrad ||
		mha.QueryLinear.Weights.RequiresGrad || (mha.QueryLinear.Biases != nil && mha.QueryLinear.Biases.RequiresGrad) ||
		mha.KeyLinear.Weights.RequiresGrad || (mha.KeyLinear.Biases != nil && mha.KeyLinear.Biases.RequiresGrad) ||
		mha.ValueLinear.Weights.RequiresGrad || (mha.ValueLinear.Biases != nil && mha.ValueLinear.Biases.RequiresGrad) ||
		mha.OutputLinear.Weights.RequiresGrad || (mha.OutputLinear.Biases != nil && mha.OutputLinear.Biases.RequiresGrad)

	output.RequiresGrad = outputRequiresGrad
	if output.RequiresGrad {
		output.Creator = mha // Set the creator to the MultiHeadAttention layer itself
	}

	mha.attentionOutput = output

	return output, nil // Return the output tensor and nil error
}

// Inputs returns the input tensors of the MultiHeadAttention operation.
func (mha *MultiHeadAttention) Inputs() []*Tensor {
	// For self-attention, the input is stored in inputTensor
	if mha.inputTensor != nil {
		return []*Tensor{mha.inputTensor}
	}
	// If no inputs are stored (e.g., before forward pass), return an empty slice
	return []*Tensor{}
}

// MultiHeadCrossAttention represents a multi-head cross-attention layer.
type MultiHeadCrossAttention struct {
	NumQHeads  int // Number of query heads
	NumKVHeads int // Number of key/value heads
	DimModel   int
	DimKVHeads int // Dimension per key/value head (head_dim for keys/values)
	Depth      int // Dimension per query head (head_dim for queries)

	// Stored intermediate tensors for backward pass
	queryTensor                 *Tensor // Original query input from decoder
	keyTensor                   *Tensor // Original key input from encoder
	valueTensor                 *Tensor // Original value input from encoder
	q, k, v                     *Tensor // Q, K, V after linear projection and splitting heads
	attentionScores             *Tensor // Q @ K^T
	attentionWeights            *Tensor // Softmax(attentionScores) + Mask
	attentionOutputBeforeConcat *Tensor // attentionWeights @ V (before concatenating heads)
	// Add the following field to store the final output of the layer
	attentionOutput *Tensor
	QueryLinear     *Linear
	KeyLinear       *Linear // Linear layer for keys from encoder output
	ValueLinear     *Linear // Linear layer for values from encoder output
	OutputLinear    *Linear
	Wo              *Linear // This seems to be a duplicate of OutputLinear based on its usage
}

func (m *MultiHeadCrossAttention) Query() *Tensor { return m.queryTensor }
func (m *MultiHeadCrossAttention) Key() *Tensor   { return m.keyTensor }
func (m *MultiHeadCrossAttention) Value() *Tensor { return m.valueTensor }

func (m *MultiHeadCrossAttention) SetInput(q *Tensor) {
	m.queryTensor = q
}

// NewMultiHeadCrossAttention creates a new MultiHeadCrossAttention layer.
func NewMultiHeadCrossAttention(dimModel, queryDim, kvDim, numQHeads, numKVHeads int) (*MultiHeadCrossAttention, error) {
	if dimModel%numQHeads != 0 {
		return nil, fmt.Errorf("dimModel (%d) must be divisible by numQHeads (%d)", dimModel, numQHeads)
	}
	if dimModel%numKVHeads != 0 {
		return nil, fmt.Errorf("dimModel (%d) must be divisible by numKVHeads (%d)", dimModel, numKVHeads)
	}
	dimKVHeads := dimModel / numKVHeads

	queryLinear, err := NewLinear(queryDim, dimModel) // Input dim is queryDim, output dim is dimModel
	if err != nil {
		return nil, fmt.Errorf("failed to create cross-attention query linear layer: %w", err)
	}
	keyLinear, err := NewLinear(kvDim, dimModel) // Input dim is kvDim, output dim is dimModel
	if err != nil {
		return nil, fmt.Errorf("failed to create cross-attention key linear layer: %w", err)
	}
	valueLinear, err := NewLinear(kvDim, dimModel) // Input dim is kvDim, output dim is dimModel
	if err != nil {
		return nil, fmt.Errorf("failed to create cross-attention value linear layer: %w", err)
	}
	outputLinear, err := NewLinear(dimModel, queryDim) // Input dim is dimModel, output is queryDim
	if err != nil {
		return nil, fmt.Errorf("failed to create cross-attention output linear layer: %w", err)
	}

	return &MultiHeadCrossAttention{
		NumQHeads:    numQHeads,
		NumKVHeads:   numKVHeads,
		DimModel:     dimModel,
		DimKVHeads:   dimKVHeads,           // Store dim per KV head
		Depth:        dimModel / numQHeads, // Initialize Depth
		QueryLinear:  queryLinear,
		KeyLinear:    keyLinear,
		ValueLinear:  valueLinear,
		OutputLinear: outputLinear,
	}, nil
}

// Parameters returns all learnable parameters of the layer.
func (mha *MultiHeadCrossAttention) Parameters() []*Tensor {
	params := mha.QueryLinear.Parameters()
	params = append(params, mha.KeyLinear.Parameters()...)
	params = append(params, mha.ValueLinear.Parameters()...)
	params = append(params, mha.OutputLinear.Parameters()...)
	return params
}

// ClearState clears the intermediate states to free memory.
func (mha *MultiHeadCrossAttention) ClearState() {
	mha.attentionOutput = nil
	mha.queryTensor = nil
	mha.keyTensor = nil
	mha.valueTensor = nil
	mha.q = nil
	mha.k = nil
	mha.v = nil
	mha.attentionScores = nil
	mha.attentionWeights = nil
	mha.attentionOutputBeforeConcat = nil
	
	if mha.QueryLinear != nil { mha.QueryLinear.ClearState() }
	if mha.KeyLinear != nil { mha.KeyLinear.ClearState() }
	if mha.ValueLinear != nil { mha.ValueLinear.ClearState() }
	if mha.OutputLinear != nil { mha.OutputLinear.ClearState() }
}

// Inputs returns the input tensors of the MultiHeadCrossAttention operation.
func (mha *MultiHeadCrossAttention) Inputs() []*Tensor {
	// Return the original input tensors: query, key, and value
	// The mask is not typically considered a tensor that requires gradients in the same way,
	// so it's usually not included in the Inputs for backpropagation.
	if mha.queryTensor != nil && mha.keyTensor != nil && mha.valueTensor != nil {
		return []*Tensor{mha.queryTensor, mha.keyTensor, mha.valueTensor}
	}
	// If inputs are not stored (e.g., before forward pass), return an empty slice
	return []*Tensor{}
}

func (mha *MultiHeadCrossAttention) Backward(grad *Tensor) error {
	if grad == nil || grad.Data == nil {
		return nil
	}

	if mha.attentionOutput == nil {
		return errors.New("mha.attentionOutput is nil in backward pass")
	}
	if mha.attentionOutput.Grad == nil {
		mha.attentionOutput.Grad = NewTensor(grad.Shape, make([]float64, len(grad.Data)), false)
	}
	if len(mha.attentionOutput.Grad.Data) != len(grad.Data) {
		return fmt.Errorf("MHCA backward: gradient data length mismatch: expected %d, got %d", len(mha.attentionOutput.Grad.Data), len(grad.Data))
	}
	safeAccumulate(mha.attentionOutput.Grad.Data, grad.Data)

	if mha.OutputLinear != nil {
		err := mha.OutputLinear.Backward(mha.attentionOutput.Grad)
		if err != nil {
			return err
		}
		// Since we added a residual (output + query), the gradient flows back to query too
		if mha.queryTensor != nil && mha.queryTensor.RequiresGrad {
			if mha.queryTensor.Grad == nil {
				mha.queryTensor.Grad = NewTensor(mha.queryTensor.Shape, make([]float64, len(mha.queryTensor.Data)), false)
			}
			safeAccumulate(mha.queryTensor.Grad.Data, mha.attentionOutput.Grad.Data)
		}
	}

	batchSize := mha.attentionOutput.Grad.Shape[0]
	querySeqLen := mha.attentionOutput.Grad.Shape[1]
	dimModel := mha.attentionOutput.Grad.Shape[2]
	numQHeads := mha.NumQHeads
	depth := mha.Depth

	// 1. Compute gradient w.r.t. input of OutputLinear:
	//    grad_context = grad @ OutputLinear.Weights^T  ([b,seq,dim] @ [dim,dim] -> [b,seq,dim])
	//    Then reshape to [b, num_q_heads, query_seq_len, depth]
	weightsT, wtErr := mha.OutputLinear.Weights.Transpose(0, 1)
	if wtErr != nil {
		return fmt.Errorf("MHCA backward: transpose output weights failed: %w", wtErr)
	}
	batchSeq := batchSize * querySeqLen
	gradFlat := NewTensor([]int{batchSeq, dimModel}, mha.attentionOutput.Grad.Data, false)
	gradContextFlat, matErr := gradFlat.MatMul(weightsT)
	if matErr != nil {
		return fmt.Errorf("MHCA backward: grad @ OutputWeights^T failed: %w", matErr)
	}
	gradContext3D := NewTensor([]int{batchSize, querySeqLen, dimModel}, gradContextFlat.Data, false)
	gradReshaped, reshErr := gradContext3D.Reshape([]int{batchSize, querySeqLen, numQHeads, depth})
	if reshErr != nil {
		return fmt.Errorf("MHCA backward: reshape grad failed: %w", reshErr)
	}
	gradBeforeConcat, transpErr := gradReshaped.Transpose(1, 2)
	if transpErr != nil {
		return fmt.Errorf("MHCA backward: transpose grad failed: %w", transpErr)
	}

	// 2. Backprop through MatMul(attentionWeights @ V)
	// mha.v is [batch, heads, seq_v, depth]
	vTransposed, _ := mha.v.Transpose(2, 3)
	gradAttentionWeights, _ := gradBeforeConcat.MatMul(vTransposed)
	if len(gradAttentionWeights.Data) != len(mha.attentionWeights.Data) {
		return fmt.Errorf("MHCA backward: gradAttentionWeights size mismatch: %d vs %d", len(gradAttentionWeights.Data), len(mha.attentionWeights.Data))
	}

	// Always store attentionWeights.Grad for the softmax backward below.
	if mha.attentionWeights.Grad == nil {
		mha.attentionWeights.Grad = NewTensor(mha.attentionWeights.Shape, make([]float64, len(mha.attentionWeights.Data)), false)
	}
	safeAccumulate(mha.attentionWeights.Grad.Data, gradAttentionWeights.Data)

	attentionWeightsTransposed, _ := mha.attentionWeights.Transpose(2, 3)
	gradV_per_head, _ := attentionWeightsTransposed.MatMul(gradBeforeConcat)

	// Always store gradV — required to propagate back to ValueLinear and context vector.
	if mha.v.Grad == nil {
		mha.v.Grad = NewTensor(mha.v.Shape, make([]float64, len(mha.v.Data)), false)
	}
	safeAccumulate(mha.v.Grad.Data, gradV_per_head.Data)

	// 3. Backprop through Softmax: dL/dS[i] = P[i]*(dL/dP[i] - dot(dL/dP, P))
	// Uses SIMD-accelerated SoftmaxBackwardRow.
	var gradAttentionScoresMHCA *Tensor
	{
		attScoresShape := mha.attentionWeights.Shape // [b, h, q_seq, kv_seq]
		b0 := attScoresShape[0]
		h0 := attScoresShape[1]
		s0 := attScoresShape[2]
		s1 := attScoresShape[3]
		gradScoresData := make([]float64, len(mha.attentionWeights.Data))
		for b := range b0 {
			for h := range h0 {
				for i := range s0 {
					base := (b*h0*s0+h*s0+i)*s1
					p := mha.attentionWeights.Data[base : base+s1]
					dp := gradAttentionWeights.Data[base : base+s1]
					out := gradScoresData[base : base+s1]
					SoftmaxBackwardRow(p, dp, out)
				}
			}
		}
		gradAttentionScoresMHCA = NewTensor(attScoresShape, gradScoresData, false)
	}

	// 4. Backprop through scaling
	scale := 1.0 / math.Sqrt(float64(mha.Depth))
	scaledGradScoresData := make([]float64, len(gradAttentionScoresMHCA.Data))
	MulScalar(gradAttentionScoresMHCA.Data, scale, scaledGradScoresData)
	gradAttentionScoresMHCA = NewTensor(gradAttentionScoresMHCA.Shape, scaledGradScoresData, false)

	// 5. Backprop through MatMul(Q @ K^T)
	kForMatMul := mha.k
	// Fix for shape mismatch: grad [1, N, N] or [1, 1, N, N] vs K [1, 64]
	if (len(gradAttentionScoresMHCA.Shape) == 3 || len(gradAttentionScoresMHCA.Shape) == 4) && len(mha.k.Shape) == 2 {
		b := gradAttentionScoresMHCA.Shape[0]
		var kvSeq int
		if len(gradAttentionScoresMHCA.Shape) == 3 {
			kvSeq = gradAttentionScoresMHCA.Shape[2]
		} else {
			kvSeq = gradAttentionScoresMHCA.Shape[3]
		}
		dim := mha.k.Shape[1]

		if kvSeq > 1 {
			// Broadcast K
			newData := make([]float64, b*kvSeq*dim)
			for i := 0; i < b; i++ {
				srcStart := i * dim
				if srcStart+dim > len(mha.k.Data) {
					continue
				}
				src := mha.k.Data[srcStart : srcStart+dim] // [dim]
				for s := 0; s < kvSeq; s++ {
					dstStart := (i*kvSeq + s) * dim
					copy(newData[dstStart:dstStart+dim], src)
				}
			}
			
			if len(gradAttentionScoresMHCA.Shape) == 4 {
				// Broadcast over heads: [b, h, kvSeq, dim]
				h := gradAttentionScoresMHCA.Shape[1]
				fullData := make([]float64, b*h*kvSeq*dim)
				chunkSize := kvSeq * dim
				for i := 0; i < b; i++ {
					srcChunk := newData[i*chunkSize : (i+1)*chunkSize]
					for j := 0; j < h; j++ {
						dstStart := (i*h + j) * chunkSize
						copy(fullData[dstStart:dstStart+chunkSize], srcChunk)
					}
				}
				kForMatMul = NewTensor([]int{b, h, kvSeq, dim}, fullData, mha.k.RequiresGrad)
			} else {
				kForMatMul = NewTensor([]int{b, kvSeq, dim}, newData, mha.k.RequiresGrad)
			}
		}
	}

	gradQ_per_head, err := gradAttentionScoresMHCA.MatMul(kForMatMul)
	if err != nil {
		return fmt.Errorf("MHCA backward: gradQ MatMul failed: %w", err)
	}
	// Always store gradQ — required to propagate back to QueryLinear and the decoder hidden state.
	if mha.q.Grad == nil {
		mha.q.Grad = NewTensor(mha.q.Shape, make([]float64, len(mha.q.Data)), false)
	}
	safeAccumulate(mha.q.Grad.Data, gradQ_per_head.Data)


	gradScoresMHCATransposed, _ := gradAttentionScoresMHCA.Transpose(2, 3)
	
	qForMatMul := mha.q
	// Fix for shape mismatch: grad^T [1, N, N] or [1, 1, N, N] vs Q [1, 64]
	if (len(gradScoresMHCATransposed.Shape) == 3 || len(gradScoresMHCATransposed.Shape) == 4) && len(mha.q.Shape) == 2 {
		b := gradScoresMHCATransposed.Shape[0]
		var qSeq int
		if len(gradScoresMHCATransposed.Shape) == 3 {
			qSeq = gradScoresMHCATransposed.Shape[2]
		} else {
			qSeq = gradScoresMHCATransposed.Shape[3]
		}
		dim := mha.q.Shape[1]

		if qSeq > 1 {
			newData := make([]float64, b*qSeq*dim)
			for i := 0; i < b; i++ {
				srcStart := i * dim
				if srcStart+dim > len(mha.q.Data) {
					continue
				}
				src := mha.q.Data[srcStart : srcStart+dim] // [dim]
				for s := 0; s < qSeq; s++ {
					dstStart := (i*qSeq + s) * dim
					copy(newData[dstStart:dstStart+dim], src)
				}
			}
			
			if len(gradScoresMHCATransposed.Shape) == 4 {
				// Broadcast over heads: [b, h, qSeq, dim]
				h := gradScoresMHCATransposed.Shape[1]
				fullData := make([]float64, b*h*qSeq*dim)
				chunkSize := qSeq * dim
				for i := 0; i < b; i++ {
					srcChunk := newData[i*chunkSize : (i+1)*chunkSize]
					for j := 0; j < h; j++ {
						dstStart := (i*h + j) * chunkSize
						copy(fullData[dstStart:dstStart+chunkSize], srcChunk)
					}
				}
				qForMatMul = NewTensor([]int{b, h, qSeq, dim}, fullData, mha.q.RequiresGrad)
			} else {
				qForMatMul = NewTensor([]int{b, qSeq, dim}, newData, mha.q.RequiresGrad)
			}
		}
	}
	gradK_per_head, err := gradScoresMHCATransposed.MatMul(qForMatMul)
	if err != nil {
		return fmt.Errorf("MHCA backward: gradK MatMul failed: %w", err)
	}
	// Always store gradK — required to propagate back to KeyLinear and context vector.
	if mha.k.Grad == nil {
		mha.k.Grad = NewTensor(mha.k.Shape, make([]float64, len(mha.k.Data)), false)
	}
	safeAccumulate(mha.k.Grad.Data, gradK_per_head.Data)

	// 6. Combine gradients, backprop to Linear layers, and then propagate to
	//    the original keyTensor and valueTensor (the encoder context vector).
	qGradTransposed, _ := mha.q.Grad.Transpose(1, 2)
	gradQCombined, _ := qGradTransposed.Reshape([]int{batchSize, querySeqLen, dimModel})
	if err = mha.QueryLinear.Backward(gradQCombined); err != nil {
		return err
	}
	// Propagate query grad back to queryTensor (decoder hidden state).
	if mha.queryTensor != nil {
		if mha.queryTensor.Grad == nil {
			mha.queryTensor.Grad = NewTensor(mha.queryTensor.Shape, make([]float64, len(mha.queryTensor.Data)), false)
		}
		if mha.QueryLinear.Input() != nil && mha.QueryLinear.Input().Grad != nil {
			qlg := mha.QueryLinear.Input().Grad
			if len(qlg.Data) == len(mha.queryTensor.Grad.Data) {
				safeAccumulate(mha.queryTensor.Grad.Data, qlg.Data)
			}
		}
	}

	kvSeqLen := mha.k.Shape[2]
	kGradTransposed, _ := mha.k.Grad.Transpose(1, 2)
	gradKCombined, _ := kGradTransposed.Reshape([]int{batchSize, kvSeqLen, dimModel})
	if err = mha.KeyLinear.Backward(gradKCombined); err != nil {
		return err
	}
	// Propagate key grad back to keyTensor (encoder context vector).
	if mha.keyTensor != nil {
		if mha.keyTensor.Grad == nil {
			mha.keyTensor.Grad = NewTensor(mha.keyTensor.Shape, make([]float64, len(mha.keyTensor.Data)), false)
		}
		if mha.KeyLinear.Input() != nil && mha.KeyLinear.Input().Grad != nil {
			klg := mha.KeyLinear.Input().Grad
			if len(klg.Data) == len(mha.keyTensor.Grad.Data) {
				safeAccumulate(mha.keyTensor.Grad.Data, klg.Data)
			}
		}
	}

	vkvSeqLen := mha.v.Shape[2]
	vGradTransposed, _ := mha.v.Grad.Transpose(1, 2)
	gradVCombined, _ := vGradTransposed.Reshape([]int{batchSize, vkvSeqLen, dimModel})
	if err = mha.ValueLinear.Backward(gradVCombined); err != nil {
		return err
	}
	// Propagate value grad back to valueTensor (encoder context vector).
	if mha.valueTensor != nil {
		if mha.valueTensor.Grad == nil {
			mha.valueTensor.Grad = NewTensor(mha.valueTensor.Shape, make([]float64, len(mha.valueTensor.Data)), false)
		}
		if mha.ValueLinear.Input() != nil && mha.ValueLinear.Input().Grad != nil {
			vlg := mha.ValueLinear.Input().Grad
			if len(vlg.Data) == len(mha.valueTensor.Grad.Data) {
				safeAccumulate(mha.valueTensor.Grad.Data, vlg.Data)
			}
		}
	}

	return nil
}

// Forward performs the forward pass of the MultiHeadCrossAttention layer.
// query: Input from the decoder layer (shape: [batch_size, target_sequence_length, dim_model]).
// key/value: Input from the encoder output (shape: [batch_size, source_sequence_length, dim_model]).
// mask: Optional mask for attention (e.g., padding mask for encoder output).
func (mha *MultiHeadCrossAttention) Forward(inputs ...*Tensor) (*Tensor, error) { // Changed to accept variadic inputs
	// Expect 3 or 4 inputs (query, key, value, and optional mask)
	if len(inputs) < 3 || len(inputs) > 4 {
		return nil, fmt.Errorf("MultiHeadCrossAttention.Forward expects 3 or 4 inputs (query, key, value, optional mask), got %d", len(inputs))
	}

	query := inputs[0]
	key := inputs[1]
	value := inputs[2]
	var mask *Tensor // Declare mask as optional

	if len(inputs) == 4 {
		mask = inputs[3] // Extract mask if provided
	}

	// Store the original input tensors
	mha.queryTensor = query
	mha.keyTensor = key
	mha.valueTensor = value

	// ... (rest of your forward pass logic, which uses query, key, value, and mask) ...

	batchSize := query.Shape[0]
	qSeqLength := query.Shape[1]
	kvSeqLength := key.Shape[1] // Key and Value should have the same sequence length

	// Apply linear transformations to get Q, K, V
	q, err := mha.QueryLinear.Forward(query) // Q from decoder input
	if err != nil {
		return nil, fmt.Errorf("cross-attention query linear failed: %w", err)
	}
	k, err := mha.KeyLinear.Forward(key) // K from encoder output
	if err != nil {
		return nil, fmt.Errorf("cross-attention key linear failed: %w", err)
	}
	v, err := mha.ValueLinear.Forward(value) // V from encoder output
	if err != nil {
		return nil, fmt.Errorf("cross-attention value linear failed: %w", err)
	}

	// Reshape Q, K, V for multi-head attention
	// Q shape: [batch_size, num_q_heads, q_seq_length, head_dim]
	qReshaped, err := q.Reshape([]int{batchSize, qSeqLength, mha.NumQHeads, mha.DimModel / mha.NumQHeads}) // Use DimModel/NumQHeads
	if err != nil {
		return nil, fmt.Errorf("failed to reshape cross-attention query tensor: %w", err)
	}

	qTransposed, err := qReshaped.Transpose(1, 2) // Transpose to [batch_size, num_heads, q_seq_length, head_dim]
	if err != nil {
		return nil, fmt.Errorf("failed to transpose cross-attention query tensor: %w", err)
	}
	mha.q = qTransposed

	// K, V shapes: [batch_size, num_kv_heads, kv_seq_length, head_dim]
	kReshaped, err := k.Reshape([]int{batchSize, kvSeqLength, mha.NumKVHeads, mha.DimKVHeads}) // Use DimKVHeads
	if err != nil {
		return nil, fmt.Errorf("failed to reshape cross-attention key tensor: %w", err)
	}
	kTransposed, err := kReshaped.Transpose(1, 2) // Transpose to [batch_size, num_heads, kv_seq_length, head_dim]
	if err != nil {
		return nil, fmt.Errorf("failed to transpose cross-attention key tensor: %w", err)
	}
	mha.k = kTransposed

	vReshaped, err := v.Reshape([]int{batchSize, kvSeqLength, mha.NumKVHeads, mha.DimKVHeads}) // Use DimKVHeads
	if err != nil {
		return nil, fmt.Errorf("failed to reshape cross-attention value tensor: %w", err)
	}
	vTransposed, err := vReshaped.Transpose(1, 2) // Transpose to [batch_size, num_heads, kv_seq_length, head_dim]
	if err != nil {
		return nil, fmt.Errorf("failed to transpose cross-attention value tensor: %w", err)
	}
	mha.v = vTransposed

	// Calculate attention scores: Q @ K^T
	// K^T will have shape [batch_size, num_heads, head_dim, kv_seq_length]
	kT_Transposed, err := kTransposed.Transpose(len(kTransposed.Shape)-2, len(kTransposed.Shape)-1)
	if err != nil {
		return nil, fmt.Errorf("failed to transpose key tensor for cross-attention multiplication: %w", err)
	}

	// MatMul Q [batch, num_q_heads, q_seq_len, head_dim] @ K^T [batch, num_kv_heads, head_dim, kv_seq_len]
	// This requires broadcasting or repeating K^T along the num_q_heads dimension if num_q_heads != num_kv_heads.
	// Assuming num_q_heads == num_kv_heads for simplicity in this simplified model.
	if mha.NumQHeads != mha.NumKVHeads {
		return nil, errors.New("cross-attention simplified model requires NumQHeads == NumKVHeads")
	}

	attentionScores, err := qTransposed.MatMul(kT_Transposed)
	if err != nil {
		return nil, fmt.Errorf("failed to calculate cross-attention scores (Q@K^T): %w", err)
	}
	mha.attentionScores = attentionScores // Store attention scores

	// Scale attention scores
	scale := 1.0 / math.Sqrt(float64(mha.DimModel/mha.NumQHeads)) // Scale by sqrt of head dim
	scaledAttentionScores, err := attentionScores.MulScalar(scale)
	if err != nil {
		return nil, fmt.Errorf("failed to scale cross-attention scores: %w", err)
	}

	// Apply mask (if provided) - Optimized for common attention mask shape [Batch, 1, 1, KVSeq]
	if mask != nil {
		if len(mask.Shape) == 4 && mask.Shape[1] == 1 && mask.Shape[2] == 1 && mask.Shape[0] == scaledAttentionScores.Shape[0] && mask.Shape[3] == scaledAttentionScores.Shape[3] {
			// Fast path for [B, 1, 1, S] broadcast
			resultData := make([]float64, len(scaledAttentionScores.Data))
			kvSeq := scaledAttentionScores.Shape[3]
			qSeq := scaledAttentionScores.Shape[2]
			heads := scaledAttentionScores.Shape[1]
			batchSize := scaledAttentionScores.Shape[0]
			
			for b := 0; b < batchSize; b++ {
				mOffset := b * kvSeq
				for h := 0; h < heads; h++ {
					for q := 0; q < qSeq; q++ {
						offset := ((b * heads + h) * qSeq + q) * kvSeq
						for k := 0; k < kvSeq; k++ {
							resultData[offset+k] = scaledAttentionScores.Data[offset+k] + mask.Data[mOffset+k]
						}
					}
				}
			}
			maskedScores := NewTensor(scaledAttentionScores.Shape, resultData, scaledAttentionScores.RequiresGrad || mask.RequiresGrad)
			if maskedScores.RequiresGrad {
				maskedScores.Creator = &AddWithBroadcastOperation{scaledAttentionScores, mask}
			}
			scaledAttentionScores = maskedScores
		} else {
			// Fallback to generic broadcast
			maskedAttentionScores, err := scaledAttentionScores.AddWithBroadcast(mask)
			if err != nil {
				return nil, fmt.Errorf("failed to apply mask to cross-attention scores: %w", err)
			}
			scaledAttentionScores = maskedAttentionScores
		}
	}

	// Apply Softmax to get attention weights
	attentionWeights, err := scaledAttentionScores.Softmax(len(scaledAttentionScores.Shape) - 1) // Softmax along the last dimension
	if err != nil {
		return nil, fmt.Errorf("failed to apply softmax to attention scores: %w", err)
	}
	mha.attentionWeights = attentionWeights // Store attention weights

	// Apply attention weights to Value: Attention Weights @ V
	contextLayer, err := attentionWeights.MatMul(vTransposed)
	if err != nil {
		return nil, fmt.Errorf("failed to calculate context layer (Attention@V): %w", err)
	}

	mha.attentionOutputBeforeConcat = contextLayer // Store attention output before concatenation

	// Concatenate heads and apply final linear layer
	// [batch_size, num_heads, q_seq_length, head_dim] -> [batch_size, q_seq_length, num_heads * head_dim] (which is dim_model)
	contextLayerTransposed, err := contextLayer.Transpose(1, 2) // Transpose back to [batch_size, q_seq_length, num_heads, head_dim]
	if err != nil {
		return nil, fmt.Errorf("failed to transpose context layer: %w", err)
	}

	// Reshape to [batch_size, q_seq_length, dim_model]
	outputShape := []int{batchSize, qSeqLength, mha.DimModel}
	contextLayerReshaped, err := contextLayerTransposed.Reshape(outputShape)
	if err != nil {
		return nil, fmt.Errorf("failed to reshape cross-attention context layer: %w", err)
	}

	// Apply output linear layer
	output, err := mha.OutputLinear.Forward(contextLayerReshaped)
	if err != nil {
		return nil, fmt.Errorf("cross-attention output linear failed: %w", err)
	}

	// NEW: Residual Connection
	output, err = output.Add(query)
	if err != nil {
		log.Printf("⚠️ MHCA Residual connection failed: %v", err)
	}

	// Store the final output tensor
	mha.attentionOutput = output // Store the final output

	// Set creator and RequiresGrad for the output tensor
	outputRequiresGrad := query.RequiresGrad || key.RequiresGrad || value.RequiresGrad ||
		mha.QueryLinear.Weights.RequiresGrad || (mha.QueryLinear.Biases != nil && mha.QueryLinear.Biases.RequiresGrad) ||
		mha.KeyLinear.Weights.RequiresGrad || (mha.KeyLinear.Biases != nil && mha.KeyLinear.Biases.RequiresGrad) ||
		mha.ValueLinear.Weights.RequiresGrad || (mha.ValueLinear.Biases != nil && mha.ValueLinear.Biases.RequiresGrad) ||
		mha.OutputLinear.Weights.RequiresGrad || (mha.OutputLinear.Biases != nil && mha.OutputLinear.Biases.RequiresGrad)

	output.RequiresGrad = outputRequiresGrad
	if output.RequiresGrad {
		output.Creator = mha // Set the creator to the MultiHeadCrossAttention layer itself
	}

	mha.attentionOutput = output

	return output, nil
}
