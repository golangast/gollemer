package train

import "math"

// ClipGradients applies global L2-norm clipping to a flat gradient slice.
// It preserves the direction of the gradient while capping its magnitude.
// This is the standard approach for MoE models because it avoids the
// "loud expert" problem where a single expert's large gradient overwhelms
// the rest of the network.
//
// Returns the raw norm before clipping and a bool indicating whether
// clipping was applied, useful for logging.
func ClipGradients(grads []float64, maxNorm float64) (rawNorm float64, clipped bool) {
	var sumSq float64
	for _, g := range grads {
		sumSq += g * g
	}

	rawNorm = math.Sqrt(sumSq)
	if rawNorm > maxNorm {
		scale := maxNorm / (rawNorm + 1e-6) // 1e-6 prevents div-by-zero
		for i := range grads {
			grads[i] *= scale
		}
		clipped = true
	}
	return rawNorm, clipped
}

// ClipParamGrads applies ClipGradients across a set of parameter gradient
// slices, treating them as a single unified gradient vector.
// This is the correct way to clip across the full model, not layer by layer.
func ClipParamGrads(paramGrads [][]float64, maxNorm float64) (rawNorm float64, clipped bool) {
	// First pass: compute global L2 norm
	var sumSq float64
	for _, grads := range paramGrads {
		for _, g := range grads {
			sumSq += g * g
		}
	}

	rawNorm = math.Sqrt(sumSq)
	if rawNorm > maxNorm {
		scale := maxNorm / (rawNorm + 1e-6)
		// Second pass: scale all grads uniformly
		for _, grads := range paramGrads {
			for i := range grads {
				grads[i] *= scale
			}
		}
		clipped = true
	}
	return rawNorm, clipped
}
