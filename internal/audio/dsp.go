package audio

import (
	"math"
)

// ComputeLogMelSpectrogram converts a raw PCM audio chunk into a feature vector
// optimized for Gollemer's TemporalEncoder (GRU).
func ComputeLogMelSpectrogram(pcm []int16) []float32 {
	// 1. Apply a Hanning Window to the raw samples to reduce edge leakage
	windowed := applyHanningWindow(pcm)

	// 2. Compute the power spectrum magnitudes (Simplified inline magnitude calculation)
	fftMagnitudes := naivePowerSpectrum(windowed)

	// 3. Compress the values logarithmically to mimic human hearing perception
	features := make([]float32, len(fftMagnitudes))
	for i, mag := range fftMagnitudes {
		features[i] = float32(math.Log(float64(mag) + 1e-5))
	}

	return features
}

func applyHanningWindow(samples []int16) []float64 {
	n := len(samples)
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		// Hanning window formula: 0.5 * (1 - cos(2*pi*i / (N-1)))
		window := 0.5 * (1.0 - math.Cos(2.0*math.Pi*float64(i)/float64(n-1)))
		out[i] = float64(samples[i]) * window
	}
	return out
}

func naivePowerSpectrum(input []float64) []float64 {
	// Placeholder for your native FFT transformation array loop
	// Returns a frequency coefficient bin array
	return input[:len(input)/2]
}
