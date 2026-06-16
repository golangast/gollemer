package moe

import (
	"fmt"
	"sync"
)

// MotionWindow is a thread-safe sliding ring buffer that holds the N most recent
// per-frame geometric feature vectors (CoM-x, CoM-y, var-x, var-y).
//
// A background goroutine pushes one signature per frame from the camera;
// any NLP or voice thread can call Classify() at any moment to get an instant
// [MOTION: …] token without blocking the conversational pipeline.
type MotionWindow struct {
	mu         sync.RWMutex
	Capacity   int
	Signatures [][]float32 // ring buffer of feature vectors
	Head       int         // index where the NEXT write will go
	filled     int         // number of valid entries (saturates at Capacity)

	// Classifier wired in at construction time
	encoder    *TemporalEncoder
	headW      []float32 // [numClasses × hiddenDim] classification head
	headB      []float32 // [numClasses]
	ClassNames []string
}

// NewMotionWindow creates a MotionWindow backed by the given TemporalEncoder and
// a pre-trained linear classification head (headW, headB).
// classNames must be ordered to match the class indices used during training.
func NewMotionWindow(capacity int, te *TemporalEncoder, headW, headB []float32, classNames []string) *MotionWindow {
	return &MotionWindow{
		Capacity:   capacity,
		Signatures: make([][]float32, capacity),
		encoder:    te,
		headW:      headW,
		headB:      headB,
		ClassNames: classNames,
	}
}

// NewMotionWindowSimple creates an untrained MotionWindow (no classifier).
// Useful for collecting frames before a trained head is available.
func NewMotionWindowSimple(capacity int) *MotionWindow {
	return &MotionWindow{
		Capacity:   capacity,
		Signatures: make([][]float32, capacity),
	}
}

// Push appends a new frame's geometric feature vector, evicting the oldest.
// Safe to call from any goroutine.
func (mw *MotionWindow) Push(sig []float32) {
	mw.mu.Lock()
	defer mw.mu.Unlock()
	mw.Signatures[mw.Head] = sig
	mw.Head = (mw.Head + 1) % mw.Capacity
	if mw.filled < mw.Capacity {
		mw.filled++
	}
}

// Ready returns true once the buffer has received at least Capacity frames.
func (mw *MotionWindow) Ready() bool {
	mw.mu.RLock()
	defer mw.mu.RUnlock()
	return mw.filled >= mw.Capacity
}

// GetOrderedSequence returns the buffered frames in strict chronological order
// (oldest → newest), suitable for direct input to TemporalEncoder.Forward().
func (mw *MotionWindow) GetOrderedSequence() [][]float32 {
	mw.mu.RLock()
	defer mw.mu.RUnlock()
	seq := make([][]float32, mw.Capacity)
	for i := 0; i < mw.Capacity; i++ {
		idx := (mw.Head + i) % mw.Capacity
		seq[i] = mw.Signatures[idx]
	}
	return seq
}

// Classify runs a forward pass through the GRU and the linear head and returns
// a human-readable motion token string, e.g. "[MOTION: TILT_UP]".
// Returns an empty string and an error if the buffer is not yet full or no
// classifier has been wired in.
func (mw *MotionWindow) Classify() (string, error) {
	if !mw.Ready() {
		return "", fmt.Errorf("motion window not yet full (%d/%d frames)", mw.filled, mw.Capacity)
	}
	if mw.encoder == nil || len(mw.headW) == 0 {
		return "", fmt.Errorf("no classifier wired into MotionWindow")
	}

	seq := mw.GetOrderedSequence()

	// GRU forward pass
	motionVec := mw.encoder.Forward(seq)

	numClasses := len(mw.ClassNames)
	hiddenDim := mw.encoder.HiddenDim
	logits := make([]float32, numClasses)
	for c := 0; c < numClasses; c++ {
		for d := 0; d < hiddenDim; d++ {
			logits[c] += mw.headW[c*hiddenDim+d] * motionVec[d]
		}
		logits[c] += mw.headB[c]
	}

	// Argmax (no softmax needed for classification)
	best := 0
	for c := 1; c < numClasses; c++ {
		if logits[c] > logits[best] {
			best = c
		}
	}

	return fmt.Sprintf("[MOTION: %s]", mw.ClassNames[best]), nil
}
