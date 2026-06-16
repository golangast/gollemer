package main

import (
	"bytes"
	"fmt"
	"image"
	"math"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/golangast/gollemer/internal/ai/moe"

	_ "image/jpeg"
	_ "image/png"
)

// ─────────────────────────────────────────────────────────────────────────────
// Synthetic frame generators
// ─────────────────────────────────────────────────────────────────────────────

func GenerateCameraPanRightSample() [][]float32 {
	frames := make([][]float32, 4)
	objectCanvasX := 50
	cameraViewportX := 40
	for i := 0; i < 4; i++ {
		frame := make([]float32, 64*64)
		relativeX := objectCanvasX - cameraViewportX
		if relativeX >= 0 && relativeX < 64 {
			for y := 28; y < 36; y++ {
				frame[y*64+relativeX] = 1.0
			}
		}
		frames[i] = frame
		cameraViewportX -= 8
	}
	return frames
}

func GenerateCameraPanLeftSample() [][]float32 {
	frames := make([][]float32, 4)
	objectCanvasX := 50
	cameraViewportX := 20
	for i := 0; i < 4; i++ {
		frame := make([]float32, 64*64)
		relativeX := objectCanvasX - cameraViewportX
		if relativeX >= 0 && relativeX < 64 {
			for y := 28; y < 36; y++ {
				frame[y*64+relativeX] = 1.0
			}
		}
		frames[i] = frame
		cameraViewportX += 8
	}
	return frames
}

func GenerateStaticSample() [][]float32 {
	frames := make([][]float32, 4)
	for i := 0; i < 4; i++ {
		frame := make([]float32, 64*64)
		for y := 28; y < 36; y++ {
			frame[y*64+32] = 1.0
		}
		frames[i] = frame
	}
	return frames
}

// GenerateTiltUpSample: camera tilts up → object drifts downward in frame.
func GenerateTiltUpSample(numFrames int) [][]float32 {
	frames := make([][]float32, numFrames)
	posY := 10
	posX := 32
	for i := 0; i < numFrames; i++ {
		frame := make([]float32, 64*64)
		for y := posY; y < posY+8; y++ {
			for x := posX - 4; x < posX+4; x++ {
				if x >= 0 && x < 64 && y >= 0 && y < 64 {
					frame[y*64+x] = 1.0
				}
			}
		}
		frames[i] = frame
		posY += 8
	}
	return frames
}

// GenerateTiltDownSample: camera tilts down → object drifts upward in frame.
func GenerateTiltDownSample(numFrames int) [][]float32 {
	frames := make([][]float32, numFrames)
	posY := 46
	posX := 32
	for i := 0; i < numFrames; i++ {
		frame := make([]float32, 64*64)
		for y := posY; y < posY+8; y++ {
			for x := posX - 4; x < posX+4; x++ {
				if x >= 0 && x < 64 && y >= 0 && y < 64 {
					frame[y*64+x] = 1.0
				}
			}
		}
		frames[i] = frame
		posY -= 8
	}
	return frames
}

// GenerateZoomInSample: object expands symmetrically (variance grows, CoM stays centred).
func GenerateZoomInSample(numFrames int) [][]float32 {
	frames := make([][]float32, numFrames)
	boxSize := 6
	centerX, centerY := 32, 32
	for i := 0; i < numFrames; i++ {
		frame := make([]float32, 64*64)
		half := boxSize / 2
		for y := centerY - half; y < centerY+half; y++ {
			for x := centerX - half; x < centerX+half; x++ {
				if x >= 0 && x < 64 && y >= 0 && y < 64 {
					frame[y*64+x] = 1.0
				}
			}
		}
		frames[i] = frame
		boxSize += 6
	}
	return frames
}

// GenerateZoomOutSample: object shrinks symmetrically (variance falls, CoM stays centred).
func GenerateZoomOutSample(numFrames int) [][]float32 {
	frames := make([][]float32, numFrames)
	boxSize := 30
	centerX, centerY := 32, 32
	for i := 0; i < numFrames; i++ {
		frame := make([]float32, 64*64)
		half := boxSize / 2
		for y := centerY - half; y < centerY+half; y++ {
			for x := centerX - half; x < centerX+half; x++ {
				if x >= 0 && x < 64 && y >= 0 && y < 64 {
					frame[y*64+x] = 1.0
				}
			}
		}
		frames[i] = frame
		boxSize -= 6
	}
	return frames
}

// ─────────────────────────────────────────────────────────────────────────────
// Real file loaders
// ─────────────────────────────────────────────────────────────────────────────

// imageToLumaFlat converts any image.Image to a normalised luma float32 slice.
func imageToLumaFlat(img image.Image) []float32 {
	b := img.Bounds()
	w, h := b.Max.X-b.Min.X, b.Max.Y-b.Min.Y
	flat := make([]float32, w*h)
	idx := 0
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			r, g, bb, _ := img.At(x, y).RGBA()
			luma := 0.299*float32(r) + 0.587*float32(g) + 0.114*float32(bb)
			flat[idx] = luma / 65535.0
			idx++
		}
	}
	return flat
}

// imageSize returns width and height of an image.
func imageSize(img image.Image) (int, int) {
	b := img.Bounds()
	return b.Max.X - b.Min.X, b.Max.Y - b.Min.Y
}

// loadImageFrames decodes a static image file into a single-element frame slice.
func loadImageFrames(path string) ([][]float32, int, int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, 0, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		return nil, 0, 0, err
	}
	w, h := imageSize(img)
	return [][]float32{imageToLumaFlat(img)}, w, h, nil
}

// extractVideoFrames uses ffmpeg to pull numFrames evenly spaced frames from
// the video and returns them as luma float32 slices.
func extractVideoFrames(path string, numFrames int) ([][]float32, int, int, error) {
	frames := [][]float32{}
	var w, h int

	for i := 0; i < numFrames; i++ {
		// Calculate timestamp: spread frames evenly across duration
		// Use select_eq(n,…) via fps filter to pick the Nth frame efficiently.
		// Simpler: use -ss for seeking, -vframes 1.
		timestamp := fmt.Sprintf("%.3f", float64(i)/float64(numFrames)*5.0) // 5s video
		cmd := exec.Command("ffmpeg", "-y",
			"-ss", timestamp,
			"-i", path,
			"-vframes", "1",
			"-f", "image2pipe",
			"-vcodec", "png",
			"-",
		)
		out, err := cmd.Output()
		if err != nil {
			fmt.Printf("  Warning: could not extract frame %d from %s: %v\n", i, filepath.Base(path), err)
			continue
		}
		img, _, err := image.Decode(bytes.NewReader(out))
		if err != nil {
			continue
		}
		if w == 0 {
			w, h = imageSize(img)
		}
		frames = append(frames, imageToLumaFlat(img))
	}
	return frames, w, h, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// Frame feature extraction
// ─────────────────────────────────────────────────────────────────────────────

// frameToGeometricToken computes [CoM-x, CoM-y, var-x, var-y] from a luma frame.
func frameToGeometricToken(frame []float32, w, h int) []float32 {
	var sumX, sumY, total float32
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			v := frame[y*w+x]
			sumX += v * float32(x)
			sumY += v * float32(y)
			total += v
		}
	}
	if total == 0 {
		return make([]float32, 4)
	}
	comX := sumX / total / float32(w)
	comY := sumY / total / float32(h)
	var varX, varY float32
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			v := frame[y*w+x]
			dx := float32(x)/float32(w) - comX
			dy := float32(y)/float32(h) - comY
			varX += v * dx * dx
			varY += v * dy * dy
		}
	}
	varX /= total
	varY /= total
	return []float32{comX, comY, varX * 4, varY * 4}
}

// framesToTokens converts a slice of raw luma frames to geometric feature tokens.
func framesToTokens(frames [][]float32, w, h int) [][]float32 {
	tokens := make([][]float32, len(frames))
	for i, f := range frames {
		tokens[i] = frameToGeometricToken(f, w, h)
	}
	return tokens
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

func softmax(logits []float32) []float32 {
	out := make([]float32, len(logits))
	maxV := logits[0]
	for _, v := range logits {
		if v > maxV {
			maxV = v
		}
	}
	var sum float32
	for i, v := range logits {
		out[i] = float32(math.Exp(float64(v - maxV)))
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

func crossEntropyLoss(probs []float32, targetClass int) (float32, []float32) {
	loss := -float32(math.Log(float64(probs[targetClass]) + 1e-9))
	grad := make([]float32, len(probs))
	copy(grad, probs)
	grad[targetClass] -= 1.0
	return loss, grad
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

func main() {
	fmt.Println("Temporal Motion Training — Synthetic + Real Files")
	fmt.Println("Classes: 0=PAN_RIGHT  1=PAN_LEFT  2=STATIC  3=REAL_IMAGE  4=REAL_VIDEO")
	fmt.Println()

	featureDim := 4
	hiddenDim  := 32
	numClasses := 9

	te := moe.NewTemporalEncoder(featureDim, hiddenDim)

	headW     := make([]float32, numClasses*hiddenDim)
	headB     := make([]float32, numClasses)
	headGradW := make([]float32, numClasses*hiddenDim)
	headGradB := make([]float32, numClasses)
	limit := float32(math.Sqrt(1.0 / float64(hiddenDim)))
	for i := range headW {
		headW[i] = (float32(i%7) - 3.0) * limit * 0.1
	}

	// ── Build dataset ─────────────────────────────────────────────────────────
	type tokenisedSample struct {
		frameTokens [][]float32
		targetClass int
		label       string
	}
	var dataset []tokenisedSample

	// --- Synthetic samples ---
	synth := []struct {
		frames [][]float32
		class  int
		label  string
	}{
		{GenerateCameraPanRightSample(), 0, "PAN_RIGHT"},
		{GenerateCameraPanLeftSample(), 1, "PAN_LEFT"},
		{GenerateStaticSample(), 2, "STATIC"},
		{GenerateTiltUpSample(8), 5, "TILT_UP"},
		{GenerateTiltDownSample(8), 6, "TILT_DOWN"},
		{GenerateZoomInSample(8), 7, "ZOOM_IN"},
		{GenerateZoomOutSample(8), 8, "ZOOM_OUT"},
	}
	for _, s := range synth {
		toks := make([][]float32, len(s.frames))
		for i, f := range s.frames {
			toks[i] = frameToGeometricToken(f, 64, 64)
		}
		dataset = append(dataset, tokenisedSample{toks, s.class, s.label})
		fmt.Printf("Synthetic %-10s  CoM-x: ", s.label)
		for _, t := range toks {
			fmt.Printf("%.3f ", t[0])
		}
		fmt.Println()
	}

	// --- Real image (video/Gemini_Generated_Image_*.png) ---
	imgPath := "video/Gemini_Generated_Image_idbmrridbmrridbm.png"
	imgFrames, imgW, imgH, err := loadImageFrames(imgPath)
	if err != nil {
		fmt.Printf("Warning: could not load %s: %v\n", imgPath, err)
	} else {
		toks := framesToTokens(imgFrames, imgW, imgH)
		dataset = append(dataset, tokenisedSample{toks, 3, "REAL_IMAGE"})
		fmt.Printf("Real      REAL_IMAGE   CoM-x: %.3f  CoM-y: %.3f  (1 frame, %dx%d)\n",
			toks[0][0], toks[0][1], imgW, imgH)
	}

	// --- Real video (extract 8 frames) ---
	vidPath := "video/Screen recording 2026-06-14 11.15.37 AM.webm"
	fmt.Printf("Extracting 8 frames from %s...\n", filepath.Base(vidPath))
	vidFrames, vidW, vidH, err := extractVideoFrames(vidPath, 8)
	if err != nil || len(vidFrames) == 0 {
		fmt.Printf("Warning: could not extract video frames: %v\n", err)
	} else {
		toks := framesToTokens(vidFrames, vidW, vidH)
		dataset = append(dataset, tokenisedSample{toks, 4, "REAL_VIDEO"})
		fmt.Printf("Real      REAL_VIDEO   frames=%d  CoM-x trajectory: ", len(toks))
		for _, t := range toks {
			fmt.Printf("%.3f ", t[0])
		}
		fmt.Println()
	}

	fmt.Printf("\nDataset: %d samples total\n\n", len(dataset))

	// ── Training ─────────────────────────────────────────────────────────────
	epochs := 2000
	lr     := float32(0.005)

	for epoch := 1; epoch <= epochs; epoch++ {
		totalLoss := float32(0)

		for _, s := range dataset {
			motionVec := te.Forward(s.frameTokens)

			logits := make([]float32, numClasses)
			for c := 0; c < numClasses; c++ {
				for d := 0; d < hiddenDim; d++ {
					logits[c] += headW[c*hiddenDim+d] * motionVec[d]
				}
				logits[c] += headB[c]
			}

			probs := softmax(logits)
			loss, dLogits := crossEntropyLoss(probs, s.targetClass)
			totalLoss += loss

			for i := range headGradW { headGradW[i] = 0 }
			for i := range headGradB { headGradB[i] = 0 }
			dMotion := make([]float32, hiddenDim)
			for c := 0; c < numClasses; c++ {
				for d := 0; d < hiddenDim; d++ {
					headGradW[c*hiddenDim+d] = dLogits[c] * motionVec[d]
					dMotion[d] += headW[c*hiddenDim+d] * dLogits[c]
				}
				headGradB[c] = dLogits[c]
			}

			te.Backward(dMotion, lr)

			for i := range headW { headW[i] -= lr * headGradW[i] }
			for i := range headB { headB[i] -= lr * headGradB[i] }
		}

		if epoch%200 == 0 {
			fmt.Printf("Epoch %4d/%d  avg-loss=%.4f\n", epoch, epochs, totalLoss/float32(len(dataset)))
		}
	}

	// ── Evaluation ───────────────────────────────────────────────────────────
	fmt.Println()
	fmt.Println("--- Final Evaluation ---")
	classNames := []string{"PAN_RIGHT", "PAN_LEFT", "STATIC", "REAL_IMAGE", "REAL_VIDEO", "TILT_UP", "TILT_DOWN", "ZOOM_IN", "ZOOM_OUT"}
	correct := 0
	for _, s := range dataset {
		motionVec := te.Forward(s.frameTokens)
		logits := make([]float32, numClasses)
		for c := 0; c < numClasses; c++ {
			for d := 0; d < hiddenDim; d++ {
				logits[c] += headW[c*hiddenDim+d] * motionVec[d]
			}
			logits[c] += headB[c]
		}
		probs := softmax(logits)
		pred := 0
		for c := 1; c < numClasses; c++ {
			if probs[c] > probs[pred] {
				pred = c
			}
		}
		status := " X "
		if pred == s.targetClass {
			correct++
			status = " OK"
		}
		fmt.Printf("  [%s]  Ground-truth=%-12s  Predicted=%-12s  Confidence=%.1f%%\n",
			status, s.label, classNames[pred], probs[pred]*100)
	}
	fmt.Printf("\nAccuracy: %d/%d (%.0f%%)\n", correct, len(dataset), float64(correct)/float64(len(dataset))*100)
}
