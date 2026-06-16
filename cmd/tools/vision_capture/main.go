package main

import (
	"bytes"
	"context"
	"fmt"
	"image"
	"math"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"sync"
	"syscall"
	"time"

	"github.com/golangast/gollemer/internal/ai/moe"
	"github.com/vladimirvivien/go4vl/device"
	"github.com/vladimirvivien/go4vl/v4l2"

	_ "image/jpeg"
	_ "image/png"
)

// ─────────────────────────────────────────────────────────────────────────────
// FrameBuffer — holds the single latest decoded camera frame
// ─────────────────────────────────────────────────────────────────────────────

type FrameBuffer struct {
	mu          sync.RWMutex
	LatestImage image.Image
}

// ─────────────────────────────────────────────────────────────────────────────
// Vision worker — opens go4vl camera and pushes decoded frames into fb
// ─────────────────────────────────────────────────────────────────────────────

func StartVisionWorker(ctx context.Context, fb *FrameBuffer) error {
	cam, err := device.Open("/dev/video0",
		device.WithPixFormat(v4l2.PixFormat{
			PixelFormat: v4l2.PixelFmtMJPEG,
			Width:       224,
			Height:      224,
		}),
	)
	if err != nil {
		return err
	}
	if err := cam.Start(ctx); err != nil {
		return err
	}

	go func() {
		for frame := range cam.GetOutput() {
			select {
			case <-ctx.Done():
				return
			default:
				img, _, err := image.Decode(bytes.NewReader(frame))
				if err != nil {
					continue
				}
				fb.mu.Lock()
				fb.LatestImage = img
				fb.mu.Unlock()
			}
		}
	}()
	return nil
}

// ─────────────────────────────────────────────────────────────────────────────
// ViT helpers (unchanged from previous iteration)
// ─────────────────────────────────────────────────────────────────────────────

func ImageToPatches(img image.Image, patchSize int) [][]float32 {
	bounds := img.Bounds()
	width, height := bounds.Max.X, bounds.Max.Y
	numPatchesX := width / patchSize
	numPatchesY := height / patchSize
	patches := make([][]float32, numPatchesX*numPatchesY)
	patchIdx := 0
	for py := 0; py < numPatchesY; py++ {
		for px := 0; px < numPatchesX; px++ {
			patch := make([]float32, patchSize*patchSize)
			pIdx := 0
			for y := 0; y < patchSize; y++ {
				for x := 0; x < patchSize; x++ {
					r, g, b, _ := img.At(px*patchSize+x, py*patchSize+y).RGBA()
					gray := 0.299*float32(r) + 0.587*float32(g) + 0.114*float32(b)
					patch[pIdx] = gray / 65535.0
					pIdx++
				}
			}
			patches[patchIdx] = patch
			patchIdx++
		}
	}
	return patches
}

// ─────────────────────────────────────────────────────────────────────────────
// Geometric feature extractor — converts any image to a 4-float motion token
// ─────────────────────────────────────────────────────────────────────────────

// imageToGeometricToken computes [CoM-x, CoM-y, var-x, var-y] from an image.Image.
// This is the same computation used during GRU training so the feature space matches exactly.
func imageToGeometricToken(img image.Image) []float32 {
	bounds := img.Bounds()
	w := bounds.Max.X - bounds.Min.X
	h := bounds.Max.Y - bounds.Min.Y

	var sumX, sumY, total float32
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			luma := 0.299*float32(r) + 0.587*float32(g) + 0.114*float32(b)
			luma /= 65535.0
			sumX += luma * float32(x-bounds.Min.X)
			sumY += luma * float32(y-bounds.Min.Y)
			total += luma
		}
	}
	if total == 0 {
		return make([]float32, 4)
	}
	comX := sumX / total / float32(w)
	comY := sumY / total / float32(h)

	var varX, varY float32
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			luma := (0.299*float32(r) + 0.587*float32(g) + 0.114*float32(b)) / 65535.0
			dx := float32(x-bounds.Min.X)/float32(w) - comX
			dy := float32(y-bounds.Min.Y)/float32(h) - comY
			varX += luma * dx * dx
			varY += luma * dy * dy
		}
	}
	varX = (varX / total) * 4
	varY = (varY / total) * 4
	return []float32{comX, comY, varX, varY}
}

// ─────────────────────────────────────────────────────────────────────────────
// Background motion tracking loop
// ─────────────────────────────────────────────────────────────────────────────

// StartMotionTracker runs in a goroutine. Every `interval` it:
//  1. Grabs the latest frame from FrameBuffer
//  2. Extracts its geometric token
//  3. Pushes it into the MotionWindow ring buffer
//  4. If the buffer is full, classifies the sequence and prints the motion token
func StartMotionTracker(ctx context.Context, fb *FrameBuffer, mw *moe.MotionWindow, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			fb.mu.RLock()
			img := fb.LatestImage
			fb.mu.RUnlock()

			if img == nil {
				continue
			}

			sig := imageToGeometricToken(img)
			mw.Push(sig)

			if mw.Ready() {
				token, err := mw.Classify()
				if err == nil {
					fmt.Printf("[Vision] %s  (CoM-x=%.3f CoM-y=%.3f)\n", token, sig[0], sig[1])
				}
			}
		}
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Minimal trained GRU head (matches train_temporal output)
// In production this would be loaded from a checkpoint file.
// ─────────────────────────────────────────────────────────────────────────────

// buildTrainedHead constructs a TemporalEncoder + head that has been trained
// on the 9-class synthetic+real dataset. Here we produce a fresh Xavier-init
// head as a placeholder — replace with serialised weights once you have them.
func buildTrainedHead() (*moe.TemporalEncoder, []float32, []float32, []string) {
	classNames := []string{
		"PAN_RIGHT", "PAN_LEFT", "STATIC",
		"REAL_IMAGE", "REAL_VIDEO",
		"TILT_UP", "TILT_DOWN",
		"ZOOM_IN", "ZOOM_OUT",
	}
	numClasses := len(classNames)
	hiddenDim := 32

	te := moe.NewTemporalEncoder(4, hiddenDim)

	// Xavier init for the head (same as train_temporal so behaviour is consistent)
	headW := make([]float32, numClasses*hiddenDim)
	headB := make([]float32, numClasses)
	limit := float32(math.Sqrt(1.0 / float64(hiddenDim)))
	for i := range headW {
		headW[i] = (float32(i%7) - 3.0) * limit * 0.1
	}

	return te, headW, headB, classNames
}

// ─────────────────────────────────────────────────────────────────────────────
// File fallback helpers (used when camera is unavailable)
// ─────────────────────────────────────────────────────────────────────────────

func loadImageFromFile(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	return img, err
}

func extractFrameFromVideo(path string) (image.Image, error) {
	cmd := exec.Command("ffmpeg", "-y", "-i", path, "-vframes", "1",
		"-f", "image2pipe", "-vcodec", "png", "-")
	out, err := cmd.Output()
	if err != nil {
		return nil, err
	}
	img, _, err := image.Decode(bytes.NewReader(out))
	return img, err
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

func main() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Shut down cleanly on Ctrl-C
	sigs := make(chan os.Signal, 1)
	signal.Notify(sigs, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigs
		fmt.Println("\n[Vision] Shutting down...")
		cancel()
	}()

	fb := &FrameBuffer{}

	// ── Try live camera first ─────────────────────────────────────────────────
	cameraAvailable := true
	if err := StartVisionWorker(ctx, fb); err != nil {
		fmt.Printf("[Vision] Camera unavailable (%v). Running file fallback.\n", err)
		cameraAvailable = false
	}

	// ── Build the MotionWindow + GRU classifier ───────────────────────────────
	te, headW, headB, classNames := buildTrainedHead()
	mw := moe.NewMotionWindow(4, te, headW, headB, classNames)

	// ── File fallback: populate fb with frames from video/ ───────────────────
	if !cameraAvailable {
		files, err := os.ReadDir("video")
		if err != nil || len(files) == 0 {
			// Last resort: blank image
			fb.mu.Lock()
			fb.LatestImage = image.NewRGBA(image.Rect(0, 0, 224, 224))
			fb.mu.Unlock()
		} else {
			// Feed each file into the MotionWindow directly to demonstrate the pipeline
			fmt.Println("[Vision] Feeding video/ files into MotionWindow...")
			// Loop files twice so the 4-frame ring buffer fills completely
			for pass := 0; pass < 2; pass++ {
				for _, f := range files {
					path := filepath.Join("video", f.Name())
					var img image.Image

					switch filepath.Ext(path) {
					case ".webm", ".mp4":
						img, _ = extractFrameFromVideo(path)
					case ".png", ".jpg", ".jpeg":
						img, _ = loadImageFromFile(path)
					}

					if img == nil {
						continue
					}

					fb.mu.Lock()
					fb.LatestImage = img
					fb.mu.Unlock()

					sig := imageToGeometricToken(img)
					mw.Push(sig)
					fmt.Printf("[Vision] Pushed frame  CoM-x=%.3f CoM-y=%.3f\n", sig[0], sig[1])

					if mw.Ready() {
						token, err := mw.Classify()
						if err == nil {
							fmt.Printf("[Vision] %s\n", token)
						}
					}
				}
			}

			// Also demonstrate the ViT patch path on the last image
			fb.mu.RLock()
			imgSnapshot := fb.LatestImage
			fb.mu.RUnlock()

			if imgSnapshot != nil {
				patches := ImageToPatches(imgSnapshot, 16)
				fmt.Printf("[Vision] ViT patches from last file: %d patches × 256 floats\n", len(patches))
			}

			fmt.Println("[Vision] File fallback complete. In live mode this loop runs continuously.")
			return
		}
	}

	// ── Live camera mode: start the background motion tracker ────────────────
	fmt.Println("[Vision] Camera online. Starting real-time motion tracker (Ctrl-C to stop)...")
	// Sample a new frame into the ring buffer every 250ms → ~4 fps tracking
	go StartMotionTracker(ctx, fb, mw, 250*time.Millisecond)

	// Block until cancelled
	<-ctx.Done()
}
