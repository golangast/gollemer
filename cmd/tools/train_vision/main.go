package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"image"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/golangast/gollemer/internal/ai/moe"

	_ "image/jpeg"
	_ "image/png"
)

type VisionTrainingExample struct {
	ImagePath  string `json:"image_path"`
	Query      string `json:"query"`
	FlatOutput string `json:"flat_output"`
}

// ImageToPatches extracts 16x16 luma patches from an image
func ImageToPatches(img image.Image, patchSize int) [][]float32 {
	bounds := img.Bounds()
	width, height := bounds.Max.X, bounds.Max.Y

	numPatchesX := width / patchSize
	numPatchesY := height / patchSize
	numPatches := numPatchesX * numPatchesY

	pixelsPerPatch := patchSize * patchSize
	patches := make([][]float32, numPatches)

	patchIdx := 0
	for py := 0; py < numPatchesY; py++ {
		for px := 0; px < numPatchesX; px++ {
			patch := make([]float32, pixelsPerPatch)
			pIdx := 0
			for y := 0; y < patchSize; y++ {
				for x := 0; x < patchSize; x++ {
					imgX := (px * patchSize) + x
					imgY := (py * patchSize) + y
					r, g, b, _ := img.At(imgX, imgY).RGBA()

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

func main() {
	fmt.Println("Initializing Multimodal Vision Training...")

	// 1. Load the dataset
	file, err := os.Open("data/vision_dataset.json")
	if err != nil {
		log.Fatalf("Failed to open dataset: %v", err)
	}
	defer file.Close()

	bytesData, _ := io.ReadAll(file)
	var dataset []VisionTrainingExample
	json.Unmarshal(bytesData, &dataset)

	// 2. Initialize the VisionEncoder (PatchDim: 256, DModel: 512)
	visionEncoder := moe.NewVisionEncoder(256, 512)
	learningRate := float32(0.01)

	// 3. Training Loop
	epochs := 5
	for epoch := 1; epoch <= epochs; epoch++ {
		totalLoss := float32(0.0)

		for _, example := range dataset {
			// A. Load Image or Video Frame
			var img image.Image
			ext := filepath.Ext(example.ImagePath)
			if ext == ".webm" || ext == ".mp4" {
				cmd := exec.Command("ffmpeg", "-y", "-i", example.ImagePath, "-vframes", "1", "-f", "image2pipe", "-vcodec", "png", "-")
				out, err := cmd.Output()
				if err == nil {
					img, _, _ = image.Decode(bytes.NewReader(out))
				}
			} else {
				imgFile, err := os.Open(example.ImagePath)
				if err == nil {
					img, _, _ = image.Decode(imgFile)
					imgFile.Close()
				}
			}

			if img == nil {
				fmt.Println("Failed to load:", example.ImagePath)
				continue
			}

			// B. Extract Patches
			patches := ImageToPatches(img, 16)
			
			// C. Forward Pass
			tokens := visionEncoder.Forward(patches)

			// D. Dummy Loss Calculation (Normally handled by MoE)
			// Here we pretend the network wants the tokens to match a specific pattern to learn
			gradOut := make([][]float32, len(tokens))
			exampleLoss := float32(0.0)
			
			for i, token := range tokens {
				gradOut[i] = make([]float32, 512)
				for d := 0; d < 512; d++ {
					// Target is an arbitrary value, e.g., 0.5
					target := float32(0.5)
					errorDiff := token[d] - target
					
					// Mean Squared Error gradient
					gradOut[i][d] = errorDiff
					exampleLoss += errorDiff * errorDiff
				}
			}
			
			// E. Backward Pass!
			visionEncoder.Backward(gradOut, patches, learningRate)
			
			totalLoss += exampleLoss / float32(len(tokens)*512)
		}
		
		fmt.Printf("Epoch %d/%d - Average Vision Projection Loss: %.4f\n", epoch, epochs, totalLoss/float32(len(dataset)))
	}
	
	fmt.Println("Training complete! The VisionEncoder weights have been successfully updated.")
}
