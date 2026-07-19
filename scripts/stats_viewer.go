//go:build ignore
// +build ignore

package main

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

func main() {
	filePath := "logs/training.csv"
	file, err := os.Open(filePath)
	if err != nil {
		fmt.Println("⏳ Waiting for logs/training.csv...")
		return
	}
	defer file.Close()

	// Get file info for timing
	info, _ := file.Stat()
	startTime := info.ModTime() // Approximate start

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil || len(records) < 2 {
		fmt.Println("📊 Waiting for the first epoch to finish...")
		return
	}

	last := records[len(records)-1]
	epochStr, loss := last[0], last[1]
	currentEpoch, _ := strconv.Atoi(epochStr)
	totalEpochs := 100 // Change this to match your -overfit target

	// --- Timing Logic ---
	elapsed := time.Since(startTime)
	avgTimePerEpoch := elapsed.Seconds() / float64(currentEpoch)
	remainingEpochs := totalEpochs - currentEpoch
	etaSeconds := avgTimePerEpoch * float64(remainingEpochs)
	etaTime := time.Now().Add(time.Duration(etaSeconds) * time.Second)

	fmt.Printf("\n--- 📈 Gollemer Progress (Epoch %d/%d | Loss: %s) ---\n", currentEpoch, totalEpochs, loss)

	// ... (Expert Bar Drawing Logic) ...
    for i := 0; i < 8; i++ {
		val, _ := strconv.ParseFloat(last[i+3], 64)
		barLength := int(val * 50)
		if barLength > 50 { barLength = 50 }
		bar := strings.Repeat("█", barLength) + strings.Repeat("░", 50-barLength)
		fmt.Printf("Expert %d: [%s] %.2f%%\n", i, bar, val*100)
	}

	fmt.Println("--------------------------------------------------")
	if currentEpoch < totalEpochs {
		fmt.Printf("⏱️  Avg/Epoch: %.1fs | ETA: %s (%s remaining)\n", 
            avgTimePerEpoch, 
            etaTime.Format("03:04 PM"), 
            time.Duration(etaSeconds)*time.Second)
	} else {
		fmt.Println("🏁 Training Complete!")
	}
	fmt.Println("--------------------------------------------------\n")
}