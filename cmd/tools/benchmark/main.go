package main

import (
	"fmt"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/golangast/gollemer/internal/ai/moe"
)

func main() {
	fmt.Println("📊 Gollemer MoE Engine Performance & Benchmarking Suite")
	fmt.Println("================================================================================")

	benchmarkRAMOverhead()
	fmt.Println("--------------------------------------------------------------------------------")

	benchmarkInferenceThroughput()
	fmt.Println("--------------------------------------------------------------------------------")

	benchmarkSelfHealingSuccessRate()
	fmt.Println("================================================================================")
}

func getMemoryAllocMB() float64 {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return float64(m.Alloc) / (1024 * 1024)
}

func benchmarkRAMOverhead() {
	fmt.Println("1. RAM Overhead Measurement (Base vs 1 Expert vs 3 Blended Experts)")

	runtime.GC()
	baseRAM := getMemoryAllocMB()
	fmt.Printf("   - Baseline Memory Footprint:           %.2f MB\n", baseRAM)

	cm := moe.NewCartridgeManager()
	cartDir := "data/models/intents"

	// 1 Expert
	_ = cm.LoadCartridge(filepath.Join(cartDir, "sql_builder.cartridge"), 0, 0)
	runtime.GC()
	oneExpertRAM := getMemoryAllocMB()
	fmt.Printf("   - Memory Footprint (1 Expert loaded):  %.2f MB (+%.2f MB)\n", oneExpertRAM, oneExpertRAM-baseRAM)

	// 3 Experts
	_ = cm.LoadCartridge(filepath.Join(cartDir, "goroutine_fix.cartridge"), 0, 0)
	_ = cm.LoadCartridge(filepath.Join(cartDir, "gin_router.cartridge"), 0, 0)
	runtime.GC()
	threeExpertsRAM := getMemoryAllocMB()
	fmt.Printf("   - Memory Footprint (3 Experts loaded): %.2f MB (+%.2f MB over base)\n", threeExpertsRAM, threeExpertsRAM-baseRAM)
}

func benchmarkInferenceThroughput() {
	fmt.Println("2. Inference Throughput & AST Template Generation Speed")

	prompts := []string{
		"add handler auth_handler with database connection",
		"goroutine channel deadlock fix with unit test",
		"create gin router middleware with gorm model",
		"scaffold protobuf grpc service for user",
	}

	iterations := 20
	start := time.Now()

	for i := 0; i < iterations; i++ {
		p := prompts[i%len(prompts)]
		cmd := exec.Command("go", "run", "./cmd/tools/moe_inference", "-prompt", p)
		_ = cmd.Run()
	}

	elapsed := time.Since(start)
	avgLatencyMs := elapsed.Seconds() * 1000.0 / float64(iterations)
	tokensPerSec := float64(iterations*32) / elapsed.Seconds()

	fmt.Printf("   - Total Iterations Benchmark:          %d runs in %.2fs\n", iterations, elapsed.Seconds())
	fmt.Printf("   - Average AST Generation Speed:        %.2f ms/query\n", avgLatencyMs)
	fmt.Printf("   - Simulated Tokens/Sec Throughput:     %.2f tokens/sec\n", tokensPerSec)
}

func benchmarkSelfHealingSuccessRate() {
	fmt.Println("3. Self-Healing Success Rate Batch Evaluation")

	testPrompts := []string{
		"add handler auth_handler with database connection",
		"goroutine channel deadlock fix with unit test",
		"create gin router handler",
		"add gorm struct user with crud",
	}

	_ = os.MkdirAll("handlers", 0755)
	tmpTarget := "./handlers/bench_handler.go"
	defer os.Remove(tmpTarget)

	initialCode := `package handlers

import "net/http"

func dummy_handler(w http.ResponseWriter, r *http.Request) {}
`

	passAttempt1 := 0
	passAttempt2 := 0
	passAttempt3 := 0
	failures := 0

	for i, prompt := range testPrompts {
		_ = os.WriteFile(tmpTarget, []byte(initialCode), 0644)

		fmt.Printf("   - [Eval %d/%d] Prompt: %q ... ", i+1, len(testPrompts), prompt)

		// Run pipeline via gollemer patch
		cmd := exec.Command("./bin/gollemer", "patch", prompt, "-target="+tmpTarget)
		outBytes, err := cmd.CombinedOutput()
		outStr := string(outBytes)

		if err == nil {
			if strings.Contains(outStr, "Self-Healing Attempt 1") {
				passAttempt2++
				fmt.Println("Passed (Self-Healing Attempt 1)")
			} else if strings.Contains(outStr, "Self-Healing Attempt 2") {
				passAttempt3++
				fmt.Println("Passed (Self-Healing Attempt 2)")
			} else {
				passAttempt1++
				fmt.Println("Passed (Attempt 0 - First Pass)")
			}
		} else {
			failures++
			fmt.Printf("Failed (%v)\n", err)
		}
	}

	total := len(testPrompts)
	successCount := passAttempt1 + passAttempt2 + passAttempt3
	successRate := float64(successCount) / float64(total) * 100.0

	fmt.Printf("\n   Summary:\n")
	fmt.Printf("   - Pass 1st Attempt: %d (%.1f%%)\n", passAttempt1, float64(passAttempt1)/float64(total)*100)
	fmt.Printf("   - Pass 2nd Attempt: %d (%.1f%%)\n", passAttempt2, float64(passAttempt2)/float64(total)*100)
	fmt.Printf("   - Pass 3rd Attempt: %d (%.1f%%)\n", passAttempt3, float64(passAttempt3)/float64(total)*100)
	fmt.Printf("   - Overall Self-Healing Success Rate: %.1f%%\n", successRate)
}

func init() {
	rand.Seed(time.Now().UnixNano())
}
