package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
)

func main() {
	file, err := os.Open("data/logs/expert_health.csv")
	if err != nil {
		fmt.Println("Error: Could not find health log at data/logs/expert_health.csv.")
		return
	}
	defer file.Close()

	reader := csv.NewReader(file)
	fmt.Println("\n🔥 Gollemer Expert Heatmap (Layer 0) 🔥")
	fmt.Println("Epoch | E0  | E1  | E2  | E3  | E4  | E5  | E6  | E7  | DOM %")
	fmt.Println("-------------------------------------------------------------")

	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		
		// Skip header if present (or just check for numeric Layer ID)
		if record[2] != "0" {
			continue
		}

		epoch := record[1]
		dom, _ := strconv.ParseFloat(record[len(record)-1], 64)

		fmt.Printf("%-5s |", epoch)
		
		// Print Experts 0-7 (columns 3 to 10 if they exist)
		numCols := len(record) - 1
		for i := 3; i < numCols && i <= 10; i++ {
			val, _ := strconv.ParseFloat(record[i], 64)
			printColoredCell(val)
		}

		// Print Dominance with a warning color if > 0.90
		if dom > 0.90 {
			fmt.Printf(" \033[31m%.2f%%\033[0m\n", dom*100)
		} else {
			fmt.Printf(" %.2f%%\n", dom*100)
		}
	}
}

func printColoredCell(val float64) {
	// ANSI Colors: 34 (Blue/Cold), 32 (Green/Healthy), 33 (Yellow/Warm), 31 (Red/Hot)
	color := 34 
	if val > 0.10 && val <= 0.25 {
		color = 32 // Healthy distribution
	} else if val > 0.25 && val <= 0.70 {
		color = 33 // Warming up
	} else if val > 0.70 {
		color = 31 // Overheated / Dominant
	}
	fmt.Printf(" \033[%dm%0.2f\033[0m |", color, val)
}
