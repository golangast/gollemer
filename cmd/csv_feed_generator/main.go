package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/golangast/gollemer/internal/config"
	"github.com/golangast/gollemer/internal/csvgen"
)

func main() {
	// Define command-line flags
	feedsPath := flag.String("feeds", "feeds", "Path to the feeds directory containing CSV files")
	outputDir := flag.String("output", "generated_projects/project", "Output directory for generated Go files")
	configPath := flag.String("config", "", "Path to project configuration JSON file")
	help := flag.Bool("help", false, "Show help message")

	flag.Parse()

	if *help {
		printHelp()
		return
	}

	fmt.Println("🔄 CSV to Go Struct & Handler Generator")
	fmt.Println("========================================")
	fmt.Println()

	if *configPath != "" {
		fmt.Printf("📄 Loading configuration from: %s\n", *configPath)
		cfg, err := config.LoadConfig(*configPath)
		if err != nil {
			fmt.Printf("❌ Error loading config: %v\n", err)
			os.Exit(1)
		}

		fmt.Printf("📁 Output directory: %s\n", *outputDir)
		fmt.Println()

		if err := csvgen.GenerateFromConfig(cfg, *outputDir); err != nil {
			fmt.Printf("❌ Error generating from config: %v\n", err)
			os.Exit(1)
		}
	} else {
		// Existing logic for directory scanning
		// Check if feeds directory exists
		if _, err := os.Stat(*feedsPath); os.IsNotExist(err) {
			fmt.Printf("❌ Error: Feeds directory '%s' does not exist\n", *feedsPath)
			fmt.Println("\nCreate the directory and add CSV files, or specify a different path with -feeds flag")
			os.Exit(1)
		}

		// Count CSV files
		entries, err := os.ReadDir(*feedsPath)
		if err != nil {
			fmt.Printf("❌ Error reading feeds directory: %v\n", err)
			os.Exit(1)
		}

		csvCount := 0
		for _, entry := range entries {
			if !entry.IsDir() && filepath.Ext(entry.Name()) == ".csv" {
				csvCount++
			}
		}

		if csvCount == 0 {
			fmt.Printf("⚠️  No CSV files found in '%s'\n", *feedsPath)
			fmt.Println("\nAdd some CSV files to the directory and try again")
			os.Exit(0)
		}

		fmt.Printf("📂 Feeds directory: %s\n", *feedsPath)
		fmt.Printf("📁 Output directory: %s\n", *outputDir)
		fmt.Printf("📊 CSV files found: %d\n", csvCount)
		fmt.Println()

		// Process the feeds
		if err := csvgen.ProcessFeedsFolder(*feedsPath, *outputDir); err != nil {
			fmt.Printf("❌ Error processing feeds: %v\n", err)
			os.Exit(1)
		}
	}

	fmt.Println()
	fmt.Println("✅ Successfully generated structs and handlers!")
	fmt.Println()
	fmt.Println("📝 Next steps:")
	fmt.Printf("   1. cd %s\n", *outputDir)
	fmt.Println("   2. go mod init myapi  (if not already initialized)")
	fmt.Println("   3. go run .")
	fmt.Println()
	fmt.Println("🌐 Your API will be available at http://localhost:8080")
	fmt.Println()
}

func printHelp() {
	fmt.Println("CSV to Go Struct & Handler Generator")
	fmt.Println("=====================================")
	fmt.Println()
	fmt.Println("This tool automatically generates Go structs and CRUD HTTP handlers")
	fmt.Println("from CSV files. It's perfect for quickly creating REST APIs from")
	fmt.Println("existing data files.")
	fmt.Println()
	fmt.Println("Usage:")
	fmt.Println("  csv_feed_generator [flags]")
	fmt.Println()
	fmt.Println("Flags:")
	fmt.Println("  -feeds string")
	fmt.Println("        Path to the feeds directory containing CSV files (default \"feeds\")")
	fmt.Println("  -output string")
	fmt.Println("        Output directory for generated Go files (default \"generated_projects/project\")")
	fmt.Println("  -config string")
	fmt.Println("        Path to project configuration JSON file (optional)")
	fmt.Println("  -help")
	fmt.Println("        Show this help message")
	fmt.Println()
	fmt.Println("Example:")
	fmt.Println("  csv_feed_generator -feeds ./data -output ./api")
	fmt.Println("  csv_feed_generator -config project.json")
	fmt.Println()
	fmt.Println("What it generates:")
	fmt.Println("  - Go structs based on CSV headers")
	fmt.Println("  - CRUD handlers (List, Get, Create)")
	fmt.Println("  - HTTP route registration")
	fmt.Println("  - Complete runnable server (main.go)")
	fmt.Println()
	fmt.Println("CSV Format:")
	fmt.Println("  - First row must be headers")
	fmt.Println("  - Headers will be converted to Go field names")
	fmt.Println("  - Types are inferred from first data row")
	fmt.Println()
}
