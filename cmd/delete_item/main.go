package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run main.go <path_to_delete>")
		os.Exit(1)
	}

	pathToDelete := os.Args[1]

	fmt.Printf("Are you sure you want to delete %s? (y/n): ", pathToDelete)
	reader := bufio.NewReader(os.Stdin)
	input, _ := reader.ReadString('\n')
	input = strings.TrimSpace(input)

	if strings.ToLower(input) != "y" {
		fmt.Println("Deletion aborted.")
		os.Exit(0)
	}

	err := os.RemoveAll(pathToDelete)
	if err != nil {
		fmt.Printf("Error deleting %s: %v\n", pathToDelete, err)
		os.Exit(1)
	}

	fmt.Printf("Successfully deleted %s\n", pathToDelete)
}
