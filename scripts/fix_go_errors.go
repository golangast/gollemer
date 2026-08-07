package main

import (
	"fmt"
	"log"
	"os/exec"
)

// FixGoErrors uses the shell script to fix Go files
func FixGoErrors(filePath string) error {
	cmd := exec.Command("./scripts/fix_go_errors.sh", filePath)
	output, err := cmd.CombinedOutput()
	if err != nil {
		log.Printf("Error executing script: %v\n", err)
		return err
	}
	fmt.Printf("Script output: %s\n", output)
	return nil
}
