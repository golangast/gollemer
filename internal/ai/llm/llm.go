package llm

import (
	"log"
)

func RunLLM() {
	runner, err := NewRunner()
	if err != nil {
		log.Fatalf("Failed to initialize LLM runner: %v", err)
	}

	runner.Init()
	runner.Run()
}
