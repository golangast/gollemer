package llm

import (
	"log"
)

func RunLLM(talk bool) {
	runner, err := NewRunner(talk)
	if err != nil {
		log.Fatalf("Failed to initialize LLM runner: %v", err)
	}

	runner.Init()
	runner.Run()
}
