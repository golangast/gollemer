package llm

import (
	"log"
)

func RunLLM(talk bool, listen bool) {
	runner, err := NewRunner(talk, listen)
	if err != nil {
		log.Fatalf("Failed to initialize LLM runner: %v", err)
	}

	runner.Init()
	runner.Run()
}
