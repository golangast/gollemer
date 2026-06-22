package llm

import (
	"log"
)

func RunLLM(talk bool, listen bool, cartridges string) {
	runner, err := NewRunner(talk, listen, cartridges)
	if err != nil {
		log.Fatalf("Failed to initialize LLM runner: %v", err)
	}

	runner.Init()
	runner.Run()
}
