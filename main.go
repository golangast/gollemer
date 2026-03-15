package main

import (
	"flag"
	"log"
	_ "net/http/pprof"
	"os"
	"os/exec"

	_ "modernc.org/sqlite" // Pure Go SQLite driver

	"github.com/golangast/gollemer/llm"
	"github.com/golangast/gollemer/training/chat"
)

func main() {

	// Initialize absoluteLastDirConfigPath based on the project root
	projectRoot, err := llm.FindProjectRoot()
	if err != nil {
		log.Fatalf("Failed to find project root: %v", err)
	}

	trainWord2Vec := flag.Bool("train-word2vec", false, "Train the Word2Vec model")
	trainMoE := flag.Bool("train-moe", false, "Train the MoE model")
	trainIntentClassifier := flag.Bool("train-intent-classifier", false, "Train the intent classification model")
	trainNER := flag.Bool("train-ner", false, "Train the Named Entity Recognition model")
	runLLMFlag := flag.Bool("llm", false, "Run in interactive LLM mode")
	trainChatFlag := flag.Bool("train-chat", false, "Train the Chat RAG model from human_chat.txt")
	rebalancePtr := flag.Bool("rebalance", false, "Rebalance MoE expert weights before training")

	flag.Parse()
	switch {
	case *runLLMFlag:
		llm.RunLLM()
	case *trainWord2Vec:
		runModule("cmd/train_word2vec")
	case *trainMoE:
		runModule("cmd/train_moe")
	case *trainIntentClassifier:
		runModule("cmd/train_intent_classifier")
	case *trainNER:
		runModule("cmd/train_ner")
	case *trainChatFlag:
		chat.TrainChat(projectRoot, *rebalancePtr)
	default:
		log.Println("No action specified. Use -train-word2vec, -train-moe, -train-intent-classifier, -train-ner, -train-chat, or -llm.")
	}
}

func runModule(path string) {
	cmd := exec.Command("go", "run", "./"+path)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	err := cmd.Run()
	if err != nil {
		log.Fatalf("Failed to run module %s: %v", path, err)
	}
}
