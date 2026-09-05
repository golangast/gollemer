package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/golangast/gollemer/internal/ai/training/chat"
)

func main() {
	trainFlag := flag.Bool("train", false, "Run the seq2seq LLM sentence-forming training pipeline")
	trainMultiphaseFlag := flag.Bool("train-multiphase", false, "Alias for -train")
	smallTrainFlag := flag.Bool("train-small", false, "Run the small social dataset, print loss + memory, and test the model with LLM prompts")
	smallLLMFlag := flag.Bool("small-llm", false, "Alias for -train-small")
	smallSeq2SeqFlag := flag.Bool("train-small-seq2seq", false, "Run a strict pure Q→A seq2seq tiny demo that is optimized for very low loss on the six-row social dataset")
	testSmallSeq2SeqFlag := flag.Bool("test-small-seq2seq", false, "Load the tiny seq2seq model and probe a few prompts")
	seq2SeqPromptFlag := flag.String("seq2seq-prompt", "", "Send a custom prompt to the saved tiny seq2seq model")
	seq2SeqChatFlag := flag.Bool("seq2seq-chat", false, "Start an interactive tiny seq2seq chat loop with the saved model")
	chatFlag := flag.Bool("chat", false, "Start an interactive full MoE chat loop with conversation history and reasoning")
	flag.Parse()

	if !*trainFlag && !*trainMultiphaseFlag && !*smallTrainFlag && !*smallLLMFlag && !*smallSeq2SeqFlag && !*testSmallSeq2SeqFlag && *seq2SeqPromptFlag == "" && !*seq2SeqChatFlag && !*chatFlag {
		fmt.Fprintf(os.Stderr, "Usage: gollemer -train | gollemer -train-small | gollemer -small-llm | gollemer -train-small-seq2seq | gollemer -test-small-seq2seq | gollemer -seq2seq-prompt='hello' | gollemer -seq2seq-chat | gollemer -chat\n")
		os.Exit(1)
	}

	rootDir, err := os.Getwd()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error getting working directory: %v\n", err)
		os.Exit(1)
	}

	log.Println("🤖 Gollemer LLM Trainer")
	log.Printf("   Root: %s\n", rootDir)
	log.Println("   Mode: Sentence-forming Seq2Seq")
	log.Println()

	if *smallTrainFlag || *smallLLMFlag {
		chat.RunSmallTrainLLMCheck(rootDir)
		return
	}
	if *smallSeq2SeqFlag {
		chat.RunTinySeq2SeqCurriculumCheck(rootDir)
		return
	}
	if *testSmallSeq2SeqFlag {
		chat.RunSmallSeq2SeqCheck(rootDir)
		return
	}
	if *seq2SeqPromptFlag != "" {
		chat.RunTinySeq2SeqPrompt(rootDir, *seq2SeqPromptFlag)
		return
	}
	if *seq2SeqChatFlag {
		chat.RunInteractiveTinySeq2SeqChat(rootDir)
		return
	}
	if *chatFlag {
		chat.RunMoEChat(rootDir)
		return
	}

	chat.TrainMultiPhaseCurriculum(rootDir, false, "")
}
