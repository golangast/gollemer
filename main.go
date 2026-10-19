package main

import (
	"flag"
	"fmt"
	"io"
	"log"
	"os"

	"github.com/golangast/gollemer/internal/ai/training/chat"
)

func main() {
	trainRealSeq2SeqFlag := flag.Bool("train-real-seq2seq", false, "Train the genuine neural seq2seq model (real BPTT, no cheat sheet)")
	trainTransformerFlag := flag.Bool("train-transformer", false, "Train the Transformer seq2seq model (attention-based, not LSTM)")
	pretrainTransformerFlag := flag.Bool("pretrain-transformer", false, "Pretrain the Transformer as a Go causal language model on raw Go code")
	testTransformerFlag := flag.String("test-transformer", "", "Test a trained Transformer with a prompt (e.g. -test-transformer='write a function')")
	realChatFlag := flag.Bool("real-chat", false, "Chat with the trained neural model (pure generation, no lookup)")
	debugChatFlag := flag.Bool("debug-chat", false, "Show the thought process and debug prints in chat (default: clean replies only)")
	importPairsFlag := flag.String("import-pairs", "", "Import new training pairs from a JSONL file through the quality gate")
	domainFlag := flag.String("domain", "social", "Training/chat domain (social, go, ...)")
	resumeFlag := flag.Bool("resume", false, "Resume -train-real-seq2seq from the last checkpoint instead of starting at epoch 0")
	reclassifyFlag := flag.Bool("reclassify-domains", false, "Re-tag all dataset pairs with the current domain classifier")
	flag.Parse()

	if !*trainRealSeq2SeqFlag && !*trainTransformerFlag && !*pretrainTransformerFlag && *testTransformerFlag == "" && !*realChatFlag && *importPairsFlag == "" && !*reclassifyFlag {
		fmt.Fprintf(os.Stderr, "Usage: gollemer -train-real-seq2seq | gollemer -real-chat | gollemer -import-pairs=file.jsonl | gollemer -reclassify-domains\n")
		os.Exit(1)
	}

	rootDir, err := os.Getwd()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error getting working directory: %v\n", err)
		os.Exit(1)
	}

	if *realChatFlag && !*debugChatFlag {
		// Clean chat: replies only, no banner or debug prints.
		log.SetOutput(io.Discard)
	} else {
		log.Println("🤖 Gollemer LLM Trainer")
		log.Printf("   Root: %s\n", rootDir)
		log.Println("   Mode: Sentence-forming Seq2Seq")
		log.Println()
	}

	if *trainRealSeq2SeqFlag {
		if err := chat.RunRealSeq2SeqTraining(rootDir, *domainFlag, *resumeFlag); err != nil {
			fmt.Fprintf(os.Stderr, "real training failed: %v\n", err)
			os.Exit(1)
		}
		return
	}
	if *trainTransformerFlag {
		if err := chat.RunTransformerTraining(rootDir, *domainFlag, *resumeFlag); err != nil {
			fmt.Fprintf(os.Stderr, "transformer training failed: %v\n", err)
			os.Exit(1)
		}
		return
	}
	if *pretrainTransformerFlag {
		if err := chat.RunTransformerPretraining(rootDir, *resumeFlag); err != nil {
			fmt.Fprintf(os.Stderr, "transformer pretraining failed: %v\n", err)
			os.Exit(1)
		}
		return
	}
	if *testTransformerFlag != "" {
		if err := chat.RunTransformerTest(rootDir, *domainFlag, *testTransformerFlag); err != nil {
			fmt.Fprintf(os.Stderr, "transformer test failed: %v\n", err)
			os.Exit(1)
		}
		return
	}
	if *realChatFlag {
		if err := chat.RunRealChat(rootDir, *domainFlag, *debugChatFlag); err != nil {
			fmt.Fprintf(os.Stderr, "real chat failed: %v\n", err)
			os.Exit(1)
		}
		return
	}
	if *importPairsFlag != "" {
		if err := chat.RunImportChatPairs(rootDir, *importPairsFlag); err != nil {
			fmt.Fprintf(os.Stderr, "import failed: %v\n", err)
			os.Exit(1)
		}
		return
	}
	if *reclassifyFlag {
		if err := chat.RunReclassifyDomains(rootDir); err != nil {
			fmt.Fprintf(os.Stderr, "reclassify failed: %v\n", err)
			os.Exit(1)
		}
		return
	}
}
