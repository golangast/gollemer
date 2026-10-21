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
	watchFlag := flag.String("watch", "", "Watch a folder while you code: report meaningful structural changes as you save")
	realChatFlag := flag.Bool("real-chat", false, "Chat with the trained neural model (pure generation, no lookup)")
	debugChatFlag := flag.Bool("debug-chat", false, "Show the thought process and debug prints in chat (default: clean replies only)")
	importPairsFlag := flag.String("import-pairs", "", "Import new training pairs from a JSONL file through the quality gate")
	domainFlag := flag.String("domain", "social", "Training/chat domain (social, go, ...)")
	resumeFlag := flag.Bool("resume", false, "Resume -train-real-seq2seq from the last checkpoint instead of starting at epoch 0")
	reclassifyFlag := flag.Bool("reclassify-domains", false, "Re-tag all dataset pairs with the current domain classifier")
	genUnderstandingFlag := flag.Bool("gen-understanding", false, "Generate structure->language training pairs from Go source")
	uexplainProbeFlag := flag.String("uexplain-probe", "", "Feed facts text to the understanding model and print its raw output")
	srcDirFlag := flag.String("src", ".", "Source directory for -gen-understanding")
	outFileFlag := flag.String("out", "data/training/understanding_pairs.jsonl", "Output file for -gen-understanding")
	trainTransformerFlag := flag.Bool("train-transformer", false, "Train the mid-size Transformer on a pairs file (mid-transformer experiment)")
	evalTransformerFlag := flag.Bool("eval-transformer", false, "Evaluate a trained Transformer on held-out pairs (greedy exact-match echo; mid-transformer experiment)")
	pairsFlag := flag.String("pairs", "", "Training pairs JSONL for -train-transformer (vocab is built from this file only)")
	heldoutFlag := flag.String("heldout", "", "Held-out pairs JSONL for -train-transformer (vocab coverage report only; never trained on)")
	expNameFlag := flag.String("exp-name", "copyexp", "Experiment name for -train-transformer artifacts (transformer_<name>.gob)")
	maxStepsFlag := flag.Int("max-steps", 0, "For -train-transformer: stop after N optimizer steps (smoke mode; 0 = full training)")
	batchTokensFlag := flag.Int("batch-tokens", 0, "For -train-transformer: padded source-token budget per batch (0 = default 2048)")
	flag.Parse()

	if !*trainRealSeq2SeqFlag && !*trainTransformerFlag && !*evalTransformerFlag && !*realChatFlag && *importPairsFlag == "" && !*reclassifyFlag && !*genUnderstandingFlag && *uexplainProbeFlag == "" && *watchFlag == "" {
		fmt.Fprintf(os.Stderr, "Usage: gollemer -train-real-seq2seq | gollemer -real-chat | gollemer -import-pairs=file.jsonl | gollemer -reclassify-domains | gollemer -gen-understanding\n")
		os.Exit(1)
	}

	if *genUnderstandingFlag {
		if err := runGenUnderstanding(*srcDirFlag, *outFileFlag); err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		return
	}

	rootDir, err := os.Getwd()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error getting working directory: %v\n", err)
		os.Exit(1)
	}

	if *uexplainProbeFlag != "" {
		if err := chat.RunUexplainProbe(rootDir, *uexplainProbeFlag); err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		return
	}

	if *watchFlag != "" {
		chat.RunWatch(*watchFlag)
		return
	}

	if *trainTransformerFlag {
		if *pairsFlag == "" {
			fmt.Fprintf(os.Stderr, "Error: -train-transformer needs -pairs <train.jsonl>\n")
			os.Exit(1)
		}
		if err := chat.RunTransformerPairTraining(rootDir, *pairsFlag, *heldoutFlag, *expNameFlag, *resumeFlag, *maxStepsFlag, *batchTokensFlag); err != nil {
			fmt.Fprintf(os.Stderr, "transformer training failed: %v\n", err)
			os.Exit(1)
		}
		return
	}

	if *evalTransformerFlag {
		if *heldoutFlag == "" {
			fmt.Fprintf(os.Stderr, "Error: -eval-transformer needs -heldout <heldout.jsonl>\n")
			os.Exit(1)
		}
		if err := chat.RunTransformerEval(rootDir, *expNameFlag, *heldoutFlag, *pairsFlag); err != nil {
			fmt.Fprintf(os.Stderr, "transformer eval failed: %v\n", err)
			os.Exit(1)
		}
		return
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
