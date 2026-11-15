package main

import (
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"strings"

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
	augmentMutationsFlag := flag.Bool("augment-mutations", false, "Social only: add one mutated sibling (typo / function-word drop / word swap) per training input, target unchanged")
	mutationBlocklistFlag := flag.String("mutation-blocklist", "", "JSONL whose inputs mutation siblings must never reproduce (the frozen robustness eval); required with -augment-mutations")
	trainLayersFlag := flag.Bool("train-layers", false, "Layers pilot: train the social seq2seq with generic layers on (see -layers); separate model file, production trainer untouched")
	layersFlag := flag.String("layers", "", "Comma-separated layers for -train-layers: char,class,frame,frame-top,frame-slots,frame-soft,copy,turn,memory")
	layersEpochsFlag := flag.Int("layers-epochs", 0, "Epoch cap for -train-layers (0 = the production 200)")
	layersMaxPairsFlag := flag.Int("layers-max-pairs", 0, "Seeded subsample of prepared pairs for -train-layers (0 = all; smoke/screening)")
	buildHoldoutFlag := flag.String("build-combination-holdout", "", "Layers pilot: build the combination-holdout artifacts (DESIGN §12) into this directory and exit")
	reclassifyFlag := flag.Bool("reclassify-domains", false, "Re-tag all dataset pairs with the current domain classifier")
	genUnderstandingFlag := flag.Bool("gen-understanding", false, "Generate structure->language training pairs from Go source")
	uexplainProbeFlag := flag.String("uexplain-probe", "", "Feed facts text to the understanding model and print its raw output")
	srcDirFlag := flag.String("src", ".", "Source directory for -gen-understanding")
	outFileFlag := flag.String("out", "data/training/understanding_pairs.jsonl", "Output file for -gen-understanding")
	flag.Parse()

	if !*trainRealSeq2SeqFlag && !*trainLayersFlag && !*realChatFlag && *importPairsFlag == "" && !*reclassifyFlag && !*genUnderstandingFlag && *uexplainProbeFlag == "" && *watchFlag == "" && *buildHoldoutFlag == "" {
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

	if *buildHoldoutFlag != "" {
		if err := chat.BuildCombinationHoldout(rootDir, *buildHoldoutFlag); err != nil {
			fmt.Fprintf(os.Stderr, "combination holdout build failed: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("combination holdout artifacts written to %s\n", *buildHoldoutFlag)
		return
	}

	if *trainLayersFlag {
		var layers []string
		for _, name := range strings.Split(*layersFlag, ",") {
			if name = strings.TrimSpace(name); name != "" {
				layers = append(layers, name)
			}
		}
		if err := chat.RunLayersTraining(rootDir, chat.LayersTrainOptions{
			Layers:   layers,
			Epochs:   *layersEpochsFlag,
			MaxPairs: *layersMaxPairsFlag,
			Resume:   *resumeFlag,
		}); err != nil {
			fmt.Fprintf(os.Stderr, "layers training failed: %v\n", err)
			os.Exit(1)
		}
		return
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
		if *domainFlag == chat.SpecCopyDomain {
			// The copy/pointer probe has its own trainer: same data and
			// protocol as the spec domain, pointer-generator model.
			if err := chat.RunSpecCopyTraining(rootDir, *resumeFlag); err != nil {
				fmt.Fprintf(os.Stderr, "speccopy training failed: %v\n", err)
				os.Exit(1)
			}
			return
		}
		if *domainFlag == chat.PhrasingDomain {
			// The phrasing layer has its own trainer too: same
			// pointer-generator protocol, its own data (messy ->
			// canonical) and slot masking.
			if err := chat.RunPhrasingTraining(rootDir, *resumeFlag); err != nil {
				fmt.Fprintf(os.Stderr, "phrasing training failed: %v\n", err)
				os.Exit(1)
			}
			return
		}
		if *domainFlag == chat.Phrasing2Domain {
			// Phrasing attempt 2, arm A: v2 data, indexed-UNK
			// masking, checkpoint selection by validation.
			if err := chat.RunPhrasing2Training(rootDir, *resumeFlag); err != nil {
				fmt.Fprintf(os.Stderr, "phrasing2 training failed: %v\n", err)
				os.Exit(1)
			}
			return
		}
		if err := chat.RunRealSeq2SeqTrainingOpts(rootDir, *domainFlag, *resumeFlag, chat.RealTrainOptions{
			AugmentMutations:  *augmentMutationsFlag,
			MutationBlocklist: *mutationBlocklistFlag,
		}); err != nil {
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
