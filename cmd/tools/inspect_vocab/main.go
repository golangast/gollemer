package main

import (
	"bufio"
	"compress/gzip"
	"encoding/gob"
	"fmt"
	"io"
	"os"

	"github.com/golangast/gollemer/internal/ai/moe"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/word2vec"
	"github.com/golangast/gollemer/internal/ai/neural/tokenizer"
)

func main() {
	fmt.Println("================================================================================")
	fmt.Println("🔍 GOLLEMER VOCABULARY & TOKENIZATION DIAGNOSTIC")
	fmt.Println("================================================================================")

	// 1. Load Word2Vec Model
	w2vPath := "data/models/gob_models/word2vec_model.gob"
	w2v, err := word2vec.LoadModel(w2vPath)
	if err != nil {
		fmt.Printf("❌ Word2Vec: Failed to load from %s: %v\n", w2vPath, err)
	} else {
		fmt.Printf("✅ Word2Vec: Loaded successfully\n")
		fmt.Printf("   • Vocabulary Size:  %d\n", w2v.VocabSize)
		fmt.Printf("   • Vector Dimension: %d\n", w2v.VectorSize)
	}

	// 2. Load BPE Tokenizer
	bpePath := "data/models/gob_models/bpe_tokenizer.gob"
	bpe, err := tokenizer.LoadBPETokenizer(bpePath)
	if err != nil {
		fmt.Printf("❌ BPE Tokenizer: Failed to load from %s: %v\n", bpePath, err)
	} else {
		fmt.Printf("✅ BPE Tokenizer: Loaded successfully\n")
		fmt.Printf("   • Vocabulary Size:  %d\n", bpe.Vocab.Size())
	}

	// 3. Load MoE Model Checkpoint
	moePath := "data/models/gob_models/moe_social_model.gob"
	file, err := os.Open(moePath)
	var model *moe.IntentMoE
	if err != nil {
		fmt.Printf("❌ MoE Model: Failed to open %s: %v\n", moePath, err)
	} else {
		defer file.Close()
		// Attempt multi-format decoding
		if gz, gzErr := gzip.NewReader(file); gzErr == nil {
			var dc moe.Checkpoint
			if decErr := gob.NewDecoder(gz).Decode(&dc); decErr == nil && dc.Model != nil {
				model = dc.Model
			} else if decErr != nil {
				fmt.Printf("❌ MoE Model: Failed to decode GZIP/Checkpoint format: %v\n", decErr)
			}
			gz.Close()
		} else {
			fmt.Printf("❌ MoE Model: Failed to read as GZIP: %v\n", gzErr)
		}
		if model == nil {
			_, _ = file.Seek(0, io.SeekStart)
			var dc moe.Checkpoint
			if decErr := gob.NewDecoder(bufio.NewReader(file)).Decode(&dc); decErr == nil && dc.Model != nil {
				model = dc.Model
			}
		}
		if model == nil {
			_, _ = file.Seek(0, io.SeekStart)
			if gz, gzErr := gzip.NewReader(file); gzErr == nil {
				var dm moe.IntentMoE
				if decErr := gob.NewDecoder(gz).Decode(&dm); decErr == nil {
					model = &dm
				} else {
					fmt.Printf("❌ MoE Model: Failed to decode GZIP/Raw format: %v\n", decErr)
				}
				gz.Close()
			}
		}

		if model == nil {
			fmt.Println("❌ MoE Model: Failed to decode model in all checkpoint/raw formats.")
		} else {
			fmt.Printf("✅ MoE Model: Loaded successfully\n")
			fmt.Printf("   • Model Output Vocab Size (SentenceVocabSize): %d\n", model.SentenceVocabSize)
			if model.Embedding != nil {
				fmt.Printf("   • Embedding Table Vocabulary Size:           %d\n", model.Embedding.VocabSize)
			}
			if model.SentenceVocab != nil {
				fmt.Printf("   • Internal SentenceVocab size:               %d\n", model.SentenceVocab.Size())
			} else {
				fmt.Println("   • ⚠️ Internal SentenceVocab is nil!")
			}
		}
	}

	// 4. Diagnostic & Analysis
	fmt.Println("\n================================================================================")
	fmt.Println("📋 DIAGNOSTIC REPORT")
	fmt.Println("================================================================================")

	if bpe != nil && model != nil {
		tokenizerSize := bpe.Vocab.Size()
		modelVocabSize := model.SentenceVocabSize

		fmt.Printf("Tokenizer Vocab Size: %5d\n", tokenizerSize)
		fmt.Printf("Model Vocab Size:     %5d\n", modelVocabSize)

		if tokenizerSize != modelVocabSize {
			fmt.Println("\n🚨 CRITICAL MISMATCH DETECTED!")
			fmt.Printf("The BPE Tokenizer's vocabulary size (%d) does not match the MoE model's vocabulary size (%d).\n", tokenizerSize, modelVocabSize)
			fmt.Println("This is a major source of the UNK/Word-Salad issue:")
			fmt.Println("1. During inference/sampling, the BPE tokenizer encodes prompt words into high token IDs (e.g. >25).")
			fmt.Println("2. Since these token IDs are larger than the model's vocabulary limit, the model falls back to the UNK token (ID 1).")
			fmt.Println("3. Consequently, the model processes almost all input words as UNK, generating meaningless 'word salad'.")
			fmt.Println("\n💡 RECOMMENDED REMEDIAL ACTIONS:")
			fmt.Println("A. Run a full cold-start fresh training cycle (e.g., 'make train-fresh' or 'make train') to align vocabs.")
			fmt.Println("B. Ensure BPE Tokenizer is disabled or the model embeddings are resized properly if resuming training.")
		} else {
			fmt.Println("\n✨ Alignment check: Tokenizer and Model Vocab sizes are synchronized! No out-of-range mismatch.")
		}
	} else if model != nil {
		fmt.Println("⚠️  Cannot check alignment since BPE Tokenizer could not be loaded.")
	}

	// 5. Inspect sample tokens from the model's vocab
	if model != nil && model.SentenceVocab != nil {
		fmt.Println("\n================================================================================")
		fmt.Printf("📖 Sample tokens from Model's SentenceVocab (Total: %d)\n", model.SentenceVocab.Size())
		fmt.Println("================================================================================")
		maxToPrint := 30
		if model.SentenceVocab.Size() < maxToPrint {
			maxToPrint = model.SentenceVocab.Size()
		}
		for id := 0; id < maxToPrint; id++ {
			fmt.Printf("  • ID %3d: %q\n", id, model.SentenceVocab.GetWord(id))
		}
	}

	fmt.Println("================================================================================")
}

func loadLegacyVocab(path string) (*vocab.Vocabulary, error) {
	return vocab.LoadVocabulary(path)
}
