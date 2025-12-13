package main

import (
	"bufio"
	"errors"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/golangast/gollemer/neural/moe"
	mainvocab "github.com/golangast/gollemer/neural/nnu/vocab"
	"github.com/golangast/gollemer/neural/tensor"
	"github.com/golangast/gollemer/neural/tokenizer"
	"github.com/golangast/gollemer/tagger/nertagger"
	"github.com/golangast/gollemer/tagger/postagger"
	"github.com/golangast/gollemer/tagger/tag"
)

func main() {
	trainWord2Vec := flag.Bool("train-word2vec", false, "Train the Word2Vec model")
	trainMoE := flag.Bool("train-moe", false, "Train the MoE model")
	trainIntentClassifier := flag.Bool("train-intent-classifier", false, "Train the intent classification model")
	moeInferenceQuery := flag.String("moe_inference", "", "Run MoE inference with the given query")
	runLLMFlag := flag.Bool("llm", false, "Run in interactive LLM mode")

	flag.Parse()

	if *runLLMFlag {
		runLLM()
	} else if *trainWord2Vec {
		runModule("cmd/train_word2vec")
	} else if *trainMoE {
		runModule("cmd/train_moe")
	} else if *trainIntentClassifier {
		runModule("cmd/train_intent_classifier")
	} else if *moeInferenceQuery != "" {
		runMoeInference(*moeInferenceQuery)
	} else {
		log.Println("No action specified. Use -train-word2vec, -train-moe, -train-intent-classifier, -moe_inference <query>, or -llm.")
	}
}

func runMoeInference(query string) {
	cmd := exec.Command("go", "run", "./cmd/moe_inference", "-query", query)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	err := cmd.Run()
	if err != nil {
		log.Fatalf("Failed to run MoE inference: %v", err)
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

func findName(taggedData tag.Tag) string {
	// First, look for a FILENAME tag
	for i, tag := range taggedData.NerTag {
		if tag == "FILENAME" {
			return taggedData.Tokens[i]
		}
	}
	// Fallback for "named"
	for i, token := range taggedData.Tokens {
		if token == "named" && i+1 < len(taggedData.Tokens) {
			return taggedData.Tokens[i+1]
		}
	}
	// Fallback for NAME tag
	for i, tag := range taggedData.NerTag {
		if tag == "NAME" {
			return taggedData.Tokens[i]
		}
	}
	return ""
}

func runLLM() {
	rand.Seed(1) // Seed the random number generator for deterministic behavior

	// Define paths
	const vocabPath = "gob_models/query_vocabulary.gob"
	const moeModelPath = "gob_models/moe_classification_model.gob"
	const semanticOutputVocabPath = "gob_models/semantic_output_vocabulary.gob"

	// Load vocabularies
	vocabulary, err := mainvocab.LoadVocabulary(vocabPath)
	if err != nil {
		log.Fatalf("Failed to set up input vocabulary: %v", err)
	}

	semanticOutputVocabulary, err := mainvocab.LoadVocabulary(semanticOutputVocabPath)
	if err != nil {
		log.Fatalf("Failed to set up semantic output vocabulary: %v", err)
	}

	// Create tokenizer
	tok, err := tokenizer.NewTokenizer(vocabulary)
	if err != nil {
		log.Fatalf("Failed to create tokenizer: %v", err)
	}

	semanticOutputTokenizer, err := tokenizer.NewTokenizer(semanticOutputVocabulary)
	if err != nil {
		log.Fatalf("Failed to create semantic output tokenizer: %v", err)
	}

	// Load the trained MoEClassificationModel model
	model, err := moe.LoadIntentMoEModelFromGOB(moeModelPath)
	if err != nil {
		log.Fatalf("Failed to load MoE model: %v", err)
	}
	// Set the SentenceVocab for output decoding
	model.SentenceVocab = semanticOutputVocabulary

	reader := bufio.NewReader(os.Stdin)

	for {
		fmt.Print("> ")
		query, _ := reader.ReadString('\n')
		query = strings.TrimSpace(query)

		if query == "exit" {
			break
		}

		// --- Tagging ---
		words := strings.Fields(query)
		posTags := postagger.TagTokens(words)
		taggedData := nertagger.Nertagger(tag.Tag{Tokens: words, PosTag: posTags})

		// Encode the query
		tokenIDs, err := tok.Encode(query)
		if err != nil {
			log.Printf("Failed to encode query: %v", err)
			continue
		}

		// Pad or truncate the sequence to a fixed length
		maxSeqLength := 120 // Increased to 120 to match training
		if len(tokenIDs) > maxSeqLength {
			tokenIDs = tokenIDs[:maxSeqLength] // Truncate from the end
		} else {
			for len(tokenIDs) < maxSeqLength {
				tokenIDs = append(tokenIDs, vocabulary.PaddingTokenID) // Appends padding
			}
		}
		inputData := make([]float64, len(tokenIDs))
		for i, id := range tokenIDs {
			inputData[i] = float64(id)
		}
		inputTensor := tensor.NewTensor([]int{1, len(inputData)}, inputData, false)

		// Create a dummy target tensor for inference
		dummyTargetTokenIDs := make([]float64, maxSeqLength)
		for i := 0; i < maxSeqLength; i++ {
			dummyTargetTokenIDs[i] = float64(vocabulary.PaddingTokenID)
		}
		dummyTargetTensor := tensor.NewTensor([]int{1, maxSeqLength}, dummyTargetTokenIDs, false)

		// Forward pass to get the context vector
		_, contextVector, err := model.Forward(0.0, inputTensor, dummyTargetTensor)
		if err != nil {
			log.Printf("MoE model forward pass failed: %v", err)
			continue
		}

		// Greedy search decode to get the predicted token IDs
		predictedIDs, err := model.GreedySearchDecode(contextVector, maxSeqLength, semanticOutputVocabulary.GetTokenID("<s>"), semanticOutputVocabulary.GetTokenID("</s>"), 1.2, 100, taggedData) // topK=100
		if err != nil {
			log.Printf("Greedy search decode failed: %v", err)
			continue
		}

		// Decode the predicted IDs to a sentence (now in flat key:value format)
		predictedSentence, err := semanticOutputTokenizer.Decode(predictedIDs)
		if err != nil {
			log.Printf("Failed to decode predicted IDs: %v", err)
			continue
		}

		// --- Start of new logic ---

		// fmt.Println("--- DEBUG INFO ---")
		// fmt.Printf("Tagged Tokens: %v\n", taggedData.Tokens)
		// fmt.Printf("NER Tags: %v\n", taggedData.NerTag)

		        var command string
				// Try to explicitly identify the command if it's the first token
				if len(taggedData.Tokens) > 0 {
					if strings.ToLower(taggedData.Tokens[0]) == "create" {
						command = "create"
					} else if strings.ToLower(taggedData.Tokens[0]) == "list" {
						command = "list"
					} else if strings.ToLower(taggedData.Tokens[0]) == "go" {
						command = "go"
					}
				}
		
				var targetDirectory string
				for i, token := range taggedData.Tokens {
					if i < len(taggedData.NerTag) {
						switch taggedData.NerTag[i] {
						case "QUESTION_WORD":
							if token == "what" {
								hasQuestionWord = true
							}
						case "VERB":
							if token == "is" {
								hasVerb = true
							}
						case "OBJECT_TYPE":
							objectTypeParts = append(objectTypeParts, token)
						case "PREPOSITION":
							if token == "in" || token == "into" {
								hasPrepositionIn = true
								if i+1 < len(taggedData.Tokens) {
									targetDirectory = taggedData.Tokens[i+1]
								}
							}
						}
					}
				}
		
				objectType := strings.Join(objectTypeParts, " ")
				fileName := findName(taggedData)
		
				// Heuristic: If fileName is still empty, and objectType is "file",
				// check for tokens that look like filenames (e.g., ends with .go)
				if fileName == "" && contains(objectTypeParts, "file") {
					for _, token := range taggedData.Tokens {
						if strings.HasSuffix(token, ".go") || strings.HasSuffix(token, ".txt") || strings.HasSuffix(token, ".md") {
							fileName = token
							break
						}
					}
				}
		objectType := strings.Join(objectTypeParts, " ")
		fileName := findName(taggedData)

		var hasDirectoryToken bool
		for _, t := range taggedData.Tokens {
			if t == "directory" {
				hasDirectoryToken = true
				break
			}
		}

		fmt.Printf("ObjectTypeParts: %v\n", objectTypeParts)
		fmt.Printf("ObjectType: %s\n", objectType)
		fmt.Printf("HasQuestionWord: %t\n", hasQuestionWord)
		fmt.Printf("HasPrepositionIn: %t\n", hasPrepositionIn)
		fmt.Printf("Command: %s\n", command)
		fmt.Printf("FileName: %s\n", fileName)
		fmt.Printf("HasDirectoryToken: %t\n", hasDirectoryToken)
		fmt.Printf("TargetDirectory: %s\n", targetDirectory) // New debug info
		fmt.Println("--------------------")

		contains := func(s []string, e string) bool {
			for _, a := range s {
				if a == e {
					return true
				}
			}
			return false
		}

		if command == "create" && contains(objectTypeParts, "file") {
			if fileName != "" {
				filePath := fileName
				if targetDirectory != "" {
					filePath = filepath.Join(targetDirectory, fileName)
				}
				err := os.WriteFile(filePath, []byte(""), 0644)
				if err != nil {
					predictedSentence = fmt.Sprintf("I couldn't create the file %s: %v", filePath, err)
				} else {
					predictedSentence = fmt.Sprintf("I have created the file %s.", filePath)
				}
			} else {
				predictedSentence = "You need to provide a name for the file."
			}
		} else if command == "create" && contains(objectTypeParts, "folder") {
			folderName := findName(taggedData)
			if folderName != "" {
				folderPath := folderName
				if targetDirectory != "" {
					folderPath = filepath.Join(targetDirectory, folderName)
				}
				err := os.MkdirAll(folderPath, 0755) // Use MkdirAll to create parent directories if needed
				if err != nil {
					predictedSentence = fmt.Sprintf("I couldn't create the folder %s: %v", folderPath, err)
				} else {
					predictedSentence = fmt.Sprintf("I have created the folder %s.", folderPath)
				}
			} else {
				predictedSentence = "You need to provide a name for the folder."
			}
		} else if command == "go" && (contains(taggedData.Tokens, "in") || contains(taggedData.Tokens, "into")) {
			folderName := findName(taggedData)
			if folderName != "" {
				err := os.Chdir(folderName)
				if err != nil {
					predictedSentence = fmt.Sprintf("I couldn't change the directory to %s: %v", folderName, err)
				} else {
					cwd, _ := os.Getwd()
					predictedSentence = fmt.Sprintf("I have changed the directory to %s.", cwd)
				}
			} else {
				predictedSentence = "You need to provide a folder name."
			}
		} else if command == "list" && (contains(objectTypeParts, "files") || contains(objectTypeParts, "file") || contains(objectTypeParts, "folder") || contains(objectTypeParts, "folders") || contains(objectTypeParts, "directory")) {
			entries, err := os.ReadDir(".")
			if err != nil {
				predictedSentence = "I'm sorry, I couldn't list the files."
			} else {
				var fileNames []string
				for _, e := range entries {
					if e.IsDir() {
						fileNames = append(fileNames, e.Name()+"/")
					} else {
						fileNames = append(fileNames, e.Name())
					}
				}
				predictedSentence = strings.Join(fileNames, "\n")
			}
		} else if hasQuestionWord && (contains(objectTypeParts, "files") || contains(objectTypeParts, "file") || contains(objectTypeParts, "folder") || contains(objectTypeParts, "folders")) {
			entries, err := os.ReadDir(".")
			if err != nil {
				predictedSentence = "I'm sorry, I couldn't list the files."
			} else {
				var fileNames []string
				for _, e := range entries {
					if e.IsDir() {
						fileNames = append(fileNames, e.Name()+"/")
					} else {
						fileNames = append(fileNames, e.Name())
					}
				}
				predictedSentence = strings.Join(fileNames, "\n")
			}
		} else if hasQuestionWord && hasDirectoryToken {
			cwd, err := os.Getwd()
			if err != nil {
				predictedSentence = "I'm sorry, I couldn't determine the current directory."
			} else {
				predictedSentence = fmt.Sprintf("The current directory is: %s", cwd)
			}
		} else if hasQuestionWord && hasVerb && hasPrepositionIn && fileName != "" {
			var foundPath string
			filepath.Walk(".", func(path string, info os.FileInfo, err error) error {
				if err != nil {
					return err
				}
				if !info.IsDir() && info.Name() == fileName {
					foundPath = path
					return errors.New("found")
				}
				return nil
			})

			if foundPath != "" {
				content, err := os.ReadFile(foundPath)
				if err != nil {
					predictedSentence = fmt.Sprintf("I found the file %s, but I couldn't read it: %v", foundPath, err)
				} else {
					predictedSentence = string(content)
				}
			} else {
				predictedSentence = fmt.Sprintf("I couldn't find the file %s.", fileName)
			}
		}
		// --- End of new logic ---

		// Print the output (flat format: operation:Create type:Webserver name:jill ...)
		fmt.Println("---")
		fmt.Println(predictedSentence)
		fmt.Println("---------------------------------")
	}
}
