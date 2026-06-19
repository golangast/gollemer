package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

var targets = map[string][]string{
	"iterator.go": {
		"type Batch struct",
		"type ChatDataIterator struct",
		"func NewChatDataIterator",
		"func (it *ChatDataIterator) HasNext",
		"func (it *ChatDataIterator) Next",
		"func (it *ChatDataIterator) NextBatch",
		"func (it *ChatDataIterator) Reset",
	},
	"initialization.go": {
		"func InitializeXavier",
		"func InitializeHeNormal",
		"func InitializeRouterGating",
		"func InitializeOrthogonal",
		"func InitializeLSTMBias",
		"func isLSTMWeight",
		"func isLSTMBias",
	},
	"loss.go": {
		"func WeightedCrossEntropy",
	},
	"generator.go": {
		"func StrictGenerate",
		"func StrictGenerateWithExperts",
		"func GenerateTokens",
		"func BeamSearchDecode",
		"func BeamSearchDecodeFiltered",
	},
	"chatbot.go": {
		"type DialogRole string",
		"type DialogTurn struct",
		"type ConversationSample struct",
		"type ChatSession struct",
		"func NewChatSession",
		"func (s *ChatSession) AddToHistory",
		"func (s *ChatSession) updateContextVector",
		"func (s *ChatSession) GetContextVector",
		"func StartChat",
		"type MoEChatBot struct",
		"func NewMoEChatBot",
		"func (b *MoEChatBot) Reply",
		"func (b *MoEChatBot) StreamReply",
		"func (b *MoEChatBot) sampleNextToken",
		"func StressTestBot",
	},
	"evaluator.go": {
		"func ValidateChat",
		"func ValidateModelHealth",
		"func VerifyModelIntegrity",
		"func scoreSentenceHeuristic",
		"func scoreGrammarHeuristic",
		"func calculateSequenceReward",
	},
}

func main() {
	sourceFile := "internal/ai/training/chat/chat.go"
	file, err := os.Open(sourceFile)
	if err != nil {
		fmt.Println("Error opening chat.go:", err)
		return
	}
	defer file.Close()

	var lines []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}

	extractedCode := make(map[string][]string)
	for k := range targets {
		extractedCode[k] = []string{}
	}

	var remainingLines []string

	i := 0
	for i < len(lines) {
		matched := false
		for targetFile, patterns := range targets {
			for _, p := range patterns {
				if strings.HasPrefix(lines[i], p) {
					// Extract block
					block, endIdx := extractBlock(lines, i)
					extractedCode[targetFile] = append(extractedCode[targetFile], block...)
					extractedCode[targetFile] = append(extractedCode[targetFile], "")
					i = endIdx
					matched = true
					break
				}
			}
			if matched {
				break
			}
		}

		if !matched {
			remainingLines = append(remainingLines, lines[i])
		}
		i++
	}

	// Write extracted files
	for targetFile, code := range extractedCode {
		if len(code) > 0 {
			f, err := os.Create("internal/ai/training/chat/" + targetFile)
			if err != nil {
				fmt.Println("Error creating", targetFile, ":", err)
				return
			}
			f.WriteString("package chat\n\n")
			for _, line := range code {
				f.WriteString(line + "\n")
			}
			f.Close()
		}
	}

	// Rewrite chat.go
	f, err := os.Create(sourceFile)
	if err != nil {
		fmt.Println("Error writing chat.go:", err)
		return
	}
	for _, line := range remainingLines {
		f.WriteString(line + "\n")
	}
	f.Close()

	fmt.Println("Refactoring complete.")
}

func extractBlock(lines []string, startIdx int) ([]string, int) {
	var block []string
	openBraces := 0
	inBlock := false

	for i := startIdx; i < len(lines); i++ {
		line := lines[i]
		block = append(block, line)

		openBraces += strings.Count(line, "{")
		openBraces -= strings.Count(line, "}")

		if strings.Contains(line, "{") {
			inBlock = true
		}

		if inBlock && openBraces == 0 {
			return block, i
		}
	}

	return block, len(lines) - 1
}
