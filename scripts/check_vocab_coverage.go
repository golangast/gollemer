package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"sort"
)

// Pair is a struct to hold key-value pairs for sorting.
type Pair struct {
	Key   string
	Value int
}

// PairList is a slice of Pairs that implements sort.Interface.
type PairList []Pair

func (p PairList) Len() int           { return len(p) }
func (p PairList) Less(i, j int) bool { return p[i].Value < p[j].Value }
func (p PairList) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }

func findMissing(obj interface{}, vocab map[string]int, missing map[string]int) {
	switch v := obj.(type) {
	case map[string]interface{}:
		for k, val := range v {
			if _, exists := vocab[k]; !exists {
				missing[k]++
			}
			findMissing(val, vocab, missing)
		}
	case []interface{}:
		for _, item := range v {
			findMissing(item, vocab, missing)
		}
	default:
		token := fmt.Sprintf("%v", v)
		if _, exists := vocab[token]; !exists {
			missing[token]++
		}
	}
}

func main() {
	if len(os.Args) != 3 {
		fmt.Println("Usage: go run check_vocab_coverage.go <vocab.json> <data.json>")
		os.Exit(1)
	}
	vocabPath := os.Args[1]
	dataPath := os.Args[2]

	// Read and parse the vocabulary file
	vocabFile, err := ioutil.ReadFile(vocabPath)
	if err != nil {
		fmt.Printf("Error reading vocab file: %v\n", err)
		os.Exit(1)
	}

	var vocab map[string]int
	if err := json.Unmarshal(vocabFile, &vocab); err != nil {
		fmt.Printf("Error unmarshalling vocab json: %v\n", err)
		os.Exit(1)
	}

	// Read and parse the data file
	dataFile, err := ioutil.ReadFile(dataPath)
	if err != nil {
		fmt.Printf("Error reading data file: %v\n", err)
		os.Exit(1)
	}

	var data []map[string]interface{}
	if err := json.Unmarshal(dataFile, &data); err != nil {
		fmt.Printf("Error unmarshalling data json: %v\n", err)
		os.Exit(1)
	}

	missing := make(map[string]int)
	for _, entry := range data {
		if semanticOutput, ok := entry["semantic_output"]; ok {
			findMissing(semanticOutput, vocab, missing)
		}
	}

	// Create a slice of pairs to sort the map by value.
	p := make(PairList, len(missing))
	i := 0
	for k, v := range missing {
		p[i] = Pair{k, v}
		i++
	}

	// Sort the slice by value in descending order.
	sort.Sort(sort.Reverse(p))

	fmt.Println("Missing tokens:")
	for _, k := range p {
		fmt.Printf("%s: %d\n", k.Key, k.Value)
	}
}
