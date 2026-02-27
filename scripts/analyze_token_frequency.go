//go:build ignore

package main

import (
	"encoding/json"
	"fmt"
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

func extractTokens(obj interface{}, counter map[string]int) {
	switch v := obj.(type) {
	case map[string]interface{}:
		for k, val := range v {
			counter[k]++
			extractTokens(val, counter)
		}
	case []interface{}:
		for _, item := range v {
			extractTokens(item, counter)
		}
	default:
		token := fmt.Sprintf("%v", v)
		counter[token]++
	}
}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run analyze_token_frequency.go <path_to_json>")
		os.Exit(1)
	}
	path := os.Args[1]

	file, err := os.ReadFile(path)
	if err != nil {
		fmt.Printf("Error reading file: %v\n", err)
		os.Exit(1)
	}

	var data []map[string]interface{}
	if err := json.Unmarshal(file, &data); err != nil {
		fmt.Printf("Error unmarshalling json: %v\n", err)
		os.Exit(1)
	}

	counter := make(map[string]int)
	for _, entry := range data {
		if semanticOutput, ok := entry["semantic_output"]; ok {
			extractTokens(semanticOutput, counter)
		}
	}

	// Create a slice of pairs to sort the map by value.
	p := make(PairList, len(counter))
	i := 0
	for k, v := range counter {
		p[i] = Pair{k, v}
		i++
	}

	// Sort the slice by value in descending order.
	sort.Sort(sort.Reverse(p))

	fmt.Println("Token Frequency:")
	for _, k := range p {
		fmt.Printf("%s: %d\n", k.Key, k.Value)
	}
}
