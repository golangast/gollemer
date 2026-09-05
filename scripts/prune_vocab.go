//go:build ignore

package main

import (
	"bufio"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
)

func main() {
	maxKeep := 2000
	if len(os.Args) > 1 {
		if v, err := strconv.Atoi(os.Args[1]); err == nil && v > 10 {
			maxKeep = v
		}
	}
	root := "data/training/trainingdata"
	tokenRegexp := regexp.MustCompile(`[^
\w']+`)
	counts := make(map[string]int)

	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		if !strings.HasSuffix(path, ".yaml") && !strings.HasSuffix(path, ".csv") && !strings.HasSuffix(path, ".txt") {
			return nil
		}
		f, err := os.Open(path)
		if err != nil {
			return err
		}
		defer f.Close()
		s := bufio.NewScanner(f)
		for s.Scan() {
			line := strings.TrimSpace(s.Text())
			if line == "" {
				continue
			}
			parts := tokenRegexp.Split(line, -1)
			for _, p := range parts {
				p = strings.TrimSpace(p)
				if p == "" {
					continue
				}
				p = strings.Trim(p, "\"'`.,:;()[]{}<>\\/|@#=+*-_~")
				p = strings.TrimSpace(p)
				p = strings.ReplaceAll(p, "\t", " ")
				if p == "" {
					continue
				}
				counts[p]++
			}
		}
		return nil
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "walk error: %v\n", err)
		os.Exit(1)
	}

	// Build slice and sort by count desc
	tokens := make([]struct {
		Tok string
		Cnt int
	}, 0, len(counts))
	for w, c := range counts {
		tokens = append(tokens, struct {
			Tok string
			Cnt int
		}{w, c})
	}
	sort.Slice(tokens, func(i, j int) bool { return tokens[i].Cnt > tokens[j].Cnt })

	keep := maxKeep
	if keep > len(tokens) {
		keep = len(tokens)
	}

	selected := tokens[:keep]

	// create map with special tokens first
	// use NewVocabulary to ensure special tokens are present
	v := vocab.NewVocabulary()
	// add selected tokens
	for _, t := range selected {
		v.AddToken(t.Tok)
	}

	outGob := "data/models/gob_models/sentence_vocabulary.gob"
	if err := v.Save(outGob); err != nil {
		fmt.Fprintf(os.Stderr, "failed to save vocab gob: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Saved pruned vocab gob to %s (size=%d)\n", outGob, v.Size())

	outJSON := "data/models/gob_models/sentence_vocab.json"
	f, err := os.Create(outJSON)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to create json: %v\n", err)
		os.Exit(1)
	}
	defer f.Close()

	fmt.Fprintln(f, "{")
	first := true
	for w, id := range v.WordToToken {
		if !first {
			fmt.Fprintln(f, ",")
		}
		first = false
		fmt.Fprintf(f, "  \"%s\": %d", w, id)
	}
	fmt.Fprintln(f, "\n}")

	fmt.Printf("Exported pruned JSON to %s\n", outJSON)
}
