//go:build ignore

package main

import (
	"bufio"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
)

func main() {
	root := "data/training/trainingdata"
	v := vocab.NewVocabulary()
	// keep special tokens

	tokenRegexp := regexp.MustCompile(`[^\w']+`)

	error := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
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
			line := s.Text()
			// remove punctuation-ish yaml markers
			line = strings.TrimSpace(line)
			if line == "" {
				continue
			}
			// split on non-word characters
			parts := tokenRegexp.Split(line, -1)
			for _, p := range parts {
				p = strings.TrimSpace(p)
				if p == "" {
					continue
				}
				// normalize
				p = strings.Trim(p, "\"'`.,:;()[]{}<>\\/|@#=+*-_~")
				p = strings.TrimSpace(p)
				p = strings.ReplaceAll(p, "\t", " ")
				if p == "" {
					continue
				}
				v.AddToken(p)
			}
		}
		return nil
	})
	if error != nil {
		fmt.Fprintf(os.Stderr, "Walk error: %v\n", error)
		os.Exit(1)
	}

	outGob := "data/models/gob_models/sentence_vocabulary.gob"
	if err := v.Save(outGob); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to save vocab gob: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Saved vocab gob to %s (size=%d)\n", outGob, v.Size())

	// Also export JSON for inspection
	outJSON := "data/models/gob_models/sentence_vocab.json"
	f, err := os.Create(outJSON)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to create json: %v\n", err)
		os.Exit(1)
	}
	defer f.Close()
	// write simple map
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

	fmt.Printf("Exported JSON to %s\n", outJSON)
}
