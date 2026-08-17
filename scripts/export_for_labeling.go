package main

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"os"
)

type Example struct {
	Input    string          `json:"input"`
	Expected json.RawMessage `json:"expected"`
}

func Exporter() {
	in, err := os.Open("data/training/edits_failed.jsonl")
	if err != nil {
		fmt.Fprintf(os.Stderr, "error opening dataset: %v\n", err)
		os.Exit(1)
	}
	defer in.Close()

	out, err := os.Create("data/training/edits_for_labeling.csv")
	if err != nil {
		fmt.Fprintf(os.Stderr, "error creating CSV: %v\n", err)
		os.Exit(1)
	}
	defer out.Close()

	w := csv.NewWriter(out)
	defer w.Flush()
	w.Write([]string{"input", "expected_json", "label", "notes"})

	s := bufio.NewScanner(in)
	for s.Scan() {
		var e Example
		if err := json.Unmarshal([]byte(s.Text()), &e); err != nil {
			continue
		}
		w.Write([]string{e.Input, string(e.Expected), "", ""})
	}
	if err := s.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "scanner error: %v\n", err)
	}
	fmt.Println("Wrote data/training/edits_for_labeling.csv")
}
