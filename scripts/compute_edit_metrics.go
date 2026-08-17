package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
)

type rectr struct {
	Time    string          `json:"time"`
	Query   string          `json:"query"`
	Edits   json.RawMessage `json:"edits"`
	Success bool            `json:"success"`
}

func Editmetrics() {
	f, err := os.Open("logs/edits/edits.log")
	if err != nil {
		fmt.Printf("cannot open logs/edits/edits.log: %v\n", err)
		os.Exit(1)
	}
	defer f.Close()
	s := bufio.NewScanner(f)
	total := 0
	parsed := 0
	success := 0
	for s.Scan() {
		line := s.Text()
		var r rectr
		if err := json.Unmarshal([]byte(line), &r); err != nil {
			continue
		}
		total++
		if len(r.Edits) > 0 && string(r.Edits) != "null" {
			parsed++
		}
		if r.Success {
			success++
		}
	}
	fmt.Printf("Total requests: %d\nParsed (edits non-empty): %d\nSuccessful executions: %d\n", total, parsed, success)
}
