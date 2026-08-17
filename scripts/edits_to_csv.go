package main

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"os"
)

type rect struct {
	Time    string          `json:"time"`
	Query   string          `json:"query"`
	Edits   json.RawMessage `json:"edits"`
	Success bool            `json:"success"`
}

func EditoCSV() {
	in, err := os.Open("logs/edits/edits.log")
	if err != nil {
		fmt.Fprintf(os.Stderr, "error opening edits.log: %v\n", err)
		os.Exit(1)
	}
	defer in.Close()

	out, err := os.Create("logs/edits/edits.csv")
	if err != nil {
		fmt.Fprintf(os.Stderr, "error creating edits.csv: %v\n", err)
		os.Exit(1)
	}
	defer out.Close()

	w := csv.NewWriter(out)
	defer w.Flush()

	// header
	w.Write([]string{"time", "query", "parsed_edits_count", "success", "edits_json"})

	s := bufio.NewScanner(in)
	for s.Scan() {
		var r rect
		if err := json.Unmarshal([]byte(s.Text()), &r); err != nil {
			continue
		}
		count := 0
		if len(r.Edits) > 0 && string(r.Edits) != "null" {
			// try to parse as array
			var arr []interface{}
			if err := json.Unmarshal(r.Edits, &arr); err == nil {
				count = len(arr)
			}
		}
		w.Write([]string{r.Time, r.Query, fmt.Sprintf("%d", count), fmt.Sprintf("%v", r.Success), string(r.Edits)})
	}
	if err := s.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "scanner error: %v\n", err)
	}
	fmt.Println("Wrote logs/edits/edits.csv")
}
