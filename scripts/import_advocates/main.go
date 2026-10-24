// import_advocates turns John's "Advocates" Google Doc — his own
// compilation of Go learning links — into the snapshot the chat's
// build-goal resources answer from: data/advocates.json.
//
// The doc's shape: an intro, a table of contents (table 0), then
// one TABLE per topic. In a two-column table each row is (label,
// links). In the wider people/category tables row 1 names the
// columns and the links below belong to that column's name. The
// importer normalizes both shapes into topics with {label, url}
// entries, in doc order: topic order is his curation order, and so
// is entry order within a topic.
//
// Refresh (the doc stays the source of truth; the repo holds a
// snapshot so chat stays offline):
//
//	hatch_gws_cli docs documents get \
//	  --params '{"documentId":"1Zb9GCWPKeEJ4Dyn2TkT-O3wJ8AFc-IMxZzTugNCjr-8"}' \
//	  > advocates_doc.json
//	go run ./scripts/import_advocates -in advocates_doc.json -check
//
// Without -check the import is shape validation only (fully
// offline): entries need a parseable http(s) URL. With -check each
// unique link is probed (HEAD, GET fallback); ONLY a definitive
// 404/410 excludes a link — timeouts, 5xx and connection errors
// are kept and reported as suspects, because a dead network is not
// a dead link (the doc's own intro warns its links rot and to try
// the Wayback Machine).
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"
)

// --- documents.get JSON shape (only the fields the importer reads) ---

type gdoc struct {
	Body struct {
		Content []struct {
			Table *gtable `json:"table"`
		} `json:"content"`
	} `json:"body"`
}

type gtable struct {
	TableRows []struct {
		TableCells []gcell `json:"tableCells"`
	} `json:"tableRows"`
}

type gcell struct {
	Content []struct {
		Paragraph *struct {
			Elements []struct {
				TextRun *struct {
					Content   string `json:"content"`
					TextStyle struct {
						Link *struct {
							URL string `json:"url"`
						} `json:"link"`
					} `json:"textStyle"`
				} `json:"textRun"`
			} `json:"elements"`
		} `json:"paragraph"`
	} `json:"content"`
}

// --- snapshot shape (mirrored by the chat's loader) ---

type Entry struct {
	Label string `json:"label"`
	URL   string `json:"url"`
}

type Topic struct {
	Name    string  `json:"name"`
	Entries []Entry `json:"entries"`
}

type Snapshot struct {
	Source string  `json:"source"`
	Topics []Topic `json:"topics"`
}

type stats struct {
	tablesSeen   int
	topicsKept   int
	topicsEmpty  int
	entriesKept  int
	dupesDropped int
	noURLRows    int
	malformed    int
}

// cellRuns flattens a cell into its text runs, in order.
type run struct {
	text string
	url  string // styled link target, "" when the run is plain text
}

func cellRuns(c gcell) []run {
	var out []run
	for _, el := range c.Content {
		if el.Paragraph == nil {
			continue
		}
		for _, e := range el.Paragraph.Elements {
			if e.TextRun == nil {
				continue
			}
			r := run{text: e.TextRun.Content}
			if e.TextRun.TextStyle.Link != nil {
				r.url = e.TextRun.TextStyle.Link.URL
			}
			out = append(out, r)
		}
	}
	return out
}

func cellText(c gcell) string {
	var b strings.Builder
	for _, r := range cellRuns(c) {
		b.WriteString(r.text)
	}
	return cleanText(b.String())
}

// cleanText collapses the doc's whitespace (every run ends in \n)
// into single spaces.
func cleanText(s string) string {
	return strings.Join(strings.Fields(s), " ")
}

var plainURLRe = regexp.MustCompile(`https?://[^\s<>"'\]\)]+`)

// cellLinks returns the cell's URLs in order, each with the anchor
// text that carried it ("" for a plain-text URL typed into the doc).
func cellLinks(c gcell) []struct{ url, anchor string } {
	var out []struct{ url, anchor string }
	seen := map[string]bool{}
	add := func(u, anchor string) {
		u = strings.TrimRight(u, ".,;:")
		if u == "" || seen[u] {
			return
		}
		seen[u] = true
		out = append(out, struct{ url, anchor string }{u, cleanText(anchor)})
	}
	for _, r := range cellRuns(c) {
		if r.url != "" {
			add(r.url, r.text)
		}
	}
	// Plain-text URLs in unstyled runs (a handful in the real doc).
	for _, r := range cellRuns(c) {
		if r.url != "" {
			continue
		}
		for _, m := range plainURLRe.FindAllString(r.text, -1) {
			add(m, "")
		}
	}
	return out
}

func validURL(u string) bool {
	pu, err := url.Parse(u)
	return err == nil && (pu.Scheme == "http" || pu.Scheme == "https") && pu.Host != ""
}

// meaningfulAnchor reports whether link text adds information over
// the URL itself ("Winscp", "Blog") versus noise ("1.", "Link").
func meaningfulAnchor(anchor, u string) bool {
	if anchor == "" || anchor == u {
		return false
	}
	switch strings.ToLower(anchor) {
	case "link", "here", "click", "url", "site", "website":
		return false
	}
	letters := 0
	for _, r := range anchor {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') {
			letters++
		}
	}
	return letters >= 2
}

// entryLabel composes the printed label: the row/column label,
// enriched with a meaningful anchor when it adds information.
func entryLabel(base, anchor, u string) string {
	label := base
	if meaningfulAnchor(anchor, u) && !strings.Contains(label, anchor) {
		if label == "" {
			label = anchor
		} else {
			label += " — " + anchor
		}
	}
	if label == "" {
		// Never print a bare link with no words: name it by address.
		label = strings.TrimPrefix(strings.TrimPrefix(u, "https://"), "http://")
	}
	return label
}

// parseDoc converts a documents.get payload into the snapshot.
func parseDoc(data []byte) (*Snapshot, stats, error) {
	var d gdoc
	if err := json.Unmarshal(data, &d); err != nil {
		return nil, stats{}, fmt.Errorf("parse document JSON: %w", err)
	}
	var st stats
	snap := &Snapshot{Source: "Advocates — John's Go learning links (Google Doc snapshot)"}
	for ti, el := range d.Body.Content {
		if el.Table == nil {
			continue
		}
		st.tablesSeen++
		t := el.Table
		if len(t.TableRows) == 0 || len(t.TableRows[0].TableCells) == 0 {
			continue
		}
		title := cellText(t.TableRows[0].TableCells[0])
		if ti == 0 || strings.Contains(strings.ToLower(title), "table of content") {
			continue // the TOC names the topics; it is not one
		}
		topic := Topic{Name: title}
		seen := map[string]bool{}
		addEntry := func(label, u string) {
			if !validURL(u) {
				st.malformed++
				return
			}
			if seen[u] {
				st.dupesDropped++
				return
			}
			seen[u] = true
			topic.Entries = append(topic.Entries, Entry{Label: label, URL: u})
		}
		ncols := 0
		for _, r := range t.TableRows {
			if len(r.TableCells) > ncols {
				ncols = len(r.TableCells)
			}
		}
		if ncols <= 2 {
			// (label, links) rows.
			for _, row := range t.TableRows[1:] {
				if len(row.TableCells) == 0 {
					continue
				}
				label := cellText(row.TableCells[0])
				var links []struct{ url, anchor string }
				for _, c := range row.TableCells {
					links = append(links, cellLinks(c)...)
				}
				if len(links) == 0 {
					if label != "" {
						st.noURLRows++
					}
					continue
				}
				for _, l := range links {
					addEntry(entryLabel(label, l.anchor, l.url), l.url)
				}
			}
		} else {
			// Column tables: row 1 names the columns; a link in
			// column j belongs to that column's name.
			headers := make([]string, ncols)
			if len(t.TableRows) > 1 {
				for j, c := range t.TableRows[1].TableCells {
					headers[j] = cellText(c)
				}
			}
			for ri, row := range t.TableRows {
				if ri == 0 {
					continue
				}
				for j, c := range row.TableCells {
					links := cellLinks(c)
					if len(links) == 0 {
						continue
					}
					base := ""
					if j < len(headers) {
						base = headers[j]
					}
					if base == "" && len(row.TableCells) > 0 {
						base = cellText(row.TableCells[0])
					}
					for _, l := range links {
						addEntry(entryLabel(base, l.anchor, l.url), l.url)
					}
				}
			}
		}
		if len(topic.Entries) == 0 {
			st.topicsEmpty++
			continue
		}
		st.topicsKept++
		st.entriesKept += len(topic.Entries)
		snap.Topics = append(snap.Topics, topic)
	}
	return snap, st, nil
}

// --- link checking (-check) ---

type verdict int

const (
	alive verdict = iota
	definitivelyDead
	suspect
)

// probeOne classifies one URL. Only a 404/410 — from HEAD, or from
// the GET fallback when HEAD is refused or fails — is definitive.
func probeOne(client *http.Client, u string) verdict {
	head, err := client.Head(u)
	if err == nil {
		head.Body.Close()
		if head.StatusCode == 404 || head.StatusCode == 410 {
			return definitivelyDead
		}
		if head.StatusCode < 500 && head.StatusCode != 405 && head.StatusCode != 501 {
			return alive
		}
	}
	resp, err := client.Get(u)
	if err != nil {
		return suspect
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 2048))
	if resp.StatusCode == 404 || resp.StatusCode == 410 {
		return definitivelyDead
	}
	if resp.StatusCode >= 500 {
		return suspect
	}
	return alive
}

// probeAll probes each unique URL with bounded concurrency.
func probeAll(urls []string, workers int, timeout time.Duration) map[string]verdict {
	client := &http.Client{
		Timeout: timeout,
		Transport: &http.Transport{
			Proxy:               http.ProxyFromEnvironment,
			TLSHandshakeTimeout: timeout,
		},
	}
	out := make(map[string]verdict, len(urls))
	var mu sync.Mutex
	var wg sync.WaitGroup
	sem := make(chan struct{}, workers)
	for _, u := range urls {
		wg.Add(1)
		go func(u string) {
			defer wg.Done()
			sem <- struct{}{}
			v := probeOne(client, u)
			<-sem
			mu.Lock()
			out[u] = v
			mu.Unlock()
		}(u)
	}
	wg.Wait()
	return out
}

func main() {
	in := flag.String("in", "advocates_doc.json", "documents.get JSON export of the Advocates doc")
	out := flag.String("out", "data/advocates.json", "snapshot file to write")
	check := flag.Bool("check", false, "probe links; exclude only definitive 404/410 deaths")
	flag.Parse()

	data, err := os.ReadFile(*in)
	if err != nil {
		fmt.Fprintf(os.Stderr, "import_advocates: %v\n", err)
		os.Exit(1)
	}
	snap, st, err := parseDoc(data)
	if err != nil {
		fmt.Fprintf(os.Stderr, "import_advocates: %v\n", err)
		os.Exit(1)
	}

	var dead, suspects []string
	if *check {
		uniq := map[string]bool{}
		for _, t := range snap.Topics {
			for _, e := range t.Entries {
				uniq[e.URL] = true
			}
		}
		urls := make([]string, 0, len(uniq))
		for u := range uniq {
			urls = append(urls, u)
		}
		sort.Strings(urls)
		fmt.Printf("probing %d unique links…\n", len(urls))
		verdicts := probeAll(urls, 20, 8*time.Second)
		deadSet := map[string]bool{}
		for u, v := range verdicts {
			switch v {
			case definitivelyDead:
				deadSet[u] = true
				dead = append(dead, u)
			case suspect:
				suspects = append(suspects, u)
			}
		}
		sort.Strings(dead)
		sort.Strings(suspects)
		kept := snap.Topics[:0]
		removed := 0
		for _, t := range snap.Topics {
			es := t.Entries[:0]
			for _, e := range t.Entries {
				if deadSet[e.URL] {
					removed++
					continue
				}
				es = append(es, e)
			}
			t.Entries = es
			if len(es) > 0 {
				kept = append(kept, t)
			} else {
				st.topicsEmpty++
				st.topicsKept--
			}
		}
		snap.Topics = kept
		st.entriesKept -= removed
	}

	blob, err := json.MarshalIndent(snap, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "import_advocates: %v\n", err)
		os.Exit(1)
	}
	if err := os.WriteFile(*out, append(blob, '\n'), 0o644); err != nil {
		fmt.Fprintf(os.Stderr, "import_advocates: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("wrote %s\n", *out)
	fmt.Printf("tables seen: %d · topics kept: %d · topics dropped (no usable entries): %d\n",
		st.tablesSeen, st.topicsKept, st.topicsEmpty)
	fmt.Printf("entries kept: %d · duplicates dropped: %d · label rows without a URL: %d · malformed URLs dropped: %d\n",
		st.entriesKept, st.dupesDropped, st.noURLRows, st.malformed)
	if *check {
		fmt.Printf("link check: %d definitively dead (excluded), %d suspect (kept)\n", len(dead), len(suspects))
		for _, u := range dead {
			if len(u) > 0 {
				fmt.Printf("  dead: %s\n", u)
			}
		}
		for _, u := range suspects {
			fmt.Printf("  suspect: %s\n", u)
		}
	}
}
