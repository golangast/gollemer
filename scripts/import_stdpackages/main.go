// import_stdpackages turns pkg.go.dev's standard-library index
// (https://pkg.go.dev/std) into the snapshot the chat's package
// lookup answers from: data/stdpackages.json — every standard
// package path with its one-line synopsis, in path order, with
// the source recorded.
//
// The chat itself never fetches anything; this importer is the
// build-time half of the feature, the same division of labour as
// scripts/import_advocates: the site stays the source of truth,
// the repo holds a snapshot, the chat reads the snapshot offline.
//
// Refresh:
//
//	go run ./scripts/import_stdpackages
//
// The parser is deliberately structural, not class-based: inside
// the page's Directories table, a row's link href IS the
// package's full path — version-pinned (…/url@go1.27.3), the pin
// stripped for the snapshot — and the row's desktop synopsis
// cell is its one-line description. Group rows (archive,
// compress, …) carry no link and are not packages; rows for
// internal helpers may carry no synopsis.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"html"
	"io"
	"net/http"
	"os"
	"regexp"
	"sort"
	"strings"
	"time"
)

// --- snapshot shape (mirrored by the chat's loader) ---

type Package struct {
	Path     string `json:"path"`
	Synopsis string `json:"synopsis"`
}

type Snapshot struct {
	Source   string    `json:"source"`
	Packages []Package `json:"packages"`
}

const stdSource = "Standard library package index — https://pkg.go.dev/std (snapshot; refresh with: go run ./scripts/import_stdpackages)"

var (
	hrefRe  = regexp.MustCompile(`<a\s[^>]*?href="/([^"#?]+)"`)
	tagRe   = regexp.MustCompile(`<[^>]*>`)
	pathRe  = regexp.MustCompile(`^[a-z0-9._+-]+(/[a-z0-9._+-]+)*$`)
	tableRe = regexp.MustCompile(`(?is)<table.*?</table>`)
)

// assetExts are href tails that name page furniture, never a
// package path (a dotted segment like v1.0.0 is a real path and
// survives: only these endings are refused).
var assetExts = []string{".css", ".js", ".png", ".jpg", ".jpeg", ".svg", ".ico", ".woff", ".woff2", ".map", ".html", ".txt", ".xml"}

// cellText flattens one table cell (or row fragment) to its
// visible text: tags out, entities decoded, whitespace single.
// The fragment starts mid-tag (the split that produced it ate
// the "<td"), so everything through the first ">" — the cell
// tag's own attributes — goes first.
func cellText(fragment string) string {
	if i := strings.Index(fragment, ">"); i >= 0 {
		fragment = fragment[i+1:]
	}
	return strings.Join(strings.Fields(html.UnescapeString(tagRe.ReplaceAllString(fragment, " "))), " ")
}

// rowSynopsis returns the row's synopsis: the desktop synopsis
// cell when the row carries one (it is the site's canonical
// one-liner), else the first cell whose text carries a
// "Package …" line. "" when the row has none (internal helpers
// the index lists bare).
func rowSynopsis(row string) string {
	cells := strings.Split(row, "<td")
	for _, cell := range cells {
		if strings.Contains(cell, "desktopSynopsis") {
			if text := cellText(cell); text != "" {
				return text
			}
		}
	}
	for _, cell := range cells {
		text := cellText(cell)
		if i := strings.Index(text, "Package "); i >= 0 {
			return strings.TrimSpace(text[i:])
		}
	}
	return ""
}

// tableRegion returns the page's Directories table: the one the
// site marks for its own tests, else the first table after the
// Directories heading, else the first table on the page.
func tableRegion(doc string) string {
	if i := strings.Index(doc, `data-test-id="UnitDirectories-table"`); i >= 0 {
		if start := strings.LastIndex(doc[:i], "<table"); start >= 0 {
			if end := strings.Index(doc[start:], "</table>"); end >= 0 {
				return doc[start : start+end+len("</table>")]
			}
		}
	}
	region := doc
	if i := strings.Index(doc, "Directories"); i >= 0 {
		region = doc[i:]
	}
	return tableRe.FindString(region)
}

// parseStdPage extracts the package list from the /std page's
// Directories table. It fails loudly on a shape it cannot read
// (too few rows) so markup drift can never silently snapshot a
// stub catalog.
func parseStdPage(page []byte) ([]Package, error) {
	m := tableRegion(string(page))
	if m == "" {
		return nil, fmt.Errorf("no table found on the page — the /std markup has changed")
	}
	seen := map[string]bool{}
	var out []Package
	for _, row := range strings.Split(m, "<tr") {
		hm := hrefRe.FindStringSubmatch(row)
		if hm == nil {
			continue // a group row (archive, compress, …) or the header
		}
		path := hm[1]
		// The index pins its links to the current release
		// (/net/url@go1.27.3); the snapshot stores the bare path.
		if at := strings.Index(path, "@"); at >= 0 {
			path = path[:at]
		}
		if path == "std" || !pathRe.MatchString(path) {
			continue
		}
		bad := false
		for _, ext := range assetExts {
			if strings.HasSuffix(path, ext) {
				bad = true
				break
			}
		}
		if bad || seen[path] {
			continue
		}
		seen[path] = true
		out = append(out, Package{Path: path, Synopsis: rowSynopsis(row)})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Path < out[j].Path })
	if len(out) < 100 {
		return nil, fmt.Errorf("only %d package rows parsed — the /std markup has changed", len(out))
	}
	return out, nil
}

func main() {
	src := flag.String("url", "https://pkg.go.dev/std", "the standard-library index page to snapshot")
	out := flag.String("out", "data/stdpackages.json", "snapshot file to write")
	dump := flag.String("dump", "", "also write the fetched page here (markup-drift debugging)")
	flag.Parse()

	client := &http.Client{Timeout: 60 * time.Second}
	req, err := http.NewRequest(http.MethodGet, *src, nil)
	if err != nil {
		fmt.Fprintf(os.Stderr, "import_stdpackages: %v\n", err)
		os.Exit(1)
	}
	req.Header.Set("User-Agent", "gollemer import_stdpackages (build-time snapshot; the chat never fetches)")
	resp, err := client.Do(req)
	if err != nil {
		fmt.Fprintf(os.Stderr, "import_stdpackages: fetch %s: %v\n", *src, err)
		os.Exit(1)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		fmt.Fprintf(os.Stderr, "import_stdpackages: fetch %s: %s\n", *src, resp.Status)
		os.Exit(1)
	}
	page, err := io.ReadAll(resp.Body)
	if err != nil {
		fmt.Fprintf(os.Stderr, "import_stdpackages: read %s: %v\n", *src, err)
		os.Exit(1)
	}
	if *dump != "" {
		if err := os.WriteFile(*dump, page, 0o644); err != nil {
			fmt.Fprintf(os.Stderr, "import_stdpackages: dump: %v\n", err)
			os.Exit(1)
		}
	}
	pkgs, err := parseStdPage(page)
	if err != nil {
		fmt.Fprintf(os.Stderr, "import_stdpackages: %v\n", err)
		os.Exit(1)
	}

	blob, err := json.MarshalIndent(Snapshot{Source: stdSource, Packages: pkgs}, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "import_stdpackages: %v\n", err)
		os.Exit(1)
	}
	if err := os.WriteFile(*out, append(blob, '\n'), 0o644); err != nil {
		fmt.Fprintf(os.Stderr, "import_stdpackages: %v\n", err)
		os.Exit(1)
	}

	without := 0
	for _, p := range pkgs {
		if p.Synopsis == "" {
			without++
		}
	}
	fmt.Printf("wrote %s\n", *out)
	fmt.Printf("packages: %d (%d without a synopsis — internal helpers the index lists bare)\n", len(pkgs), without)
	fmt.Printf("first: %s · last: %s\n", pkgs[0].Path, pkgs[len(pkgs)-1].Path)
}
