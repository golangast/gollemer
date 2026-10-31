package main

import (
	"strconv"
	"strings"
	"testing"
)

// A fixture in the /std page's real shape (2026-10-31): a
// shortcuts table and a "Directories" option before the index
// table (neither is the index), the table marked with its
// data-test-id, version-pinned hrefs, mobile + desktop synopsis
// cells, a group row (no link — not a package), an internal row
// with no synopsis, a versioned internal path, and an entity in
// a synopsis.
const stdPageFixture = `<!DOCTYPE html>
<html><head><title>Standard library</title></head><body>
<table class="ShortcutsDialog"><tr><td class="ShortcutsDialog-key">?</td></tr></table>
<select><option value="std">Directories</option></select>
<div class="UnitDirectories js-unitDirectories">
<h2 class="UnitDirectories-title" id="section-directories">Directories</h2>
<table class="UnitDirectories-table UnitDirectories-table--tree" data-test-id="UnitDirectories-table">
<tr class="UnitDirectories-tableHeader"><th>Path</th><th class="UnitDirectories-desktopSynopsis">Synopsis</th></tr>
<tr data-aria-controls="archive-tar archive-zip"><td data-id="archive"><div>archive</div></td><td class="UnitDirectories-desktopSynopsis"></td></tr>
<tr data-id="archive-tar"><td><div class="UnitDirectories-subdirectory"><span><a href="/archive/tar@go1.27.3">tar</a></span><div class="UnitDirectories-mobileSynopsis">Package tar implements access to tar archives.</div></div></td><td class="UnitDirectories-desktopSynopsis">Package tar implements access to tar archives.</td></tr>
<tr data-id="archive-zip"><td><div class="UnitDirectories-subdirectory"><span><a href="/archive/zip@go1.27.3">zip</a></span><div class="UnitDirectories-mobileSynopsis">Package zip provides support for reading and writing ZIP archives.</div></div></td><td class="UnitDirectories-desktopSynopsis">Package zip provides support for reading and writing ZIP archives.</td></tr>
<tr data-id="bufio"><td><div><span><a href="/bufio@go1.27.3">bufio</a></span><div class="UnitDirectories-mobileSynopsis">Package bufio implements buffered I/O.</div></div></td><td class="UnitDirectories-desktopSynopsis">Package bufio implements buffered I/O.</td></tr>
<tr data-id="crypto-internal-boring-bbig"><td><div class="UnitDirectories-subdirectory"><span><a href="/crypto/internal/boring/bbig@go1.27.3">internal/boring/bbig</a></span></div></td><td class="UnitDirectories-desktopSynopsis"></td></tr>
<tr data-id="crypto-internal-entropy-v1-0-0"><td><div class="UnitDirectories-subdirectory"><span><a href="/crypto/internal/entropy/v1.0.0@go1.27.3">internal/entropy/v1.0.0</a></span><div class="UnitDirectories-mobileSynopsis">Package entropy implements a CPU jitter-based entropy source &amp; more.</div></div></td><td class="UnitDirectories-desktopSynopsis">Package entropy implements a CPU jitter-based entropy source &amp; more.</td></tr>
</table>
</div>
</body></html>`

func TestParseStdPageFixture(t *testing.T) {
	// The fixture is far below the production floor of 100 rows;
	// pad it with generated rows so the same code path the floor
	// guards parses it.
	page := stdPageFixture
	var pad strings.Builder
	for i := 0; i < 120; i++ {
		pad.WriteString(`<tr data-id="pad-pkg` + strconv.Itoa(i) + `"><td><span><a href="/pad/pkg` + strconv.Itoa(i) + `@go1.27.3">p</a></span></td><td class="UnitDirectories-desktopSynopsis">Package p is padding.</td></tr>`)
	}
	page = strings.ReplaceAll(page, "</table>", pad.String()+"</table>")

	pkgs, err := parseStdPage([]byte(page))
	if err != nil {
		t.Fatalf("parseStdPage: %v", err)
	}
	byPath := map[string]string{}
	for _, p := range pkgs {
		byPath[p.Path] = p.Synopsis
	}
	for path, want := range map[string]string{
		"archive/tar":                    "Package tar implements access to tar archives.",
		"archive/zip":                    "Package zip provides support for reading and writing ZIP archives.",
		"bufio":                          "Package bufio implements buffered I/O.",
		"crypto/internal/boring/bbig":    "",
		"crypto/internal/entropy/v1.0.0": "Package entropy implements a CPU jitter-based entropy source & more.",
	} {
		got, ok := byPath[path]
		if !ok {
			t.Errorf("package %q missing", path)
			continue
		}
		if got != want {
			t.Errorf("package %q synopsis = %q, want %q", path, got, want)
		}
	}
	// No version pins survive into the snapshot.
	for p := range byPath {
		if strings.Contains(p, "@") {
			t.Errorf("version pin leaked into path %q", p)
		}
	}
	// The group row is not a package.
	if _, ok := byPath["archive"]; ok {
		t.Error("group row parsed as a package")
	}
	// Sorted, path order.
	for i := 1; i < len(pkgs); i++ {
		if pkgs[i-1].Path >= pkgs[i].Path {
			t.Fatalf("packages out of order at %d: %q >= %q", i, pkgs[i-1].Path, pkgs[i].Path)
		}
	}
}

func TestParseStdPageRefusesStub(t *testing.T) {
	if _, err := parseStdPage([]byte("<html><body><p>no table here</p></body></html>")); err == nil {
		t.Error("a page with no Directories table parsed without error")
	}
	if _, err := parseStdPage([]byte(stdPageFixture)); err == nil {
		t.Error("a six-row table parsed without error — the floor should refuse a stub")
	}
}
