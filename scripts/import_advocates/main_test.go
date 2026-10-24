package main

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// fixture is a documents.get payload in the real doc's shape: a TOC
// table, a two-column topic (multi-URL cell, duplicate URL, a
// label-only row, a malformed URL, a plain-text URL), a wide
// people table (header row names the columns), and a topic with no
// usable entries at all.
const fixture = `{
  "body": {"content": [
    {"table": {"tableRows": [
      {"tableCells": [
        {"content": [{"paragraph": {"elements": [{"textRun": {"content": "Table of Content\n"}}]}}]},
        {"content": [{"paragraph": {"elements": [{"textRun": {"content": "Servers | People | Empty\n"}}]}}]}
      ]}
    ]}},
    {"table": {"tableRows": [
      {"tableCells": [
        {"content": [{"paragraph": {"elements": [{"textRun": {"content": "Servers\n"}}]}}]},
        {"content": []}
      ]},
      {"tableCells": [
        {"content": [{"paragraph": {"elements": [{"textRun": {"content": "HTTP servers explained\n"}}]}}]},
        {"content": [{"paragraph": {"elements": [
          {"textRun": {"content": "Guide", "textStyle": {"link": {"url": "https://example.com/http-guide"}}}},
          {"textRun": {"content": "\n"}}
        ]}}]}
      ]},
      {"tableCells": [
        {"content": [{"paragraph": {"elements": [{"textRun": {"content": "Two in one cell\n"}}]}}]},
        {"content": [{"paragraph": {"elements": [
          {"textRun": {"content": "First", "textStyle": {"link": {"url": "https://example.com/one"}}}},
          {"textRun": {"content": " and ", "textStyle": {}}},
          {"textRun": {"content": "Second", "textStyle": {"link": {"url": "https://example.com/two"}}}}
        ]}}]}
      ]},
      {"tableCells": [
        {"content": [{"paragraph": {"elements": [{"textRun": {"content": "Same guide again\n"}}]}}]},
        {"content": [{"paragraph": {"elements": [
          {"textRun": {"content": "https://example.com/http-guide", "textStyle": {"link": {"url": "https://example.com/http-guide"}}}}
        ]}}]}
      ]},
      {"tableCells": [
        {"content": [{"paragraph": {"elements": [{"textRun": {"content": "A note with no link\n"}}]}}]},
        {"content": [{"paragraph": {"elements": [{"textRun": {"content": "remember this\n"}}]}}]}
      ]},
      {"tableCells": [
        {"content": [{"paragraph": {"elements": [{"textRun": {"content": "Broken\n"}}]}}]},
        {"content": [{"paragraph": {"elements": [
          {"textRun": {"content": "nope", "textStyle": {"link": {"url": "htp:/bad url"}}}}
        ]}}]}
      ]},
      {"tableCells": [
        {"content": [{"paragraph": {"elements": [{"textRun": {"content": "Typed by hand\n"}}]}}]},
        {"content": [{"paragraph": {"elements": [
          {"textRun": {"content": "see https://example.com/typed for details\n"}}
        ]}}]}
      ]}
    ]}},
    {"table": {"tableRows": [
      {"tableCells": [
        {"content": [{"paragraph": {"elements": [{"textRun": {"content": "People\n"}}]}}]},
        {"content": []}, {"content": []}
      ]},
      {"tableCells": [
        {"content": [{"paragraph": {"elements": [{"textRun": {"content": "Alice\n"}}]}}]},
        {"content": [{"paragraph": {"elements": [{"textRun": {"content": "Bob\n"}}]}}]},
        {"content": []}
      ]},
      {"tableCells": [
        {"content": [{"paragraph": {"elements": [
          {"textRun": {"content": "Blog", "textStyle": {"link": {"url": "https://alice.example.com"}}}}
        ]}}]},
        {"content": [{"paragraph": {"elements": [
          {"textRun": {"content": "Link", "textStyle": {"link": {"url": "https://bob.example.com"}}}}
        ]}}]},
        {"content": []}
      ]}
    ]}},
    {"table": {"tableRows": [
      {"tableCells": [
        {"content": [{"paragraph": {"elements": [{"textRun": {"content": "Empty Topic\n"}}]}}]},
        {"content": []}
      ]},
      {"tableCells": [
        {"content": [{"paragraph": {"elements": [{"textRun": {"content": "Just a thought\n"}}]}}]},
        {"content": [{"paragraph": {"elements": [{"textRun": {"content": "no links here\n"}}]}}]}
      ]}
    ]}}
  ]}
}`

func TestParseDocShapes(t *testing.T) {
	snap, st, err := parseDoc([]byte(fixture))
	if err != nil {
		t.Fatalf("parseDoc: %v", err)
	}
	if st.tablesSeen != 4 {
		t.Errorf("tablesSeen = %d, want 4", st.tablesSeen)
	}
	if len(snap.Topics) != 2 {
		t.Fatalf("topics = %d, want 2 (TOC skipped, empty topic dropped): %+v", len(snap.Topics), snap.Topics)
	}
	servers := snap.Topics[0]
	if servers.Name != "Servers" {
		t.Errorf("topic 0 = %q, want Servers", servers.Name)
	}
	wantURLs := []string{
		"https://example.com/http-guide",
		"https://example.com/one",
		"https://example.com/two",
		"https://example.com/typed",
	}
	if len(servers.Entries) != len(wantURLs) {
		t.Fatalf("Servers entries = %d, want %d: %+v", len(servers.Entries), len(wantURLs), servers.Entries)
	}
	for i, u := range wantURLs {
		if servers.Entries[i].URL != u {
			t.Errorf("entry %d URL = %q, want %q", i, servers.Entries[i].URL, u)
		}
	}
	// The meaningful anchor enriches the row label; the duplicate
	// (anchor == URL) adds nothing.
	if got := servers.Entries[0].Label; got != "HTTP servers explained — Guide" {
		t.Errorf("entry 0 label = %q", got)
	}
	if got := servers.Entries[3].Label; got != "Typed by hand" {
		t.Errorf("plain-text URL entry label = %q", got)
	}
	if st.dupesDropped != 1 {
		t.Errorf("dupesDropped = %d, want 1", st.dupesDropped)
	}
	if st.malformed != 1 {
		t.Errorf("malformed = %d, want 1", st.malformed)
	}
	if st.noURLRows != 2 {
		t.Errorf("noURLRows = %d, want 2 (the Servers note + the Empty Topic row)", st.noURLRows)
	}
	people := snap.Topics[1]
	if people.Name != "People" || len(people.Entries) != 2 {
		t.Fatalf("People topic = %+v", people)
	}
	if people.Entries[0].Label != "Alice — Blog" {
		t.Errorf("people entry 0 label = %q, want the column name + anchor", people.Entries[0].Label)
	}
	if people.Entries[1].Label != "Bob" {
		t.Errorf("people entry 1 label = %q, want Bob (anchor 'Link' is noise)", people.Entries[1].Label)
	}
	if st.topicsEmpty != 1 {
		t.Errorf("topicsEmpty = %d, want 1", st.topicsEmpty)
	}
}

func TestParseDocRejectsGarbage(t *testing.T) {
	if _, _, err := parseDoc([]byte("{not json")); err == nil {
		t.Error("expected an error for malformed JSON")
	}
}

func TestProbeOneVerdicts(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/gone":
			w.WriteHeader(http.StatusNotFound)
		case "/moved-away":
			w.WriteHeader(http.StatusGone)
		case "/flaky":
			w.WriteHeader(http.StatusBadGateway)
		case "/forbidden":
			w.WriteHeader(http.StatusForbidden)
		default:
			w.WriteHeader(http.StatusOK)
		}
	}))
	defer srv.Close()
	client := &http.Client{Timeout: 5 * time.Second}
	cases := map[string]verdict{
		"/ok":         alive,
		"/gone":       definitivelyDead,
		"/moved-away": definitivelyDead,
		"/flaky":      suspect,
		"/forbidden":  alive, // a live server refusing bots is not a dead link
	}
	for path, want := range cases {
		if got := probeOne(client, srv.URL+path); got != want {
			t.Errorf("probeOne(%s) = %v, want %v", path, got, want)
		}
	}
	// A connection error is suspect, never dead.
	if got := probeOne(client, "http://127.0.0.1:1/nothing"); got != suspect {
		t.Errorf("probeOne(unreachable) = %v, want suspect", got)
	}
}

func TestEntryLabelFallback(t *testing.T) {
	if got := entryLabel("", "", "https://example.com/x"); got != "example.com/x" {
		t.Errorf("fallback label = %q", got)
	}
	if got := entryLabel("Row", "https://example.com/x", "https://example.com/x"); got != "Row" {
		t.Errorf("URL-shaped anchor must not enrich: %q", got)
	}
	if !strings.Contains(entryLabel("", "Winscp", "https://winscp.net"), "Winscp") {
		t.Error("meaningful anchor alone should become the label")
	}
}
