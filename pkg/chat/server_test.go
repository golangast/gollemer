package chat

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/golangast/gollemer/pkg/memory"
)

const fixtureGoMod = `module fixturechat

go 1.21
`

const fixtureStoreGo = `package store

import "os"

// Cache is a plug socket: anything shaped like it can be used wherever a Cache is expected.
type Cache interface {
	Get(key string) (string, bool)
	Put(key, value string)
}

// Store is a simple in-memory key/value store.
type Store struct {
	items map[string]string
}

// NewStore builds an empty Store.
func NewStore() *Store {
	return &Store{items: map[string]string{}}
}

// Get fetches the value for key.
func (s *Store) Get(key string) (string, bool) {
	if s == nil {
		return "", false
	}
	v, ok := s.items[key]
	return v, ok
}

// Put saves value under key.
func (s *Store) Put(key, value string) {
	s.items[key] = value
}

// Save writes all keys to a file, one per line.
func (s *Store) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	for k := range s.items {
		if _, err := f.WriteString(k + "\n"); err != nil {
			return err
		}
	}
	return nil
}

// Demo copies every key from src into dst.
func Demo(src, dst *Store) {
	for k := range src.items {
		v, _ := src.Get(k)
		dst.Put(k, v)
	}
}
`

func writeFixture(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte(fixtureGoMod), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "store.go"), []byte(fixtureStoreGo), 0644); err != nil {
		t.Fatal(err)
	}
	return dir
}

func newFixtureServer(t *testing.T) *Server {
	t.Helper()
	srv, err := NewServer(writeFixture(t))
	if err != nil {
		t.Fatalf("NewServer: %v", err)
	}
	return srv
}

func postChat(t *testing.T, srv *Server, body any) (int, ChatResponse) {
	t.Helper()
	raw, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, "/api/chat", bytes.NewReader(raw))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)
	var resp ChatResponse
	if rec.Code == http.StatusOK {
		if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
			t.Fatalf("decode response: %v", err)
		}
	}
	return rec.Code, resp
}

func TestFormatBeginnerExplanation(t *testing.T) {
	cases := []struct {
		kind string
		want []string
	}{
		{"struct", []string{"form", "methods"}},
		{"interface", []string{"plug socket", "IMPLEMENTS"}},
		{"method", []string{"recipe", "belongs"}},
		{"function", []string{"recipe", "CALLS"}},
	}
	for _, c := range cases {
		got := FormatBeginnerExplanation(memory.CodeNode{
			Kind:       c.kind,
			SymbolName: "Example",
			DocComment: "// Example does things.",
		})
		for _, w := range c.want {
			if !strings.Contains(got, w) {
				t.Errorf("kind %q: explanation missing %q:\n%s", c.kind, w, got)
			}
		}
	}
}

func TestChatAPIEndToEnd(t *testing.T) {
	srv := newFixtureServer(t)
	code, resp := postChat(t, srv, ChatRequest{Query: "how does saving to a file work", TopK: 5})
	if code != http.StatusOK {
		t.Fatalf("status = %d", code)
	}
	if resp.Explanation == "" {
		t.Error("empty explanation")
	}
	if len(resp.Nodes) == 0 {
		t.Fatal("no nodes returned")
	}
	// Every edge must reference returned nodes and use the diagram vocabulary.
	in := map[string]bool{}
	kinds := map[string]bool{}
	for _, n := range resp.Nodes {
		in[n.ID] = true
		kinds[n.Kind] = true
		if n.Label == "" || n.Package == "" {
			t.Errorf("node missing label/package: %+v", n)
		}
	}
	for _, e := range resp.Edges {
		if !in[e.Source] || !in[e.Target] {
			t.Errorf("edge references unknown node: %+v", e)
		}
		switch e.Relation {
		case "CALLS", "IMPLEMENTS", "USES":
		default:
			t.Errorf("bad relation %q", e.Relation)
		}
	}
	for k := range kinds {
		switch k {
		case "struct", "interface", "function":
		default:
			t.Errorf("bad kind %q", k)
		}
	}
	// The fixture has an explicit nil check and a deferred Close.
	if resp.SafetySummary.FilesAnalyzed < 1 {
		t.Error("no files analyzed for safety")
	}
	if resp.SafetySummary.VerifiedNullChecks < 1 {
		t.Errorf("verifiedNullChecks = %d, want >= 1", resp.SafetySummary.VerifiedNullChecks)
	}
	if resp.SafetySummary.ResourceDefers < 1 {
		t.Errorf("resourceDefers = %d, want >= 1", resp.SafetySummary.ResourceDefers)
	}
	if resp.SafetySummary.SafetyScore <= 0 || resp.SafetySummary.SafetyScore > 100 {
		t.Errorf("safetyScore = %v out of range", resp.SafetySummary.SafetyScore)
	}
}

func TestChatAPIValidation(t *testing.T) {
	srv := newFixtureServer(t)

	if code, _ := postChat(t, srv, ChatRequest{Query: "  "}); code != http.StatusBadRequest {
		t.Errorf("empty query status = %d, want 400", code)
	}

	req := httptest.NewRequest(http.MethodGet, "/api/chat", nil)
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusMethodNotAllowed {
		t.Errorf("GET status = %d, want 405", rec.Code)
	}

	req = httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader("{oops"))
	rec = httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Errorf("bad JSON status = %d, want 400", rec.Code)
	}
}

func TestChatAPITargetFilter(t *testing.T) {
	srv := newFixtureServer(t)

	_, resp := postChat(t, srv, ChatRequest{Query: "store", Target: "store.go", TopK: 5})
	if len(resp.Nodes) == 0 {
		t.Fatal("target filter removed every node")
	}
	for _, n := range resp.Nodes {
		if n.Package != "store" {
			t.Errorf("node from package %q survived the store.go target filter", n.Package)
		}
	}
	if resp.Explanation == "" {
		t.Error("empty explanation for matching target")
	}

	code, resp := postChat(t, srv, ChatRequest{Query: "store", Target: "does-not-exist.go", TopK: 5})
	if code != http.StatusOK {
		t.Fatalf("unknown target status = %d, want 200 with helpful message", code)
	}
	if len(resp.Nodes) != 0 {
		t.Errorf("unknown target returned %d nodes, want 0", len(resp.Nodes))
	}
	if !strings.Contains(resp.Explanation, "does-not-exist.go") {
		t.Errorf("explanation does not name the bad target: %q", resp.Explanation)
	}
}

func TestChatAPIIndexPage(t *testing.T) {
	srv := newFixtureServer(t)
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("GET / status = %d", rec.Code)
	}
	body := rec.Body.String()
	if !strings.Contains(body, "Gollemer Code Chat") || !strings.Contains(body, "/api/chat") {
		t.Error("index page missing chat UI")
	}
}

func TestVisualEdgesRelationMapping(t *testing.T) {
	g := &memory.KnowledgeGraph{
		Nodes: map[string]memory.CodeNode{
			"a": {ID: "a"},
			"b": {ID: "b"},
		},
		Edges: []memory.CodeEdge{
			{SourceID: "a", TargetID: "b", Relation: memory.RelationCalls},
			{SourceID: "a", TargetID: "b", Relation: memory.RelationImplements},
			{SourceID: "a", TargetID: "b", Relation: "DEPENDS_ON"},
			{SourceID: "a", TargetID: "b", Relation: "INSTANTIATES"},
			{SourceID: "a", TargetID: "zzz", Relation: memory.RelationCalls}, // outside the answer set
		},
	}
	srv := &Server{graph: g}
	vnodes := []VisualNode{{ID: "a"}, {ID: "b"}}
	edges := srv.visualEdges(vnodes)
	if len(edges) != 4 {
		t.Fatalf("edges = %d, want 4 (outside-set edge excluded)", len(edges))
	}
	rels := map[string]int{}
	for _, e := range edges {
		rels[e.Relation]++
	}
	if rels["CALLS"] != 1 || rels["IMPLEMENTS"] != 1 || rels["USES"] != 2 {
		t.Errorf("relation mapping wrong: %v", rels)
	}
}

func TestMapRelation(t *testing.T) {
	if got := mapRelation("BOGUS"); got != "" {
		t.Errorf("mapRelation(BOGUS) = %q, want empty", got)
	}
}
