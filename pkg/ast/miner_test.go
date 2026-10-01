package ast

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// styleFixture builds a module whose conventions are:
//
//   - errors: two fmt.Errorf with %w, one errors.New  -> "fmt_wrap"
//   - tags:   db x2, json x2, validate x1            -> ["db","json","validate"]
//   - ctx:    2 of 3 exported funcs take ctx first   -> true
//   - docs:   5 of 7 exported idents documented      -> 5/7
func styleFixture(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	write := func(rel, content string) {
		p := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("go.mod", "module example.com/stylemine\n\ngo 1.26.0\n")
	tag1 := "`json:\"name\" db:\"name\" validate:\"required\"`"
	tag2 := "`json:\"id\" db:\"id\"`"
	write("svc/svc.go", `package svc

import (
	"context"
	"errors"
	"fmt"
)

// Service does things.
type Service struct {
	Name string `+tag1+`
	ID   int    `+tag2+`
}

// ErrBad signals a bad request.
var ErrBad = errors.New("bad request")

// NewService builds a Service.
func NewService() *Service {
	return &Service{}
}

// Do does the thing.
func (s *Service) Do(ctx context.Context, id int) error {
	if id < 0 {
		return fmt.Errorf("bad id %d: %w", id, ErrBad)
	}
	return nil
}

// Ping checks liveness.
func (s *Service) Ping(ctx context.Context) error {
	if err := s.Do(ctx, 1); err != nil {
		return fmt.Errorf("ping failed: %w", err)
	}
	return nil
}
`)
	return root
}

// plainFixture builds a module with none of the conventions:
// one undocumented exported func using errors.New.
func plainFixture(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	write := func(rel, content string) {
		p := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("go.mod", "module example.com/plainmine\n\ngo 1.26.0\n")
	write("plain/plain.go", `package plain

import "errors"

func Work() error {
	return errors.New("nope")
}
`)
	return root
}

func TestInferRepositoryStyle(t *testing.T) {
	root := styleFixture(t)
	ctx, err := LoadPackageContext(filepath.Join(root, "svc"))
	if err != nil {
		t.Fatalf("LoadPackageContext: %v", err)
	}
	style, err := InferRepositoryStyle(ctx)
	if err != nil {
		t.Fatalf("InferRepositoryStyle: %v", err)
	}
	if style.ErrorHandlingStyle != ErrorStyleFmtWrap {
		t.Errorf("ErrorHandlingStyle = %q, want %q", style.ErrorHandlingStyle, ErrorStyleFmtWrap)
	}
	wantTags := []string{"db", "json", "validate"}
	if strings.Join(style.CommonStructTags, ",") != strings.Join(wantTags, ",") {
		t.Errorf("CommonStructTags = %v, want %v", style.CommonStructTags, wantTags)
	}
	if !style.UsesContext {
		t.Error("UsesContext = false, want true (2 of 3 exported funcs take ctx)")
	}
	if want := 5.0 / 7.0; style.DocCommentDensity != want {
		t.Errorf("DocCommentDensity = %v, want %v", style.DocCommentDensity, want)
	}

	g := style.SystemPromptGuidelines()
	for _, want := range []string{
		"fmt.Errorf with %w",
		"db, json, validate",
		"ctx context.Context as the first",
		"doc comments for exported identifiers",
	} {
		if !strings.Contains(g, want) {
			t.Errorf("guidelines missing %q:\n%s", want, g)
		}
	}
}

func TestInferRepositoryStylePlain(t *testing.T) {
	root := plainFixture(t)
	ctx, err := LoadPackageContext(filepath.Join(root, "plain"))
	if err != nil {
		t.Fatalf("LoadPackageContext: %v", err)
	}
	style, err := InferRepositoryStyle(ctx)
	if err != nil {
		t.Fatalf("InferRepositoryStyle: %v", err)
	}
	if style.ErrorHandlingStyle != ErrorStyleStandard {
		t.Errorf("ErrorHandlingStyle = %q, want %q", style.ErrorHandlingStyle, ErrorStyleStandard)
	}
	if len(style.CommonStructTags) != 0 {
		t.Errorf("CommonStructTags = %v, want empty", style.CommonStructTags)
	}
	if style.UsesContext {
		t.Error("UsesContext = true, want false")
	}
	if style.DocCommentDensity != 0 {
		t.Errorf("DocCommentDensity = %v, want 0", style.DocCommentDensity)
	}
	g := style.SystemPromptGuidelines()
	if !strings.Contains(g, "errors.New") {
		t.Errorf("guidelines missing errors.New guidance:\n%s", g)
	}
	if strings.Contains(g, "context.Context as the first") {
		t.Errorf("guidelines should not demand ctx:\n%s", g)
	}
}

func TestInferRepositoryStyleErrors(t *testing.T) {
	if _, err := InferRepositoryStyle(nil); err == nil {
		t.Error("expected error for nil context")
	}
	if _, err := InferRepositoryStyle(&CodebaseContext{}); err == nil {
		t.Error("expected error for context without SourceDir")
	}
	var nilStyle *RepoStyle
	if g := nilStyle.SystemPromptGuidelines(); g != "" {
		t.Errorf("nil style guidelines = %q, want empty", g)
	}
}

func TestStructTagKeys(t *testing.T) {
	for _, tc := range []struct {
		in   string
		want string
	}{
		{"`json:\"name\" db:\"id\"`", "json,db"},
		{"`json:\"name,omitempty\"`", "json"},
		{"`json:\"a,b\" xml:\"c\"`", "json,xml"},
		{"``", ""},
		{"`notakey`", ""},
		{"`json:\"a\\\"b\" db:\"c\"`", "json,db"},
	} {
		got := strings.Join(structTagKeys(tc.in), ",")
		if got != tc.want {
			t.Errorf("structTagKeys(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}
