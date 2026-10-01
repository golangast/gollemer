package ast

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// impactFixture builds a tiny three-package module:
//
//	example.com/impactdemo
//	├── go.mod
//	├── main.go            (package main: uses models.User, calls store.Get)
//	├── models/user.go     (package models: type User, func NewUser)
//	└── store/store.go     (package store: type Store, method Get, plus a
//	                        function-local "type User" that shadows nothing
//	                        outside its own function body)
func impactFixture(t *testing.T) string {
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
	write("go.mod", "module example.com/impactdemo\n\ngo 1.26.0\n")
	write("models/user.go", `package models

// User is a person.
type User struct {
	Name string
}

// NewUser builds a User.
func NewUser(name string) *User {
	return &User{Name: name}
}
`)
	write("store/store.go", `package store

import "example.com/impactdemo/models"

// Store keeps users.
type Store struct {
	users map[int]models.User
}

// Get returns a user by id.
func (s *Store) Get(id int) (models.User, error) {
	if u, ok := s.users[id]; ok {
		return u, nil
	}
	return models.User{}, nil
}

func shadow() {
	type User string // local type: must NOT join the impact radius
	var u User
	_ = u
}
`)
	write("main.go", `package main

import (
	"example.com/impactdemo/models"
	"example.com/impactdemo/store"
)

var Default = models.User{Name: "root"}

func main() {
	u := models.NewUser("cli")
	_ = u
	var st store.Store
	_, _ = st.Get(1)
}
`)
	return root
}

func impactFiles(refs []string) map[string]bool {
	files := make(map[string]bool)
	for _, r := range refs {
		if i := strings.Index(r, ":"); i >= 0 {
			files[r[:i]] = true
		}
	}
	return files
}

func TestCalculateImpactRadiusType(t *testing.T) {
	root := impactFixture(t)
	ctx, err := LoadPackageContext(filepath.Join(root, "models"))
	if err != nil {
		t.Fatalf("LoadPackageContext: %v", err)
	}
	refs, err := CalculateImpactRadius(ctx, "User")
	if err != nil {
		t.Fatalf("CalculateImpactRadius: %v", err)
	}
	// 3 in models/user.go (decl, *User, &User{}), 3 in store/store.go
	// (field, result, literal), 1 in main.go. The function-local
	// "type User" in shadow() must not appear.
	if len(refs) != 7 {
		t.Fatalf("got %d refs, want 7:\n%s", len(refs), strings.Join(refs, "\n"))
	}
	files := impactFiles(refs)
	for _, want := range []string{"models/user.go", "store/store.go", "main.go"} {
		if !files[filepath.FromSlash(want)] {
			t.Errorf("missing dependent file %s in %v", want, refs)
		}
	}
	if len(files) != 3 {
		t.Errorf("got %d dependent files, want 3: %v", len(files), files)
	}
	for _, r := range refs {
		if !strings.HasSuffix(r, ": User") {
			t.Errorf("malformed ref entry %q", r)
		}
	}
	// The shadow function's local type must not leak into the radius:
	// the only legitimate store.go references are the field type
	// (line 7), the result type (line 11) and the literal (line 15).
	for _, r := range refs {
		if !strings.HasPrefix(r, "store/store.go:") {
			continue
		}
		var line int
		if _, err := fmt.Sscanf(r, "store/store.go:%d:", &line); err != nil {
			t.Fatalf("cannot parse line from %q", r)
		}
		if line != 7 && line != 11 && line != 15 {
			t.Errorf("shadowed local type leaked into radius: %q", r)
		}
	}
}

func TestCalculateImpactRadiusMethod(t *testing.T) {
	root := impactFixture(t)
	ctx, err := LoadPackageContext(filepath.Join(root, "store"))
	if err != nil {
		t.Fatalf("LoadPackageContext: %v", err)
	}
	for _, sym := range []string{"Store.Get", "(*Store).Get"} {
		refs, err := CalculateImpactRadius(ctx, sym)
		if err != nil {
			t.Fatalf("CalculateImpactRadius(%q): %v", sym, err)
		}
		files := impactFiles(refs)
		if len(refs) != 2 {
			t.Fatalf("%q: got %d refs, want 2 (decl + st.Get call):\n%s",
				sym, len(refs), strings.Join(refs, "\n"))
		}
		if !files[filepath.FromSlash("store/store.go")] || !files[filepath.FromSlash("main.go")] {
			t.Errorf("%q: wrong files: %v", sym, files)
		}
	}
}

func TestCalculateImpactRadiusErrors(t *testing.T) {
	root := impactFixture(t)
	ctx, err := LoadPackageContext(filepath.Join(root, "models"))
	if err != nil {
		t.Fatalf("LoadPackageContext: %v", err)
	}
	for _, tc := range []struct {
		name string
		ctx  *CodebaseContext
		sym  string
	}{
		{"nil context", nil, "User"},
		{"empty symbol", ctx, ""},
		{"blank symbol", ctx, "   "},
		{"unknown symbol", ctx, "NoSuchThing"},
		{"unknown method", ctx, "User.NoSuchMethod"},
		{"receiver not a type", ctx, "Nope.Get"},
		{"invalid symbol", ctx, "foo-bar"},
		{"no source dir", &CodebaseContext{ModulePath: "example.com/impactdemo"}, "User"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := CalculateImpactRadius(tc.ctx, tc.sym); err == nil {
				t.Errorf("expected error for symbol %q", tc.sym)
			}
		})
	}
}

func TestParseImpactSymbol(t *testing.T) {
	for _, tc := range []struct {
		in       string
		recv     string
		name     string
		isMethod bool
		ok       bool
	}{
		{"User", "", "User", false, true},
		{"NewUser", "", "NewUser", false, true},
		{"Store.Get", "Store", "Get", true, true},
		{"(*Store).Get", "Store", "Get", true, true},
		{"(Store).Get", "Store", "Get", true, true},
		{"foo-bar", "", "", false, false},
		{"", "", "", false, false},
		{".Get", "", "", false, false},
	} {
		recv, name, isMethod, err := parseImpactSymbol(tc.in)
		if tc.ok && err != nil {
			t.Errorf("%q: unexpected error %v", tc.in, err)
		}
		if !tc.ok && err == nil {
			t.Errorf("%q: expected error", tc.in)
		}
		if err == nil && (recv != tc.recv || name != tc.name || isMethod != tc.isMethod) {
			t.Errorf("%q: got (%q,%q,%v)", tc.in, recv, name, isMethod)
		}
	}
}
