package analyze

import (
	"testing"
)

// The wiring fact helpers the chat's landing confirmations
// read (HandlerFacts, WireRegistrations): a registered
// handler reports its route and both files, an unregistered
// one reports itself without a route, a non-handler and an
// ambiguous same-name pair report nothing a caller could
// mistake for a fact.

const factsMain = `package main

import "net/http"

func Jake(w http.ResponseWriter, r *http.Request) {}
func Jill(w http.ResponseWriter, r *http.Request) {}
func Plain() {}

func main() {
	http.HandleFunc("/jake", Jake)
	http.ListenAndServe(":8080", nil)
}
`

func TestHandlerFactsRegisteredAndNot(t *testing.T) {
	_, proj := serveProject(t, map[string]string{
		"go.mod":      wireGoMod,
		"app/main.go": factsMain,
	})

	hf := proj.HandlerFacts("Jake", "app")
	if !hf.IsHandler || hf.File != "app/main.go" {
		t.Fatalf("Jake facts = %+v, want the handler in app/main.go", hf)
	}
	if !hf.Registered || hf.Route != "/jake" || hf.RegFile != "app/main.go" {
		t.Fatalf("Jake facts = %+v, want registered at /jake in app/main.go", hf)
	}

	hf = proj.HandlerFacts("Jill", "app")
	if !hf.IsHandler || hf.Registered {
		t.Fatalf("Jill facts = %+v, want a handler that is not registered", hf)
	}

	if hf := proj.HandlerFacts("Plain", "app"); hf.IsHandler {
		t.Fatalf("Plain facts = %+v, want no facts for a non-handler", hf)
	}
	if hf := proj.HandlerFacts("Nobody", "app"); hf.IsHandler || hf.Registered {
		t.Fatalf("Nobody facts = %+v, want no facts for an unknown name", hf)
	}
	// The wrong folder is no fact either.
	if hf := proj.HandlerFacts("Jake", "elsewhere"); hf.IsHandler {
		t.Fatalf("Jake elsewhere facts = %+v, want no facts", hf)
	}
}

func TestHandlerFactsAmbiguousNameReportsNothing(t *testing.T) {
	_, proj := serveProject(t, map[string]string{
		"go.mod": wireGoMod,
		"a/a.go": `package a

import "net/http"

func Same(w http.ResponseWriter, r *http.Request) {}
`,
		"b/b.go": `package b

import "net/http"

func Same(w http.ResponseWriter, r *http.Request) {}
`,
	})
	if hf := proj.HandlerFacts("Same", ""); hf.IsHandler {
		t.Fatalf("ambiguous facts = %+v, want no facts without a folder to choose", hf)
	}
	if hf := proj.HandlerFacts("Same", "b"); !hf.IsHandler || hf.File != "b/b.go" {
		t.Fatalf("folder-chosen facts = %+v, want the handler in b/b.go", hf)
	}
}

func TestWireRegistrationsListsTheProject(t *testing.T) {
	_, proj := serveProject(t, map[string]string{
		"go.mod":      wireGoMod,
		"app/main.go": factsMain,
	})
	regs := proj.WireRegistrations()
	if len(regs) != 1 {
		t.Fatalf("registrations = %+v, want exactly the one in the fixture", regs)
	}
	r := regs[0]
	if r.Name != "Jake" || r.Route != "/jake" || r.File != "app/main.go" || r.HandlerDir != "app" {
		t.Fatalf("registration = %+v, want Jake at /jake in app/main.go (handler dir app)", r)
	}
}
