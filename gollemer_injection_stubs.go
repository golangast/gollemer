package main

import "errors"

// Stubs used only for build-time validation of injected edits by the AST orchestrator.
// These provide minimal symbols that some injected edits may reference. They are
// harmless and can be removed after orchestration succeeds.

type Router struct{}
type RouteMatch struct{}

var ErrMethodMismatch = errors.New("method mismatch")
var ErrNotFound = errors.New("not found")

// Provide a package-level `http` identifier to satisfy ad-hoc references.
var http = struct{}{}
