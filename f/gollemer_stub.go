package j

import (
	"errors"
)

// Temporary stubs to allow the AST orchestrator to build for validation.
// These are safe placeholders and can be removed after Gollemer finishes.
type Router struct{}
type RouteMatch struct{}

var ErrMethodMismatch = errors.New("method mismatch")
var ErrNotFound = errors.New("not found")

// Expose http for any code expecting the package name `http` to be present.
// Provide a package-level `http` identifier so files that reference `http`
// (without importing) compile during validation. It's a minimal stub.
var http = struct{}{}
