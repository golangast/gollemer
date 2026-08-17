package j

import "errors"

// Package-main stubs to satisfy validation when an injected file switches to
// `package main`. These are minimal and safe placeholders.

type Router struct{}
type RouteMatch struct{}

var ErrMethodMismatch = errors.New("method mismatch")
var ErrNotFound = errors.New("not found")

var http = struct{}{}
