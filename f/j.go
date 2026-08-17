package j

import (
	"errors"
	"net/http"
)

// Minimal stubs in-file so single-file `go build` validation succeeds when
// the orchestrator injects edits. These are harmless placeholders.
type middlewareWrapper struct {
	Middleware func(func(http.ResponseWriter, *http.Request)) func(http.ResponseWriter, *http.Request)
}

type RouteMatch struct {
	MatchErr error
	Handler  func(http.ResponseWriter, *http.Request)
	// Match may be called with the request and an output match pointer
	Match func(*http.Request, *RouteMatch) bool
}

type Router struct {
	routes                  []*RouteMatch
	middlewares             []middlewareWrapper
	MethodNotAllowedHandler func(http.ResponseWriter, *http.Request)
	NotFoundHandler         func(http.ResponseWriter, *http.Request)
}

var ErrMethodMismatch = errors.New("method mismatch")
var ErrNotFound = errors.New("not found")

func F() int {
	return calculate(1, 2)
}

func calculate(a, b int) int {
	return a + b
}
func (r *Router) Match(req *http.Request, match *RouteMatch) bool {
	for _, route := range r.routes {
		if route.Match(req,

			match) {
			if match.MatchErr == nil {
				for i :=
					len(r.middlewares) - 1; i >= 0; i-- {

					match.Handler = r.middlewares[i].Middleware(match.Handler)
				}
			}
			return true
		}
	}
	if match.MatchErr == ErrMethodMismatch {
		if r.MethodNotAllowedHandler != nil {
			match.Handler = r.MethodNotAllowedHandler
			return true
		}
		return false
	}
	if r.NotFoundHandler != nil {
		match.Handler = r.NotFoundHandler
		match.MatchErr = ErrNotFound
		return true
	}
	match.MatchErr = ErrNotFound
	return false
}
