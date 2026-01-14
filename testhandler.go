package main

import (
	"fmt"
	"net/http"
)

// TesthandlerHandler is a sample handler function.
func TesthandlerHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "Executing TesthandlerHandler! Request URL: %s\n", r.URL.Path)
}
