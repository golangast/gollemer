package main

import (
	"fmt"
	"net/http"
)

// NamedHandler is a sample handler function.
func NamedHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "Executing NamedHandler! Request URL: %s\n", r.URL.Path)
}
