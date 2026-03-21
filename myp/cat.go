package main

import (
	"fmt"
	"net/http"
)

// CatHandler is a sample handler function.
func CatHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "Executing CatHandler! Request URL: %s\n", r.URL.Path)
}
