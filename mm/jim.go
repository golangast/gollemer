package main

import (
	"fmt"
	"net/http"
)

// JimHandler is a sample handler function.
func JimHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "Executing JimHandler! Request URL: %s\n", r.URL.Path)
}
