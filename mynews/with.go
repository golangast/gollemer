package main

import (
	"fmt"
	"net/http"
)

// WithHandler handles requests for /jim
func WithHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "Hello from the With handler!")
}
