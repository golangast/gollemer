package main

import (
	"fmt"
	"net/http"
)

// JimHandler handles requests for /jim
func JimHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "Hello from the Jim handler!")
}
