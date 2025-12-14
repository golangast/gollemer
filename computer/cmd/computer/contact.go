package main

import (
	"fmt"
	"net/http"
)

func ContactHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "Executing ContactHandler! Request URL: %s\n", r.URL.Path)
}
