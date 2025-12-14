
package main

import (
	"fmt"
	"net/http"
)

// HomeHandler is a sample handler function.
func HomeHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "Executing HomeHandler! Request URL: %s\n", r.URL.Path)
}
