package main

import "net/http"
import "fmt"

func NamedHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "Hello from the NamedHandler!")
}
