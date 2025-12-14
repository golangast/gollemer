package main

import (
	"fmt"
	"log"
	"net/http"
)

func handler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "Hello from the %s webserver!", "computer")
}

func main() {
	http.HandleFunc("/", handler)
	http.HandleFunc("/home", HomeHandler)
	http.HandleFunc("/contact-us", ContactHandler)
	log.Println("Starting webserver on :8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}
