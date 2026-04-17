package main

import (
	"fmt"
	"log"
	"net/http"
)

func helloHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintln(w, "Hello, World!")
}

func pingHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintln(w, "Ping received!")
}

func main() {
	http.HandleFunc("/", helloHandler)
	http.HandleFunc("/ping", pingHandler)
	log.Println("Starting server on :8080")
	if err := http.ListenAndServe(":8080", nil); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}