package main

import (
	"fmt"
	"net/http"
)

func userHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "Hello from userHandler!")
}

func authHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "Hello from authHandler!")
}

func handler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "Hello from handler!")
}

func main() {
	http.HandleFunc("/user", userHandler)
	http.HandleFunc("/auth", authHandler)
	http.HandleFunc("/handle", handler)
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, "OK")
	})
	fmt.Println("Server listening on :8765")
	http.ListenAndServe(":8765", nil)
}
