package main

import (
	"fmt"
	"net/http"
)

func user_handler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "Hello from user_handler!")
}

func auth_handler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "Hello from auth_handler!")
}

func handler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "Hello from handler!")
}

func main() {
	http.HandleFunc("/user", user_handler)
	http.
		HandleFunc("/auth_handler",

			auth_handler)
	http.
		HandleFunc("/handler",

			handler)
	http.
		HandleFunc("/auth_handler",

			auth_handler)
	http.
		HandleFunc("/handler",

			handler)
	http.
		HandleFunc("/auth_handler",

			auth_handler)
	http.
		HandleFunc("/auth_handler",

			auth_handler)
	http.
		HandleFunc("/auth_handler",

			auth_handler)
	http.
		HandleFunc("/auth_handler",

			auth_handler)
	http.
		HandleFunc("/handler",

			handler)
	http.
		HandleFunc("/auth_handler",

			auth_handler)
	http.
		HandleFunc("/handler",

			handler)
	http.
		HandleFunc("/auth_handler",

			auth_handler)

}
