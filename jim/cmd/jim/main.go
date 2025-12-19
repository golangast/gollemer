package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"jim/persons"
)

var srv *http.Server

func handler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "Hello from the %s webserver!", "jim")
}

func stopHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "Shutting down the %s webserver!", "jim")
	go func() {
		if err := srv.Shutdown(context.Background()); err != nil {
			log.Printf("HTTP server Shutdown: %v", err)
		}
	}()
}

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("/", handler)
	mux.HandleFunc("/update/persons/", persons.UpdatePersonsHandler)
	mux.HandleFunc("/delete/persons/", persons.DeletePersonsHandler)
	mux.HandleFunc("/stop", stopHandler) // Register the new stop handler
	// HANDLER_REGISTRATIONS_GO_HERE

	srv = &http.Server{
		Addr:    ":8080",
		Handler: mux,
	}

	// Create a channel to listen for OS signals
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		log.Println("Starting webserver on :8080")
		if err := srv.ListenAndServe(); err != http.ErrServerClosed {
			log.Fatalf("HTTP server ListenAndServe: %v", err)
		}
	}()

	// Block until a signal is received (or server is explicitly shut down)
	<-quit
	log.Println("Shutting down webserver...")

	// Give the server a deadline to shutdown
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		log.Fatalf("HTTP server Shutdown: %v", err)
	}

	log.Println("Webserver stopped.")
}
