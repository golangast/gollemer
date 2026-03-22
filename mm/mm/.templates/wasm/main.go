package main

import (
	"database/sql"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"

	_ "modernc.org/sqlite"
)

var db *sql.DB

func InitDB(filepath string) *sql.DB {
	d, err := sql.Open("sqlite", filepath)
	if err != nil {
		log.Fatalf("Error opening database: %v", err)
	}
	if err = d.Ping(); err != nil {
		log.Fatalf("Error connecting to database: %v", err)
	}

	createTableSQL := `
	CREATE TABLE IF NOT EXISTS webservers (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		name TEXT,
		status TEXT,
		created_at DATETIME DEFAULT CURRENT_TIMESTAMP
	);`

	_, err = d.Exec(createTableSQL)
	if err != nil {
		log.Fatalf("Error creating table 'webservers': %v", err)
	}

	// Seed data if empty
	var count int
	err = d.QueryRow("SELECT COUNT(*) FROM webservers WHERE name = ?", "jim").Scan(&count)
	if err == nil && count == 0 {
		_, _ = d.Exec("INSERT INTO webservers (name, status) VALUES (?, ?)", "jim", "running")
	}

	return d
}

func main() {
	cwd, _ := os.Getwd()
	fmt.Printf("Starting jim webserver in %s\n", cwd)

	dbPath := filepath.Join(cwd, "jim.db")
	db = InitDB(dbPath)
	defer db.Close()

	// Serve the current directory for WASM and static files
	fs := http.FileServer(http.Dir("."))
	
	// Register handlers
	http.HandleFunc("/api/hello", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, "Hello from the jim webserver API!")
	})
	
	// NamedHandler is defined in named.go
	http.HandleFunc("/named", NamedHandler)

	// Fallback to FileServer for WASM/Static
	http.Handle("/", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		log.Printf("%s %s", r.Method, r.URL.Path)
		if filepath.Ext(r.URL.Path) == ".wasm" {
			w.Header().Set("Content-Type", "application/wasm")
		}
		fs.ServeHTTP(w, r)
	}))

	log.Println("Starting webserver on :8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}