package main

import (
	"database/sql"
	"fmt"
	"html/template"
	"log"
	"net/http"
	_ "net/http/pprof" // Go 1.26: Includes new /debug/pprof/goroutineleak profile
	"os"
	"path/filepath"

	"github.com/golangast/gollemer/learningfolder/pkg/render"
	"github.com/golangast/gollemer/learningfolder/routes"
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
	dbPath := filepath.Join(cwd, "jim.db")
	db = InitDB(dbPath)
	defer db.Close()

	// Parse all template files recursively
	templatePath := filepath.Join(cwd, "templates", "*", "*.html")
	if _, err := os.Stat(filepath.Join(cwd, "templates")); os.IsNotExist(err) {
		templatePath = filepath.Join(cwd, "learningfolder", "templates", "*", "*.html")
	}

	var err error
	render.Tmpl, err = template.ParseGlob(templatePath)
	if err != nil {
		log.Fatalf("Error parsing templates: %v", err)
	}

	// Go 1.26: Use range over integers
	fmt.Print("Initializing routing system")
	for i := range 3 {
		fmt.Print(".")
		_ = i
	}
	fmt.Println(" Done.")

	// Create a new ServeMux (router)
	mux := http.NewServeMux()

	// Register routes using our custom routes package
	routes.RegisterRoutes(mux)

	log.Println("Starting webserver on :8080")
	log.Println("Go 1.26 Tip: Access new leak profile at /debug/pprof/goroutineleak")

	// Start the server with the mux
	log.Fatal(http.ListenAndServe(":8080", mux))
}
