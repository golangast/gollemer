package main

import (
	"bufio"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"

	"github.com/golangast/gollemer/internal/ai/moe/model"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
	"github.com/golangast/gollemer/internal/ai/neural/tokenizer"
	_ "modernc.org/sqlite" // Pure Go SQLite driver
)

// Plugin defines the interface that all service plugins must implement.
type Plugin interface {
	Name() string
	Init(svc *Service) error
}

// User represents a database user record.
type User struct {
	ID    int    `json:"id"`
	Name  string `json:"name"`
	Email string `json:"email"`
}

var pluginsRegistry = make(map[string]Plugin)

// RegisterPlugin should be called by plugins in their init() function.
func RegisterPlugin(p Plugin) {
	pluginsRegistry[p.Name()] = p
}

// Service represents our main application service.
type Service struct {
	Port      string
	DBURL     string
	PluginDir string
	Debug     bool
	DB        *sql.DB
	Model     *model.MoEModel
}

// LoadEnv is the first function you should call. It parses the .env file
// and sets environment variables for the service to consume.
func LoadEnv(path string) error {
	file, err := os.Open(path)
	if err != nil {
		// If the file doesn't exist, we just skip it (assume env vars are set elsewhere)
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("could not open .env file: %w", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			continue
		}

		key := strings.TrimSpace(parts[0])
		val := strings.TrimSpace(parts[1])
		os.Setenv(key, val)
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("error reading .env file: %w", err)
	}

	log.Println("✅ Successfully loaded environment from .env")
	return nil
}

// InitService initializes the service configuration from environment variables.
func InitService() *Service {
	return &Service{
		Port:      os.Getenv("PORT"),
		DBURL:     os.Getenv("DB_URL"),
		PluginDir: os.Getenv("PLUGIN_DIR"),
		Debug:     os.Getenv("DEBUG") == "true",
	}
}

// LoadPlugins scans the plugins directory and initializes already registered plugins.
func (s *Service) LoadPlugins() error {
	log.Printf("🔌 Initializing %d registered plugins...", len(pluginsRegistry))
	for name, plugin := range pluginsRegistry {
		if err := plugin.Init(s); err != nil {
			log.Printf("❌ Failed to init plugin %s: %v", name, err)
		} else {
			log.Printf("✅ Plugin %s initialized", name)
		}
	}
	return nil
}

// StartHTTP sets up the routes and starts the web server.
func (s *Service) StartHTTP() error {
	if s.Port == "" {
		s.Port = "8080"
	}

	mux := http.NewServeMux()

	// 🏠 Root & Health
	mux.HandleFunc("/", s.HomeHandler)
	mux.HandleFunc("/health", s.HealthHandler)

	// 👤 User CRUD
	mux.HandleFunc("/users", s.UsersHandler) // GET all, POST new
	mux.HandleFunc("/users/delete", s.DeleteUserHandler) // Simpler delete

	// 🤖 AI Chat
	mux.HandleFunc("/chat", s.ChatHandler)

	log.Printf("🌍 Serving at http://localhost:%s", s.Port)
	return http.ListenAndServe(":"+s.Port, mux)
}

// ChatHandler processes AI requests.
func (s *Service) ChatHandler(w http.ResponseWriter, r *http.Request) {
	prompt := r.URL.Query().Get("q")
	if prompt == "" {
		http.Error(w, "Query parameter 'q' is required", http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "text/plain")
	// For web, we might not want live-streaming in this basic example, 
	// but we'll use the model to generate.
	s.Model.GenerateWithStats(prompt)
	fmt.Fprintf(w, "\nResponse generated in logs.")
}

// InitAI initializes the next-gen MoE model.
func (s *Service) InitAI() error {
	v := vocab.NewVocabulary()
	// Add some dummy tokens for demonstration
	v.AddToken("Hello")
	v.AddToken("world")
	v.AddToken("Gollemer")
	v.AddToken("is")
	v.AddToken("thinking")
	v.AddToken("...")

	t, _ := tokenizer.NewTokenizer(v)
	s.Model = model.NewMoEModel(t, model.MoEConfig{
		MaxLen:     50,
		HiddenSize: 128,
		Layers:     4,
	})
	log.Println("🧠 AI Model (MoE) Initialized with GQA/RoPE support")
	return nil
}

// RunInteractiveLoop starts the CLI chat.
func (s *Service) RunInteractiveLoop() {
	scanner := bufio.NewScanner(os.Stdin)
	fmt.Println("\n--- Gollemer Interactive AI Shell ---")
	fmt.Println("Type 'exit' to quit, 'clear' to reset memory.")

	for {
		fmt.Print("\nYOU > ")
		if !scanner.Scan() {
			break
		}
		text := strings.TrimSpace(scanner.Text())

		switch text {
		case "exit":
			return
		case "clear":
			s.Model.Cache.Reset()
			fmt.Println("ʕ◡ϖ◡ʔ > Memory cleared. What's on your mind?")
			continue
		case "":
			continue
		}

		s.Model.GenerateWithStats(text)
		fmt.Println()
	}
}

// HomeHandler serves the landing page.
func (s *Service) HomeHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "Welcome to the Gollemer Service! ✨\nCheck out /users or /health")
}

// HealthHandler returns service status.
func (s *Service) HealthHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":  "ok",
		"debug":   s.Debug,
		"plugins": len(pluginsRegistry),
	})
}

// UsersHandler manages user listing and creation.
func (s *Service) UsersHandler(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		s.ListUsers(w, r)
	case http.MethodPost:
		s.CreateUser(w, r)
	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

// ListUsers queries all users from SQLite.
func (s *Service) ListUsers(w http.ResponseWriter, r *http.Request) {
	rows, err := s.DB.Query("SELECT id, name, email FROM users")
	if err != nil {
		// Table might not exist yet
		s.DB.Exec("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, email TEXT)")
		rows, err = s.DB.Query("SELECT id, name, email FROM users")
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
	}
	defer rows.Close()

	users := []User{}
	for rows.Next() {
		var u User
		if err := rows.Scan(&u.ID, &u.Name, &u.Email); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		users = append(users, u)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(users)
}

// CreateUser adds a new user to the database.
func (s *Service) CreateUser(w http.ResponseWriter, r *http.Request) {
	var u User
	if err := json.NewDecoder(r.Body).Decode(&u); err != nil {
		http.Error(w, "Invalid input", http.StatusBadRequest)
		return
	}

	// Ensure table exists
	s.DB.Exec("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, email TEXT)")

	result, err := s.DB.Exec("INSERT INTO users (name, email) VALUES (?, ?)", u.Name, u.Email)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	id, _ := result.LastInsertId()
	u.ID = int(id)

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(u)
}

// DeleteUserHandler removes a user by name (for demonstration).
func (s *Service) DeleteUserHandler(w http.ResponseWriter, r *http.Request) {
	name := r.URL.Query().Get("name")
	if name == "" {
		http.Error(w, "Query parameter 'name' is required", http.StatusBadRequest)
		return
	}

	_, err := s.DB.Exec("DELETE FROM users WHERE name = ?", name)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	fmt.Fprintf(w, "User %s deleted successfully", name)
}

// ConnectDB initializes the database connection.
func (s *Service) ConnectDB() error {
	if s.DBURL == "" {
		return fmt.Errorf("DB_URL is not set")
	}

	// For modernc.org/sqlite, we strip the 'sqlite://' prefix if present
	dsn := strings.TrimPrefix(s.DBURL, "sqlite://")
	
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return fmt.Errorf("could not open database: %w", err)
	}

	if err := db.Ping(); err != nil {
		return fmt.Errorf("could not ping database: %w", err)
	}

	s.DB = db
	log.Println("🗄️  Connected to SQLite database successfully")
	return nil
}

func main() {
	// First Step: Load Environment
	if err := LoadEnv(".env"); err != nil {
		log.Fatalf("❌ Error loading .env: %v", err)
	}

	// Second Step: Initialize Service Struct
	svc := InitService()
	log.Printf("🚀 Initializing service on port %s (Debug: %v)", svc.Port, svc.Debug)

	// Third Step: Connect Database
	if err := svc.ConnectDB(); err != nil {
		log.Printf("⚠️  Warning: Error connecting to DB: %v", err)
	}

	// Fourth Step: Load Plugins
	if err := svc.LoadPlugins(); err != nil {
		log.Printf("⚠️  Warning: Error loading plugins: %v", err)
	}

	// Fifth Step: Initialize AI
	if err := svc.InitAI(); err != nil {
		log.Printf("⚠️  Error initializing AI: %v", err)
	}

	// Sixth Step: Start the CLI loop in a separate goroutine if needed
	// But since we want to interact, we can run it here and the server in a goroutine
	go func() {
		if err := svc.StartHTTP(); err != nil {
			log.Fatalf("❌ Server failed to start: %v", err)
		}
	}()

	svc.RunInteractiveLoop()
}
