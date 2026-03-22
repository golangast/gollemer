package sqlite_db

import (
	"database/sql"
	"fmt"
	"os"
	"path/filepath"

	_ "modernc.org/sqlite" // Pure Go SQLite driver
)

// InitDB initializes an SQLite database at the given path.
// It creates the database file if it doesn't exist and sets up a 'messages' table.
func InitDB(dataSourceName string) (*sql.DB, error) {
	// Extract the directory from the dataSourceName
	dir := filepath.Dir(dataSourceName)
	// Create the directory if it doesn't exist
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		err := os.MkdirAll(dir, 0755) // Use MkdirAll to create parent directories as well
		if err != nil {
			return nil, fmt.Errorf("failed to create directory %s: %w", dir, err)
		}
	}

	db, err := sql.Open("sqlite", dataSourceName)
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	// Create messages table
	createMessagesTableSQL := `CREATE TABLE IF NOT EXISTS messages (
		"id" INTEGER PRIMARY KEY AUTOINCREMENT,
		"role" TEXT NOT NULL,
		"content" TEXT NOT NULL,
		"timestamp" DATETIME DEFAULT CURRENT_TIMESTAMP,
		"commit_hash" TEXT
	);`

	_, err = db.Exec(createMessagesTableSQL)
	if err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to create messages table: %w", err)
	}

	createUsersTableSQL := `CREATE TABLE IF NOT EXISTS users (
		"id" INTEGER PRIMARY KEY AUTOINCREMENT,
		"name" TEXT NOT NULL,
		"age" INTEGER NOT NULL
	);`

	_, err = db.Exec(createUsersTableSQL)
	if err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to create users table: %w", err)
	}

	createTutorialTableSQL := `CREATE TABLE IF NOT EXISTS tutorial_metadata (
		id INTEGER PRIMARY KEY CHECK (id = 1),
		current_step INTEGER DEFAULT 0,
		is_active BOOLEAN DEFAULT FALSE,
		updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
	);`

	_, err = db.Exec(createTutorialTableSQL)
	if err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to create tutorial_metadata table: %w", err)
	}

	// Project Profile Tables
	createProjectProfilesTableSQL := `CREATE TABLE IF NOT EXISTS project_profiles (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		name TEXT UNIQUE NOT NULL,
		path TEXT NOT NULL,
		files_count INTEGER,
		routes_count INTEGER,
		db_count INTEGER,
		total_loc INTEGER,
		last_visited TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
		created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
		description TEXT
	);`
	_, err = db.Exec(createProjectProfilesTableSQL)
	if err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to create project_profiles table: %w", err)
	}

	createProjectRoutesTableSQL := `CREATE TABLE IF NOT EXISTS project_routes (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		project_id INTEGER,
		route TEXT,
		method TEXT,
		handler TEXT,
		FOREIGN KEY(project_id) REFERENCES project_profiles(id) ON DELETE CASCADE
	);`
	_, err = db.Exec(createProjectRoutesTableSQL)
	if err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to create project_routes table: %w", err)
	}

	createProjectDatabasesTableSQL := `CREATE TABLE IF NOT EXISTS project_databases (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		project_id INTEGER,
		db_path TEXT,
		db_type TEXT,
		FOREIGN KEY(project_id) REFERENCES project_profiles(id) ON DELETE CASCADE
	);`
	_, err = db.Exec(createProjectDatabasesTableSQL)
	if err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to create project_databases table: %w", err)
	}

	return db, nil
}

// SaveMessage saves a message to the 'messages' table and returns the new row's ID.
func SaveMessage(db *sql.DB, role, content string) (int64, error) {
	insertSQL := `INSERT INTO messages(role, content) VALUES (?, ?)`
	result, err := db.Exec(insertSQL, role, content)
	if err != nil {
		return 0, fmt.Errorf("failed to insert message: %w", err)
	}
	id, err := result.LastInsertId()
	if err != nil {
		return 0, fmt.Errorf("failed to get last insert ID: %w", err)
	}
	return id, nil
}

// UpdateCommitHash updates the commit_hash for a given message ID.
func UpdateCommitHash(db *sql.DB, id int64, hash string) error {
	updateSQL := `UPDATE messages SET commit_hash = ? WHERE id = ?`
	_, err := db.Exec(updateSQL, hash, id)
	if err != nil {
		return fmt.Errorf("failed to update commit hash: %w", err)
	}
	return nil
}

// GetCommitHash retrieves the commit_hash for a given message ID.
func GetCommitHash(db *sql.DB, id int64) (string, error) {
	query := `SELECT commit_hash FROM messages WHERE id = ?`
	var hash sql.NullString
	err := db.QueryRow(query, id).Scan(&hash)
	if err != nil {
		return "", fmt.Errorf("failed to get commit hash: %w", err)
	}
	if !hash.Valid {
		return "", fmt.Errorf("no commit hash found for id %d", id)
	}
	return hash.String, nil
}

// GetMessageByCommitHash retrieves a message by its commit hash (supports partial hash).
func GetMessageByCommitHash(db *sql.DB, hash string) (*Message, error) {
	// Support both full and partial hashes by using LIKE
	query := `SELECT id, role, content, timestamp, commit_hash FROM messages WHERE commit_hash LIKE ? ORDER BY timestamp DESC LIMIT 1`
	var msg Message
	err := db.QueryRow(query, hash+"%").Scan(&msg.ID, &msg.Role, &msg.Content, &msg.Timestamp, &msg.CommitHash)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("no message found with commit hash starting with '%s'", hash)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to query message by commit hash: %w", err)
	}
	return &msg, nil
}

// Message represents a message stored in the database.
type Message struct {
	ID         int
	Role       string
	Content    string
	Timestamp  string
	CommitHash sql.NullString
}

// GetMessages retrieves all messages from the 'messages' table.
func GetMessages(db *sql.DB) ([]Message, error) {
	query := `SELECT id, role, content, timestamp, commit_hash FROM messages ORDER BY timestamp ASC`
	rows, err := db.Query(query)
	if err != nil {
		return nil, fmt.Errorf("failed to query messages: %w", err)
	}
	defer rows.Close()

	var messages []Message
	for rows.Next() {
		var msg Message
		if err := rows.Scan(&msg.ID, &msg.Role, &msg.Content, &msg.Timestamp, &msg.CommitHash); err != nil {
			return nil, fmt.Errorf("failed to scan message: %w", err)
		}
		messages = append(messages, msg)
	}

	return messages, nil
}

// SyncStep updates the database with the current tutorial progress
func SyncStep(db *sql.DB, step int, isActive bool) error {
	query := `
		INSERT INTO tutorial_metadata (id, current_step, is_active, updated_at)
		VALUES (1, ?, ?, CURRENT_TIMESTAMP)
		ON CONFLICT(id) DO UPDATE SET 
			current_step = excluded.current_step,
			is_active = excluded.is_active,
			updated_at = CURRENT_TIMESTAMP`
	_, err := db.Exec(query, step, isActive)
	return err
}

// GetCurrentStep retrieves the user's progress
func GetCurrentStep(db *sql.DB) (int, bool) {
	var step int
	var active bool
	err := db.QueryRow("SELECT current_step, is_active FROM tutorial_metadata WHERE id = 1").Scan(&step, &active)
	if err != nil {
		return 0, false
	}
	return step, active
}

// ProjectProfile represents a project's metadata.
type ProjectProfile struct {
	ID          int
	Name        string
	Path        string
	FilesCount  int
	RoutesCount int
	DbCount     int
	TotalLOC    int
	LastVisited string
	CreatedAt   string
	Description string
}

// ProjectRoute represents a route in a project.
type ProjectRoute struct {
	ID        int
	ProjectID int
	Route     string
	Method    string
	Handler   string
}

// ProjectDatabase represents a database in a project.
type ProjectDatabase struct {
	ID        int
	ProjectID int
	DBPath    string
	DBType    string
}

// UpsertProjectProfile saves or updates a project profile.
func UpsertProjectProfile(db *sql.DB, profile ProjectProfile) (int64, error) {
	query := `
		INSERT INTO project_profiles (name, path, files_count, routes_count, db_count, total_loc, last_visited, description)
		VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
		ON CONFLICT(name) DO UPDATE SET
			path = excluded.path,
			files_count = excluded.files_count,
			routes_count = excluded.routes_count,
			db_count = excluded.db_count,
			total_loc = excluded.total_loc,
			last_visited = CURRENT_TIMESTAMP,
			description = excluded.description`
	
	_, err := db.Exec(query, profile.Name, profile.Path, profile.FilesCount, profile.RoutesCount, profile.DbCount, profile.TotalLOC, profile.Description)
	if err != nil {
		return 0, err
	}
	
	// Get ID
	var id int64
	err = db.QueryRow("SELECT id FROM project_profiles WHERE name = ?", profile.Name).Scan(&id)
	return id, err
}

// ClearProjectDetails removes routes and databases for a project to refresh them.
func ClearProjectDetails(db *sql.DB, projectID int64) error {
	_, err := db.Exec("DELETE FROM project_routes WHERE project_id = ?", projectID)
	if err != nil {
		return err
	}
	_, err = db.Exec("DELETE FROM project_databases WHERE project_id = ?", projectID)
	return err
}

// AddProjectRoute adds a route to a project.
func AddProjectRoute(db *sql.DB, projectID int64, route, method, handler string) error {
	_, err := db.Exec("INSERT INTO project_routes (project_id, route, method, handler) VALUES (?, ?, ?, ?)", projectID, route, method, handler)
	return err
}

// AddProjectDatabase adds a database info to a project.
func AddProjectDatabase(db *sql.DB, projectID int64, dbPath, dbType string) error {
	_, err := db.Exec("INSERT INTO project_databases (project_id, db_path, db_type) VALUES (?, ?, ?)", projectID, dbPath, dbType)
	return err
}

// GetProjectProfile retrieves a project profile by name.
func GetProjectProfile(db *sql.DB, name string) (*ProjectProfile, error) {
	query := `SELECT id, name, path, files_count, routes_count, db_count, total_loc, last_visited, created_at, description FROM project_profiles WHERE name = ?`
	var p ProjectProfile
	err := db.QueryRow(query, name).Scan(&p.ID, &p.Name, &p.Path, &p.FilesCount, &p.RoutesCount, &p.DbCount, &p.TotalLOC, &p.LastVisited, &p.CreatedAt, &p.Description)
	if err != nil {
		return nil, err
	}
	return &p, nil
}

// GetProjectRoutes retrieves all routes for a project.
func GetProjectRoutes(db *sql.DB, projectID int) ([]ProjectRoute, error) {
	rows, err := db.Query("SELECT id, project_id, route, method, handler FROM project_routes WHERE project_id = ?", projectID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var routes []ProjectRoute
	for rows.Next() {
		var r ProjectRoute
		if err := rows.Scan(&r.ID, &r.ProjectID, &r.Route, &r.Method, &r.Handler); err != nil {
			return nil, err
		}
		routes = append(routes, r)
	}
	return routes, nil
}

// GetProjectDatabases retrieves all databases for a project.
func GetProjectDatabases(db *sql.DB, projectID int) ([]ProjectDatabase, error) {
	rows, err := db.Query("SELECT id, project_id, db_path, db_type FROM project_databases WHERE project_id = ?", projectID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var databases []ProjectDatabase
	for rows.Next() {
		var d ProjectDatabase
		if err := rows.Scan(&d.ID, &d.ProjectID, &d.DBPath, &d.DBType); err != nil {
			return nil, err
		}
		databases = append(databases, d)
	}
	return databases, nil
}

