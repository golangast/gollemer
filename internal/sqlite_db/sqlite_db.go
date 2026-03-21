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
