package sqlite_db

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

// JSONDatabase handles persistence for Gollemer state using a flat JSON file.
type JSONDatabase struct {
	Messages         []Message         `json:"messages"`
	Users            []User            `json:"users"`
	TutorialMetadata TutorialMetadata  `json:"tutorial_metadata"`
	ProjectProfiles  []ProjectProfile  `json:"project_profiles"`
	ProjectRoutes    []ProjectRoute    `json:"project_routes"`
	ProjectDatabases []ProjectDatabase `json:"project_databases"`

	path string
	mu   sync.RWMutex
}

type Message struct {
	ID         int    `json:"id"`
	Role       string `json:"role"`
	Content    string `json:"content"`
	Timestamp  string `json:"timestamp"`
	CommitHash string `json:"commit_hash,omitempty"`
}

type User struct {
	ID   int    `json:"id"`
	Name string `json:"name"`
	Age  int    `json:"age"`
}

type TutorialMetadata struct {
	CurrentStep int    `json:"current_step"`
	IsActive    bool   `json:"is_active"`
	UpdatedAt   string `json:"updated_at"`
}

type ProjectProfile struct {
	ID          int    `json:"id"`
	Name        string `json:"name"`
	Path        string `json:"path"`
	FilesCount  int    `json:"files_count"`
	RoutesCount int    `json:"routes_count"`
	DbCount     int    `json:"db_count"`
	TotalLOC    int    `json:"total_loc"`
	LastVisited string `json:"last_visited"`
	CreatedAt   string `json:"created_at"`
	Description string `json:"description"`
}

type ProjectRoute struct {
	ID        int    `json:"id"`
	ProjectID int    `json:"project_id"`
	Route     string `json:"route"`
	Method    string `json:"method"`
	Handler   string `json:"handler"`
}

type ProjectDatabase struct {
	ID        int    `json:"id"`
	ProjectID int    `json:"project_id"`
	DBPath    string `json:"db_path"`
	DBType    string `json:"db_type"`
}

// InitDB initializes a JSON database at the given path.
func InitDB(dataSourceName string) (*JSONDatabase, error) {
	// If it ends in .db, change to .json for clarity
	if strings.HasSuffix(dataSourceName, ".db") {
		dataSourceName = strings.TrimSuffix(dataSourceName, ".db") + ".json"
	}

	dir := filepath.Dir(dataSourceName)
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		err := os.MkdirAll(dir, 0755)
		if err != nil {
			return nil, fmt.Errorf("failed to create directory %s: %w", dir, err)
		}
	}

	db := &JSONDatabase{
		path: dataSourceName,
	}

	if _, err := os.Stat(dataSourceName); err == nil {
		data, err := os.ReadFile(dataSourceName)
		if err != nil {
			return nil, fmt.Errorf("failed to read database file: %w", err)
		}
		if err := json.Unmarshal(data, db); err != nil {
			return nil, fmt.Errorf("failed to parse database file: %w", err)
		}
	} else {
		// Initialize with default data
		db.TutorialMetadata = TutorialMetadata{CurrentStep: 0, IsActive: false, UpdatedAt: time.Now().Format(time.RFC3339)}
		if err := db.save(); err != nil {
			return nil, err
		}
	}

	return db, nil
}

func (db *JSONDatabase) save() error {
	data, err := json.MarshalIndent(db, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal database: %w", err)
	}
	return os.WriteFile(db.path, data, 0644)
}

func (db *JSONDatabase) Close() error {
	return db.save()
}

// SaveMessage saves a message and returns the new row's ID.
func SaveMessage(db *JSONDatabase, role, content string) (int64, error) {
	db.mu.Lock()
	defer db.mu.Unlock()

	id := 1
	if len(db.Messages) > 0 {
		id = db.Messages[len(db.Messages)-1].ID + 1
	}

	msg := Message{
		ID:        id,
		Role:      role,
		Content:   content,
		Timestamp: time.Now().Format(time.RFC3339),
	}
	db.Messages = append(db.Messages, msg)
	return int64(id), db.save()
}

// UpdateCommitHash updates the commit_hash for a given message ID.
func UpdateCommitHash(db *JSONDatabase, id int64, hash string) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	for i := range db.Messages {
		if int64(db.Messages[i].ID) == id {
			db.Messages[i].CommitHash = hash
			return db.save()
		}
	}
	return fmt.Errorf("message with id %d not found", id)
}

// GetCommitHash retrieves the commit_hash for a given message ID.
func GetCommitHash(db *JSONDatabase, id int64) (string, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	for _, msg := range db.Messages {
		if int64(msg.ID) == id {
			if msg.CommitHash == "" {
				return "", fmt.Errorf("no commit hash found for id %d", id)
			}
			return msg.CommitHash, nil
		}
	}
	return "", fmt.Errorf("message with id %d not found", id)
}

// GetMessageByCommitHash retrieves a message by its commit hash (supports partial hash).
func GetMessageByCommitHash(db *JSONDatabase, hash string) (*Message, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	// Sort by timestamp DESC to get the latest
	sort.Slice(db.Messages, func(i, j int) bool {
		return db.Messages[i].Timestamp > db.Messages[j].Timestamp
	})

	for _, msg := range db.Messages {
		if strings.HasPrefix(msg.CommitHash, hash) {
			return &msg, nil
		}
	}
	return nil, fmt.Errorf("no message found with commit hash starting with '%s'", hash)
}

// GetMessages retrieves all messages.
func GetMessages(db *JSONDatabase) ([]Message, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	// Sort by timestamp ASC
	sort.Slice(db.Messages, func(i, j int) bool {
		return db.Messages[i].Timestamp < db.Messages[j].Timestamp
	})

	return db.Messages, nil
}

// SyncStep updates the database with the current tutorial progress
func SyncStep(db *JSONDatabase, step int, isActive bool) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	db.TutorialMetadata.CurrentStep = step
	db.TutorialMetadata.IsActive = isActive
	db.TutorialMetadata.UpdatedAt = time.Now().Format(time.RFC3339)
	return db.save()
}

// GetCurrentStep retrieves the user's progress
func GetCurrentStep(db *JSONDatabase) (int, bool) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	return db.TutorialMetadata.CurrentStep, db.TutorialMetadata.IsActive
}

// UpsertProjectProfile saves or updates a project profile.
func UpsertProjectProfile(db *JSONDatabase, profile ProjectProfile) (int64, error) {
	db.mu.Lock()
	defer db.mu.Unlock()

	now := time.Now().Format(time.RFC3339)
	for i := range db.ProjectProfiles {
		if db.ProjectProfiles[i].Name == profile.Name {
			db.ProjectProfiles[i].Path = profile.Path
			db.ProjectProfiles[i].FilesCount = profile.FilesCount
			db.ProjectProfiles[i].RoutesCount = profile.RoutesCount
			db.ProjectProfiles[i].DbCount = profile.DbCount
			db.ProjectProfiles[i].TotalLOC = profile.TotalLOC
			db.ProjectProfiles[i].LastVisited = now
			db.ProjectProfiles[i].Description = profile.Description
			return int64(db.ProjectProfiles[i].ID), db.save()
		}
	}

	id := 1
	if len(db.ProjectProfiles) > 0 {
		maxID := 0
		for _, p := range db.ProjectProfiles {
			if p.ID > maxID {
				maxID = p.ID
			}
		}
		id = maxID + 1
	}

	profile.ID = id
	profile.LastVisited = now
	profile.CreatedAt = now
	db.ProjectProfiles = append(db.ProjectProfiles, profile)
	return int64(id), db.save()
}

// ClearProjectDetails removes routes and databases for a project to refresh them.
func ClearProjectDetails(db *JSONDatabase, projectID int64) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	newRoutes := make([]ProjectRoute, 0)
	for _, r := range db.ProjectRoutes {
		if int64(r.ProjectID) != projectID {
			newRoutes = append(newRoutes, r)
		}
	}
	db.ProjectRoutes = newRoutes

	newDBs := make([]ProjectDatabase, 0)
	for _, d := range db.ProjectDatabases {
		if int64(d.ProjectID) != projectID {
			newDBs = append(newDBs, d)
		}
	}
	db.ProjectDatabases = newDBs

	return db.save()
}

// AddProjectRoute adds a route to a project.
func AddProjectRoute(db *JSONDatabase, projectID int64, route, method, handler string) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	id := 1
	if len(db.ProjectRoutes) > 0 {
		maxID := 0
		for _, r := range db.ProjectRoutes {
			if r.ID > maxID {
				maxID = r.ID
			}
		}
		id = maxID + 1
	}

	db.ProjectRoutes = append(db.ProjectRoutes, ProjectRoute{
		ID:        id,
		ProjectID: int(projectID),
		Route:     route,
		Method:    method,
		Handler:   handler,
	})
	return db.save()
}

// AddProjectDatabase adds a database info to a project.
func AddProjectDatabase(db *JSONDatabase, projectID int64, dbPath, dbType string) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	id := 1
	if len(db.ProjectDatabases) > 0 {
		maxID := 0
		for _, d := range db.ProjectDatabases {
			if d.ID > maxID {
				maxID = d.ID
			}
		}
		id = maxID + 1
	}

	db.ProjectDatabases = append(db.ProjectDatabases, ProjectDatabase{
		ID:        id,
		ProjectID: int(projectID),
		DBPath:    dbPath,
		DBType:    dbType,
	})
	return db.save()
}

// GetProjectProfile retrieves a project profile by name.
func GetProjectProfile(db *JSONDatabase, name string) (*ProjectProfile, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	for _, p := range db.ProjectProfiles {
		if p.Name == name {
			return &p, nil
		}
	}
	return nil, fmt.Errorf("project profile '%s' not found", name)
}

// GetProjectRoutes retrieves all routes for a project.
func GetProjectRoutes(db *JSONDatabase, projectID int) ([]ProjectRoute, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	routes := make([]ProjectRoute, 0)
	for _, r := range db.ProjectRoutes {
		if r.ProjectID == projectID {
			routes = append(routes, r)
		}
	}
	return routes, nil
}

// GetProjectDatabases retrieves all databases for a project.
func GetProjectDatabases(db *JSONDatabase, projectID int) ([]ProjectDatabase, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	databases := make([]ProjectDatabase, 0)
	for _, d := range db.ProjectDatabases {
		if d.ProjectID == projectID {
			databases = append(databases, d)
		}
	}
	return databases, nil
}
