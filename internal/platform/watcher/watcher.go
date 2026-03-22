package watcher

import (
	"os"
	"path/filepath"
	"strings"
	"time"
)

type FileState struct {
	Size    int64
	ModTime time.Time
}

type Workspace struct {
	LastState map[string]FileState
}

func NewWorkspace() *Workspace {
	return &Workspace{LastState: make(map[string]FileState)}
}

// Scan checks for created or modified files and returns a map of changes
func (w *Workspace) Scan(root string) map[string]string {
	changes := make(map[string]string)
	newState := make(map[string]FileState)

	filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() || isHidden(path) {
			return nil
		}

		curr := FileState{Size: info.Size(), ModTime: info.ModTime()}
		newState[path] = curr

		if old, exists := w.LastState[path]; exists {
			if curr.Size != old.Size || !curr.ModTime.Equal(old.ModTime) {
				changes[path] = "modified"
			}
		} else if len(w.LastState) > 0 {
			changes[path] = "created"
		}
		return nil
	})

	w.LastState = newState
	return changes
}

func isHidden(path string) bool {
	base := filepath.Base(path)
	return strings.HasPrefix(base, ".") || strings.Contains(path, "vendor") || strings.Contains(path, "bin")
}
