package llm

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/golangast/gollemer/internal/platform/sqlite_db"
	"github.com/golangast/gollemer/internal/platform/ui"
	"github.com/golangast/gollemer/internal/util/colors"
)

func printIntro() {
	fmt.Println("--- Welcome to Gollemer! ʕ◔ϖ◔ʔ ---")
	fmt.Println("It looks like this is your first time running Gollemer.")
	fmt.Println("\n💡 TIP: Type 'tutorial' to start an interactive guide!")
	fmt.Println("💡 TIP: Type 'menu' for an easy-to-use options menu!")
	fmt.Println("Here is a quick guide to get you started:")
	fmt.Println("")
	fmt.Println("1. Commands:")
	fmt.Println("   You can use natural language to interact with your project.")
	fmt.Println("   - Navigation: 'go to cmd', 'list files', 'tree'")
	fmt.Println("   - File Ops:   'create file main.go', 'delete folder tmp'")
	fmt.Println("   - Web Dev:    'create webserver MyApp', 'create handler Login', 'run webserver'")
	fmt.Println("   - System:     'clear', 'exit', 'history'")
	fmt.Println("")
	fmt.Println("2. The Learning System (How & Why):")
	fmt.Println("   Gollemer learns from your code to automate repetitive tasks.")
	fmt.Println("   - HOW: It scans a 'learningfolder' for templates (files like 'navbar.html', 'auth.go').")
	fmt.Println("          If it finds 'navbar.html', it learns the 'navbar' object.")
	fmt.Println("   - WHY: So you can say 'create navbar' and it generates the code for you instantly,")
	fmt.Println("          using your own preferred style and structure.")
	fmt.Println("")
	fmt.Println("3. Customizing Learning:")
	fmt.Println("   You have full control over what Gollemer learns.")
	fmt.Println("   - Add/Edit files in the 'learningfolder' to teach it new objects.")
	fmt.Println("   - Change the source folder: 'learn from ./my-templates'")
	fmt.Println("   - Teach specific words: 'learn object widget'")
	fmt.Println("")
	fmt.Println("Type 'help' at any time to see this information again.")
	fmt.Println("----------------------------------------")
	fmt.Println("")
}

// ScanAndSaveProfile scans the project directory and updates the database profile.
func ScanAndSaveProfile(projectName string, projectPath string, db *sqlite_db.JSONDatabase) error {
	if projectName == "" || projectPath == "" {
		return fmt.Errorf("invalid project name or path")
	}

	profile := sqlite_db.ProjectProfile{
		Name: projectName,
		Path: projectPath,
	}

	var routes []sqlite_db.ProjectRoute
	var databases []string

	err := filepath.WalkDir(projectPath, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			// Skip hidden dirs and common non-source dirs
			name := d.Name()
			if strings.HasPrefix(name, ".") || name == "vendor" || name == "node_modules" || name == "bin" {
				return filepath.SkipDir
			}
			return nil
		}

		profile.FilesCount++
		ext := filepath.Ext(path)

		// Database check
		if ext == ".db" || ext == ".sqlite" || ext == ".sqlite3" {
			databases = append(databases, path)
		}

		// LOC and Routes check for .go files
		if ext == ".go" {
			content, err := os.ReadFile(path)
			if err == nil {
				lines := strings.Split(string(content), "\n")
				profile.TotalLOC += len(lines)

				// Basic route detection
				for _, line := range lines {
					if strings.Contains(line, "http.HandleFunc(") || strings.Contains(line, ".HandleFunc(") {
						// Try to extract route
						re := regexp.MustCompile(`HandleFunc\("([^"]+)"`)
						matches := re.FindStringSubmatch(line)
						if len(matches) > 1 {
							routes = append(routes, sqlite_db.ProjectRoute{
								Route: matches[1],
							})
						} else {
							// Try alternate format HandleFunc(pattern, ...)
							re2 := regexp.MustCompile(`HandleFunc\(\s*"([^"]+)"`)
							matches2 := re2.FindStringSubmatch(line)
							if len(matches2) > 1 {
								routes = append(routes, sqlite_db.ProjectRoute{
									Route: matches2[1],
								})
							}
						}
					}
				}
			}
		}
		return nil
	})

	if err != nil {
		return err
	}

	profile.RoutesCount = len(routes)
	profile.DbCount = len(databases)

	// Save to DB
	pID, err := sqlite_db.UpsertProjectProfile(db, profile)
	if err != nil {
		return err
	}

	// Update details
	sqlite_db.ClearProjectDetails(db, pID)
	for _, r := range routes {
		sqlite_db.AddProjectRoute(db, pID, r.Route, "GET/POST", "Handler") // Simplified
	}
	for _, dPath := range databases {
		sqlite_db.AddProjectDatabase(db, pID, dPath, "SQLite")
	}

	return nil
}

// ShowProjectProfile prints a personable and visually rich summary of the project.
func ShowProjectProfile(projectName string, db *sqlite_db.JSONDatabase, mascot *ui.Mascot) {
	p, err := sqlite_db.GetProjectProfile(db, projectName)
	if err != nil {
		mascot.Speak(ui.MoodWaiting, fmt.Sprintf("I tried to peek into the heart of '%s', but I couldn't find its profile. Is it a secret project?", projectName))
		return
	}

	fmt.Println()
	colors.ColorizeCol("cyan", "black", "  ┌────────────────────────────────────────────────────────────┐")
	colors.ColorizeCol("cyan", "black", fmt.Sprintf("  │                 %s PROJECT PROFILE: %-22s │", "📄", strings.ToUpper(p.Name)))
	colors.ColorizeCol("cyan", "black", "  ├────────────────────────────────────────────────────────────┤")

	// Scale Bar
	locStr := fmt.Sprintf("%d LOC", p.TotalLOC)
	barLen := 20
	filled := (p.TotalLOC / 500) // 1 hash per 500 lines
	if filled > barLen {
		filled = barLen
	}
	bar := "[" + strings.Repeat("#", filled) + strings.Repeat("-", barLen-filled) + "]"

	colors.ColorizeCol("white", "black", fmt.Sprintf("  │ Scale:    %-22s %-25s │", bar, locStr))
	colors.ColorizeCol("white", "black", fmt.Sprintf("  │ Files:    %-49d │", p.FilesCount))

	routeStr := "No routes detected"
	if p.RoutesCount > 0 {
		routeStr = fmt.Sprintf("%d active routes", p.RoutesCount)
	}
	colors.ColorizeCol("white", "black", fmt.Sprintf("  │ Network:  %-49s │", routeStr))

	dbStr := "No databases found"
	if p.DbCount > 0 {
		dbStr = fmt.Sprintf("%d connected (SQLite)", p.DbCount)
	}
	colors.ColorizeCol("white", "black", fmt.Sprintf("  │ Storage:  %-49s │", dbStr))

	colors.ColorizeCol("cyan", "black", "  ├────────────────────────────────────────────────────────────┤")
	colors.ColorizeCol("white", "black", fmt.Sprintf("  │ Path: %-52s │", p.Path))
	colors.ColorizeCol("cyan", "black", "  └────────────────────────────────────────────────────────────┘")

	// Gopher's Analysis & Suggestions
	analysis := getMascotAnalysis(p)
	if len(analysis) > 0 {
		mascot.Speak(ui.MoodHappy, "I've been analyzing our structure. Here are some thoughts on how we can level up:")
		for _, tip := range analysis {
			colors.ColorizeOutPut("byellow", "black", "  "+tip)
			time.Sleep(100 * time.Millisecond)
		}
	} else {
		mascot.Speak(ui.MoodHappy, "Everything looks perfectly balanced! You've got a very clean foundation here.")
	}
	fmt.Println()
}

// getMascotAnalysis generates intuitive improvement suggestions based on the project profile.
func getMascotAnalysis(p *sqlite_db.ProjectProfile) []string {
	var advice []string

	// 1. Architecture Complexity
	if p.TotalLOC > 1500 && p.FilesCount < 4 {
		advice = append(advice, "💡 Architecture: Our main package is growing fast! Consider splitting logic into sub-folders like '/pkg' or '/internal'.")
	}

	// 2. Network vs Persistence
	if p.RoutesCount > 5 && p.DbCount == 0 {
		advice = append(advice, "💡 Persistence: We have quite a few routes but no database. Want to add SQLite for permanent storage?")
	}

	// 3. Frontend / WASM
	// Check if this looks like a webserver but might need a frontend
	if p.RoutesCount > 0 && !strings.Contains(strings.ToLower(p.Path), "wasm") {
		advice = append(advice, "💡 Frontend: I see a backend forming! We could add a Go-WebAssembly frontend to make it interactive.")
	}

	// 5. General Scale
	if p.TotalLOC < 100 {
		advice = append(advice, "💡 Next Step: We're just starting. How about we scaffold a new API handler or a middleware?")
	}

	return advice
}
