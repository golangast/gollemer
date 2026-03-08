package fileio

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// goImports formats the Go file at the given path.
func GoImports(filePath string) {
	cmd := exec.Command("goimports", "-w", filePath)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	_ = cmd.Run()
}

// registerHandlerURL adds the handler to the main.go file.
func RegisterHandlerURL(handlerName, url, mainPath string) (string, error) {
	content, err := os.ReadFile(mainPath)
	if err != nil {
		return "", err
	}

	lines := strings.SplitSeq(string(content), "\n")
	var newLines []string
	inserted := false

	for _, line := range lines {
		newLines = append(newLines, line)
		if !inserted && strings.Contains(line, "func main() {") {
			newLines = append(newLines, fmt.Sprintf(`	http.HandleFunc("%s", %s)`,
				url, strings.Title(handlerName)))
			inserted = true
		}
	}

	if !inserted {
		return "", fmt.Errorf("could not find func main() in main.go")
	}

	err = os.WriteFile(mainPath, []byte(strings.Join(newLines, "\n")), 0644)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("I've registered the handler at %s in main.go", url), nil
}
