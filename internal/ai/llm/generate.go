package llm

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
)

func generateDataStructurePackageContent(structName, packageName, dirName string, fields map[string]string) string {
	lowercaseName := strings.ToLower(structName)

	// Struct Definition
	var structDef strings.Builder
	structDef.WriteString(fmt.Sprintf("type %s struct {\n", structName))
	structDef.WriteString("\tID int `json:\"id\"`\n")

	sortedFieldNames := make([]string, 0, len(fields))
	for k := range fields {
		sortedFieldNames = append(sortedFieldNames, k)
	}
	sort.Strings(sortedFieldNames)

	for _, fieldName := range sortedFieldNames {
		structDef.WriteString(fmt.Sprintf("\t%s %s `json:\"%s\"`\n", strings.Title(fieldName), fields[fieldName], fieldName))
	}
	structDef.WriteString("}\n\n")

	// Show Handler construction
	selectColumns := []string{"id"}
	scanFields := []string{"&u.ID"}
	for _, fieldName := range sortedFieldNames {
		selectColumns = append(selectColumns, strings.ToLower(fieldName))
		scanFields = append(scanFields, "&u."+strings.Title(fieldName))
	}

	showHandlerContent := fmt.Sprintf(`
func Show%%sHandler(w http.ResponseWriter, r *http.Request) {
	cwd, _ := os.Getwd()
	dbPath := filepath.Join(cwd, "%%s", "%%s.db")
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer db.Close()

	rows, err := db.Query("SELECT %%s FROM %%s")
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer rows.Close()

	results := make([]%%s, 0)
	for rows.Next() {
		var u %%s
		if err := rows.Scan(%%s); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		results = append(results, u)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(results)
}`, structName, dirName, lowercaseName, strings.Join(selectColumns, ", "), lowercaseName, structName, structName, strings.Join(scanFields, ", "))

	var structFields []string
	var structFieldExecs []string
	for _, fieldName := range sortedFieldNames {
		structFields = append(structFields, fmt.Sprintf("%%s = ?", strings.ToLower(fieldName)))
		structFieldExecs = append(structFieldExecs, "u."+strings.Title(fieldName))
	}

	updateHandlerContent := fmt.Sprintf(`
func Update%%sHandler(w http.ResponseWriter, r *http.Request) {
	parts := strings.Split(r.URL.Path, "/")
	if len(parts) < 4 { // e.g. /update/user/123
		http.Error(w, "Invalid URL, expecting /update/%%s/{id}", http.StatusBadRequest)
		return
	}
	id := parts[len(parts)-1]

	var u %%s
	err := json.NewDecoder(r.Body).Decode(&u)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	cwd, _ := os.Getwd()
	dbPath := filepath.Join(cwd, "%%s", "%%s.db")
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer db.Close()

	stmt, err := db.Prepare("UPDATE %%s SET %%s WHERE id = ?")
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	_, err = stmt.Exec(%%s, id)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	fmt.Fprintf(w, "%%s with ID %%s updated successfully", id)
}`, structName, lowercaseName, structName, dirName, lowercaseName, lowercaseName, strings.Join(structFields, ", "), strings.Join(structFieldExecs, ", "), structName)

	deleteHandlerContent := fmt.Sprintf(`
func Delete%%sHandler(w http.ResponseWriter, r *http.Request) {
	parts := strings.Split(r.URL.Path, "/")
	if len(parts) < 4 { // e.g. /delete/user/123
		http.Error(w, "Invalid URL, expecting /delete/%%s/{id}", http.StatusBadRequest)
		return
	}
	id := parts[len(parts)-1]

	cwd, _ := os.Getwd()
	dbPath := filepath.Join(cwd, "%%s", "%%s.db")
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer db.Close()

	stmt, err := db.Prepare("DELETE FROM %%s WHERE id = ?")
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	_, err = stmt.Exec(id)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	fmt.Fprintf(w, "%%s with ID %%s deleted successfully", id)
}`, structName, lowercaseName, dirName, lowercaseName, lowercaseName, structName)

	packageFileContent := fmt.Sprintf(`package %%s
import (
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	_ "modernc.org/sqlite"
)

%%s
%%s
%%s
%%s
`, packageName, structDef.String(), showHandlerContent, updateHandlerContent, deleteHandlerContent)

	return packageFileContent
}

func generateDirectoryTree(root, indent string, depth, maxDepth int, highlightPath string) (string, error) {
	if depth > maxDepth {
		return "", nil
	}
	var sb strings.Builder
	files, err := os.ReadDir(root)
	if err != nil {
		return "", err
	}
	for i, file := range files {
		prefix := "├── "
		if i == len(files)-1 {
			prefix = "└── "
		}
		name := file.Name()
		fullPath := filepath.Join(root, name)

		hPrefix := ""
		hSuffix := ""
		if highlightPath != "" && (fullPath == highlightPath || strings.HasPrefix(highlightPath, fullPath)) {
			hPrefix = ">> "
			hSuffix = " <<"
		}
		sb.WriteString(indent + hPrefix + prefix + name + hSuffix + "\n")
		if file.IsDir() && depth < maxDepth {
			newIndent := indent + "│   "
			if i == len(files)-1 {
				newIndent = indent + "    "
			}
			subTree, err := generateDirectoryTree(fullPath, newIndent, depth+1, maxDepth, highlightPath)
			if err == nil {
				sb.WriteString(subTree)
			}
		}
	}
	return sb.String(), nil
}

func generateWordVizHTML(words []string, vectors [][]float64) string {
	wordsJSON, _ := json.Marshal(words)
	vectorsJSON, _ := json.Marshal(vectors)

	return fmt.Sprintf(`<!DOCTYPE html>
<html>
<head>
    <title>Word2Vec 2D Visualization</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/pca-js@1.0.0/pca.min.js"></script>
    <style>body { margin: 0; font-family: sans-serif; }</style>
</head>
<body>
    <div id="myDiv" style="width: 100%%; height: 100vh;"></div>
    <script>
        var words = %%s;
        var rawVectors = %%s;

        var x = [];
        var y = [];

        if (rawVectors.length > 0 && rawVectors[0].length > 2) {
            var vectors = PCA.getEigenVectors(rawVectors);
            var adData = PCA.computeAdjustedData(rawVectors, vectors[0], vectors[1]);
            x = adData.formattedAdjustedData[0];
            y = adData.formattedAdjustedData[1];
        } else {
             for (var i = 0; i < rawVectors.length; i++) {
                 x.push(rawVectors[i][0]);
                 y.push(rawVectors[i][1]);
             }
        }

        var trace = {
            x: x, y: y, mode: 'markers+text', type: 'scatter',
            text: words, textposition: 'top center', marker: { size: 8 }
        };
        var layout = { title: 'Word Vector Distribution (PCA Reduced)', hovermode: 'closest' };
        Plotly.newPlot('myDiv', [trace], layout);
    </script>
</body>
</html>`, string(wordsJSON), string(vectorsJSON))
}

func handleGenericCreate(objectType, fileName, targetDirectory, handlerURL string, kb *KnowledgeBase) (string, bool) {
	// 1. If it looks like a handler (has a URL)
	if handlerURL != "" {
		handlerName := fileName
		if handlerName == "" {
			handlerName = objectType
		}

		handlerContent := fmt.Sprintf("package main\n\nimport \"net/http\"\n\n// %%s is a generic handler for %%s\nfunc %%s(w http.ResponseWriter, r *http.Request) {\n\tw.Write([]byte(\"Generic implementation for %%s\"))\n}\n", strings.Title(handlerName), objectType, strings.Title(handlerName), objectType)
		filePath := handlerName + ".go"
		if targetDirectory != "" {
			filePath = filepath.Join(targetDirectory, filePath)
			os.MkdirAll(targetDirectory, 0755)
		}
		os.WriteFile(filePath, []byte(handlerContent), 0644)
		goImports(filePath)

		mainPath := "main.go"
		if targetDirectory != "" {
			if _, err := os.Stat(filepath.Join(targetDirectory, "main.go")); err == nil {
				mainPath = filepath.Join(targetDirectory, "main.go")
			}
		}
		regMsg, _ := registerHandlerURL(strings.Title(handlerName), handlerURL, mainPath)
		return fmt.Sprintf("I don't have a specialized handler for '%%s', but since you provided a URL, I've generated a backend handler for it. %%s", objectType, regMsg), true
	}

	// 2. If a name is provided, create a Go skeleton
	if fileName != "" {
		content := fmt.Sprintf("// %%s implementation for the %%s object\npackage main\n\nimport \"fmt\"\n\nfunc main() {\n\tfmt.Println(\"Executing %%s (%%s logic)\")\n}\n", strings.Title(fileName), objectType, fileName, objectType)
		filePath := fileName + ".go"
		if targetDirectory != "" {
			filePath = filepath.Join(targetDirectory, filePath)
			os.MkdirAll(targetDirectory, 0755)
		}
		os.WriteFile(filePath, []byte(content), 0644)
		goImports(filePath)
		return fmt.Sprintf("I identified '%%s' as a new object type. I've created a basic Go skeleton '%%s.go' for you.", objectType, fileName), true
	}

	// 3. Just a folder
	folderPath := objectType
	if targetDirectory != "" {
		folderPath = filepath.Join(targetDirectory, folderPath)
	}
	os.MkdirAll(folderPath, 0755)
	return fmt.Sprintf("I don't know how to implement '%%s' yet, so I've created a work directory for it in /%%s.", objectType, folderPath), true
}

func goImports(path string) {
	_ = exec.Command("go", "run", "golang.org/x/tools/cmd/goimports", "-w", path).Run()
}
