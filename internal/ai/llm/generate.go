package llm

import (
	"encoding/json"
	"fmt"
	"os"
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

	showHandlerContent := fmt.Sprintf(`
func Show%sHandler(w http.ResponseWriter, r *http.Request) {
	cwd, _ := os.Getwd()
	jsonPath := filepath.Join(cwd, "%s", "%s.json")
	
	data, err := os.ReadFile(jsonPath)
	if err != nil {
		if os.IsNotExist(err) {
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte("[]"))
			return
		}
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	var results []%s
	if err := json.Unmarshal(data, &results); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(results)
}`, structName, dirName, lowercaseName, structName)

	updateHandlerContent := fmt.Sprintf(`
func Update%sHandler(w http.ResponseWriter, r *http.Request) {
	parts := strings.Split(r.URL.Path, "/")
	if len(parts) < 4 {
		http.Error(w, "Invalid URL, expecting /update/%s/{id}", http.StatusBadRequest)
		return
	}
	idStr := parts[len(parts)-1]
	id, _ := strconv.Atoi(idStr)

	var updatedItem %s
	err := json.NewDecoder(r.Body).Decode(&updatedItem)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	cwd, _ := os.Getwd()
	jsonPath := filepath.Join(cwd, "%s", "%s.json")
	
	data, err := os.ReadFile(jsonPath)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	var items []%s
	json.Unmarshal(data, &items)

	found := false
	for i := range items {
		if items[i].ID == id {
			updatedItem.ID = id // Keep the ID
			items[i] = updatedItem
			found = true
			break
		}
	}

	if !found {
		http.Error(w, "Item not found", http.StatusNotFound)
		return
	}

	newData, _ := json.MarshalIndent(items, "", "  ")
	os.WriteFile(jsonPath, newData, 0644)

	fmt.Fprintf(w, "%s with ID %%d updated successfully", id)
}`, structName, lowercaseName, structName, dirName, lowercaseName, structName, structName)

	deleteHandlerContent := fmt.Sprintf(`
func Delete%sHandler(w http.ResponseWriter, r *http.Request) {
	parts := strings.Split(r.URL.Path, "/")
	if len(parts) < 4 {
		http.Error(w, "Invalid URL, expecting /delete/%s/{id}", http.StatusBadRequest)
		return
	}
	idStr := parts[len(parts)-1]
	id, _ := strconv.Atoi(idStr)

	cwd, _ := os.Getwd()
	jsonPath := filepath.Join(cwd, "%s", "%s.json")
	
	data, err := os.ReadFile(jsonPath)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	var items []%s
	json.Unmarshal(data, &items)

	newItems := make([]%s, 0)
	for _, item := range items {
		if item.ID != id {
			newItems = append(newItems, item)
		}
	}

	newData, _ := json.MarshalIndent(newItems, "", "  ")
	os.WriteFile(jsonPath, newData, 0644)

	fmt.Fprintf(w, "%s with ID %%d deleted successfully", id)
}`, structName, lowercaseName, dirName, lowercaseName, structName, structName, structName)

	packageFileContent := fmt.Sprintf(`package %s
import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

%s
%s
%s
%s
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
        var words = %s;
        var rawVectors = %s;

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
		// Strip existing "Handler" suffix to avoid doubling
		cleanName := strings.TrimSuffix(strings.Title(handlerName), "Handler")

		// Build file content with string concatenation (no fmt.Sprintf) to avoid format-verb bugs
		handlerContent := "package main\n\n" +
			"import (\n" +
			"\t\"fmt\"\n" +
			"\t\"net/http\"\n" +
			")\n\n" +
			"// " + cleanName + "Handler handles requests for " + handlerURL + "\n" +
			"func " + cleanName + "Handler(w http.ResponseWriter, r *http.Request) {\n" +
			"\tfmt.Fprintf(w, \"Hello from the " + cleanName + " handler!\")\n" +
			"}\n"

		filePath := strings.ToLower(cleanName) + ".go"
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
		regMsg, _ := registerHandlerURL(cleanName, handlerURL, mainPath)
		return fmt.Sprintf("I created '%sHandler' and registered it at '%s'. %s", cleanName, handlerURL, regMsg), true
	}

	// 2. If a name is provided, create a folder (for folder/directory) or a Go skeleton
	if fileName != "" {
		if strings.Contains(objectType, "folder") || strings.Contains(objectType, "directory") {
			// It's a folder creation: just mkdir
			folderPath := fileName
			if targetDirectory != "" {
				folderPath = filepath.Join(targetDirectory, fileName)
			}
			if err := os.MkdirAll(folderPath, 0755); err != nil {
				return fmt.Sprintf("I couldn't create the folder '%s': %v", folderPath, err), false
			}
			return fmt.Sprintf("I created the folder '%s' for you.", folderPath), true
		}

		// Otherwise create a Go skeleton file
		content := "// " + strings.Title(fileName) + " implementation for the " + objectType + " object\n" +
			"package main\n\nimport \"fmt\"\n\nfunc main() {\n" +
			"\tfmt.Println(\"Executing " + fileName + " (" + objectType + " logic)\")\n}\n"
		filePath := fileName + ".go"
		if targetDirectory != "" {
			filePath = filepath.Join(targetDirectory, filePath)
			os.MkdirAll(targetDirectory, 0755)
		}
		os.WriteFile(filePath, []byte(content), 0644)
		goImports(filePath)
		return fmt.Sprintf("I created the Go file '%s.go' for you.", fileName), true
	}

	// 3. Just a type name with no explicit file/folder name — create a work directory
	folderPath := objectType
	if targetDirectory != "" {
		folderPath = filepath.Join(targetDirectory, folderPath)
	}
	os.MkdirAll(folderPath, 0755)
	return fmt.Sprintf("I created a work directory '%s' for you.", folderPath), true
}

