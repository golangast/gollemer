package semantic

import (
	"fmt"
	"slices"
	"strings"
)

// IntentType represents the classification of user intent
type IntentType string

const (
	IntentCreateFolder         IntentType = "create_folder"
	IntentCreateFile           IntentType = "create_file"
	IntentCreateFolderWithFile IntentType = "create_folder_with_file"
	IntentDeleteFile           IntentType = "delete_file"
	IntentDeleteFolder         IntentType = "delete_folder"
	IntentReadFile             IntentType = "read_file"
	IntentMoveFile             IntentType = "move_file"
	IntentMoveFolder           IntentType = "move_folder"
	IntentRenameFile           IntentType = "rename_file"
	IntentRenameFolder         IntentType = "rename_folder"
	IntentAddFeature           IntentType = "add_feature"
	IntentAddHandler           IntentType = "add_handler"
	IntentModifyCode           IntentType = "modify_code"
	GoToProject                IntentType = "go_to_project"
	RememberProjects           IntentType = "remember_projects"
	IntentUnknown              IntentType = "unknown"
)

// Template defines a JSON structure for an intent
type Template struct {
	Intent         IntentType
	Operation      string
	Fill           func(entities map[string]string) SemanticOutput
	FillStructured func(cmd *StructuredCommand) SemanticOutput // New: structured command support
}

// GetTemplates returns all defined templates
func GetTemplates() map[IntentType]Template {
	return map[IntentType]Template{
		IntentCreateFolder: {
			Intent:    IntentCreateFolder,
			Operation: "Create",
			Fill:      fillCreateFolder,
		},
		IntentCreateFile: {
			Intent:    IntentCreateFile,
			Operation: "Create",
			Fill:      fillCreateFile,
		},
		IntentCreateFolderWithFile: {
			Intent:    IntentCreateFolderWithFile,
			Operation: "Create",
			Fill:      fillCreateFolderWithFile,
		},
		IntentDeleteFile: {
			Intent:    IntentDeleteFile,
			Operation: "Delete",
			Fill:      fillDeleteFile,
		},
		IntentDeleteFolder: {
			Intent:    IntentDeleteFolder,
			Operation: "Delete",
			Fill:      fillDeleteFolder,
		},
		IntentReadFile: {
			Intent:    IntentReadFile,
			Operation: "Read",
			Fill:      fillReadFile,
		},
		IntentMoveFile: {
			Intent:    IntentMoveFile,
			Operation: "Move",
			Fill:      fillMoveFile,
		},
		IntentMoveFolder: {
			Intent:    IntentMoveFolder,
			Operation: "Move",
			Fill:      fillMoveFolder,
		},
		IntentRenameFile: {
			Intent:    IntentRenameFile,
			Operation: "Rename",
			Fill:      fillRenameFile,
		},
		IntentRenameFolder: {
			Intent:    IntentRenameFolder,
			Operation: "Rename",
			Fill:      fillRenameFolder,
		},
		IntentAddFeature: {
			Intent:    IntentAddFeature,
			Operation: "AddFeature",
			Fill:      fillAddFeature,
		},
		IntentAddHandler: {
			Intent:    IntentAddHandler,
			Operation: "Add",
			Fill:      fillAddHandler,
		},
		IntentModifyCode: {
			Intent:    IntentModifyCode,
			Operation: "Modify",
			Fill:      fillModifyCode,
		},
	}
}

// buildPath constructs a nested path from folder entities
func buildPath(entities map[string]string) string {
	// Collect path components in order
	var pathParts []string

	// Check for explicit path
	if path, ok := entities["path"]; ok && path != "" && path != "./" {
		return path
	}

	// Build path from folder entities
	// Priority: destination_folder > folder > source_folder
	if dest, ok := entities["destination_folder"]; ok && dest != "" {
		pathParts = append(pathParts, dest)
	}

	if component, ok := entities["component"]; ok && component != "" {
		pathParts = append(pathParts, component)
	}

	if folder, ok := entities["folder"]; ok && folder != "" {
		// Only add if not already in pathParts
		alreadyAdded := slices.Contains(pathParts, folder)
		if !alreadyAdded {
			pathParts = append(pathParts, folder)
		}
	}

	if len(pathParts) > 0 {
		return strings.Join(pathParts, "/")
	}

	return "./"
}

// fillCreateFolder fills template for folder creation
func fillCreateFolder(entities map[string]string) SemanticOutput {
	folderName := entities["folder"]
	if folderName == "" {
		folderName = "new_folder"
	}

	path := entities["path"]
	if path == "" {
		path = "./"
	}

	return SemanticOutput{
		Operation: "Create",
		TargetResource: &Resource{
			Type: "Filesystem::Folder",
			Name: folderName,
			Properties: map[string]any{
				"path": path,
			},
		},
		Context: Context{
			UserRole: "admin",
		},
	}
}

// fillCreateFile fills template for file creation
func fillCreateFile(entities map[string]string) SemanticOutput {
	fileName := entities["file"]
	if fileName == "" {
		fileName = entities["destination_file"] // Fallback
	}
	if fileName == "" {
		// Use file_type to generate filename if available
		fileType := entities["file_type"]
		if fileType != "" {
			switch fileType {
			case "html":
				fileName = "index.html"
			case "css":
				fileName = "style.css"
			case "js", "javascript":
				fileName = "index.js"
			case "python", "py":
				fileName = "main.py"
			case "java":
				fileName = "Main.java"
			case "go":
				fileName = "main.go"
			default:
				fileName = "file." + fileType
			}
		} else {
			fileName = "new_file.txt"
		}
	}

	path := buildPath(entities)

	properties := map[string]any{
		"path": path,
	}

	// Add code_type if present
	if codeType, ok := entities["code_type"]; ok {
		properties["code_type"] = codeType

		// Generate content based on component
		if component, ok := entities["component"]; ok {
			if strings.Contains(strings.ToLower(component), "handler") {
				properties["content"] = fmt.Sprintf(`package main

import (
	"fmt"
	"net/http"
)

// %s handles HTTP requests
func %s(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "Hello from %s!")
}
`, strings.Title(strings.TrimSuffix(fileName, ".go")), strings.Title(strings.TrimSuffix(fileName, ".go")), fileName)
			} else if strings.Contains(strings.ToLower(component), "wasm") || strings.Contains(strings.ToLower(path), "wasm") {
				// Generate WASM component boilerplate
				structName := strings.Title(strings.TrimSuffix(fileName, ".go"))
				properties["content"] = fmt.Sprintf(`package ui

import "syscall/js"

type %s struct {
	el js.Value
}

func New%s() *%s {
	return &%s{}
}

func (c *%s) Render() js.Value {
	document := js.Global().Get("document")
	div := document.Call("createElement", "div")
	div.Set("className", "%s-component")
	div.Set("innerText", "%s WASM Component")
	return div
}
`, structName, structName, structName, structName, structName, strings.ToLower(structName), structName)
			}
		}
	}

	// Special handling for colors
	if strings.Contains(strings.ToLower(fileName), "color") || strings.Contains(strings.ToLower(path), "color") {
		if strings.HasSuffix(fileName, ".css") {
			properties["content"] = `
:root {
    --primary: #6366f1;
    --secondary: #ec4899;
    --accent: #f59e0b;
    --bg-dark: #0f172a;
    --text-white: #f8fafc;
}
`
		}
	}

	return SemanticOutput{
		Operation: "Create",
		TargetResource: &Resource{
			Type:       "Filesystem::File",
			Name:       fileName,
			Properties: properties,
		},
		Context: Context{
			UserRole: "admin",
		},
	}
}

// fillCreateFolderWithFile fills template for folder with files
func fillCreateFolderWithFile(entities map[string]string) SemanticOutput {
	folderName := entities["folder"]
	if folderName == "" {
		folderName = "new_folder"
	}

	fileName := entities["file"]
	if fileName == "" {
		// Use file_type to generate filename if available
		fileType := entities["file_type"]
		if fileType != "" {
			switch fileType {
			case "html":
				fileName = "index.html"
			case "css":
				fileName = "style.css"
			case "js", "javascript":
				fileName = "index.js"
			case "python", "py":
				fileName = "main.py"
			case "java":
				fileName = "Main.java"
			case "go":
				fileName = "main.go"
			default:
				fileName = "file." + fileType
			}
		} else {
			fileName = "file.txt"
		}
	}

	path := buildPath(entities)

	// Create child file with optional code_type property
	childProps := make(map[string]any)
	if codeType, ok := entities["code_type"]; ok {
		childProps["code_type"] = codeType
	}

	children := []Resource{
		{
			Type:       "Filesystem::File",
			Name:       fileName,
			Properties: childProps,
		},
	}

	return SemanticOutput{
		Operation: "Create",
		TargetResource: &Resource{
			Type:     "Filesystem::Folder",
			Name:     folderName,
			Children: children,
			Properties: map[string]any{
				"path": path,
			},
		},
		Context: Context{
			UserRole: "admin",
		},
	}
}

// fillDeleteFile fills template for file deletion
func fillDeleteFile(entities map[string]string) SemanticOutput {
	fileName := entities["file"]
	if fileName == "" {
		fileName = "file.txt"
	}

	path := entities["path"]
	if path == "" {
		path = "./"
	}

	return SemanticOutput{
		Operation: "Delete",
		TargetResource: &Resource{
			Type: "Filesystem::File",
			Name: fileName,
			Properties: map[string]any{
				"path": path,
			},
		},
		Context: Context{
			UserRole: "admin",
		},
	}
}

// fillDeleteFolder fills template for folder deletion
func fillDeleteFolder(entities map[string]string) SemanticOutput {
	folderName := entities["folder"]
	if folderName == "" {
		folderName = "folder"
	}

	path := entities["path"]
	if path == "" {
		path = "./"
	}

	return SemanticOutput{
		Operation: "Delete",
		TargetResource: &Resource{
			Type: "Filesystem::Folder",
			Name: folderName,
			Properties: map[string]any{
				"path": path,
			},
		},
		Context: Context{
			UserRole: "admin",
		},
	}
}

// fillReadFile fills template for file reading
func fillReadFile(entities map[string]string) SemanticOutput {
	fileName := entities["file"]
	if fileName == "" {
		fileName = "file.txt"
	}

	path := entities["path"]
	if path == "" {
		path = "./"
	}

	return SemanticOutput{
		Operation: "Read",
		TargetResource: &Resource{
			Type: "Filesystem::File",
			Name: fileName,
			Properties: map[string]any{
				"path": path,
			},
		},
		Context: Context{
			UserRole: "admin",
		},
	}
}

// fillRenameFile fills template for file renaming
func fillRenameFile(entities map[string]string) SemanticOutput {
	sourceFile := entities["source_file"]
	if sourceFile == "" {
		sourceFile = entities["file"] // fallback
	}
	if sourceFile == "" {
		sourceFile = "file.txt"
	}

	// For renames, the new name might be in destination_file or file (if only one file detected)
	newName := entities["destination_file"]
	if newName == "" {
		newName = entities["destination_folder"] // Sometimes parsed as folder
	}
	if newName == "" {
		newName = "renamed_file.txt"
	}

	return SemanticOutput{
		Operation: "Rename",
		TargetResource: &Resource{
			Type: "Filesystem::File",
			Name: sourceFile,
			Properties: map[string]any{
				"new_name": newName,
			},
		},
		Context: Context{
			UserRole: "admin",
		},
	}
}

// fillRenameFolder fills template for folder renaming
func fillRenameFolder(entities map[string]string) SemanticOutput {
	sourceFolder := entities["source_folder"]
	if sourceFolder == "" {
		sourceFolder = entities["folder"] // fallback
	}
	if sourceFolder == "" {
		sourceFolder = "folder"
	}

	newName := entities["destination_folder"]
	if newName == "" {
		newName = "renamed_folder"
	}

	return SemanticOutput{
		Operation: "Rename",
		TargetResource: &Resource{
			Type: "Filesystem::Folder",
			Name: sourceFolder,
			Properties: map[string]any{
				"new_name": newName,
			},
		},
		Context: Context{
			UserRole: "admin",
		},
	}
}

// fillAddHandler fills template for adding an HTTP handler
func fillAddHandler(entities map[string]string) SemanticOutput {
	handlerName := entities["handler"]
	if handlerName == "" {
		handlerName = entities["name"]
	}
	if handlerName == "" {
		handlerName = entities["component"]
	}
	if handlerName == "" {
		handlerName = "jim"
	}

	url := entities["url"]
	if url == "" {
		url = "/" + handlerName
	}

	properties := map[string]any{
		"handler": handlerName,
		"url":     url,
		"content": fmt.Sprintf(`package main

import (
	"fmt"
	"net/http"
)

func %s(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "Hello from %s!")
}
`, handlerName, handlerName),
	}

	return SemanticOutput{
		Operation: "Add",
		TargetResource: &Resource{
			Type:       "Code::Component",
			Name:       handlerName,
			Properties: properties,
		},
		Context: Context{
			UserRole: "admin",
		},
	}
}

// fillAddFeature fills template for adding a feature to a component
func fillAddFeature(entities map[string]string) SemanticOutput {
	featureName := entities["feature"]
	if featureName == "" {
		featureName = "feature"
	}

	componentName := entities["component"]
	if componentName == "" {
		componentName = "component"
	}

	location := entities["folder"]
	if location == "" {
		location = "./"
	}

	properties := map[string]any{
		"feature":  featureName,
		"location": location,
	}

	return SemanticOutput{
		Operation: "AddFeature",
		TargetResource: &Resource{
			Type:       "Code::Component",
			Name:       componentName,
			Properties: properties,
		},
		Context: Context{
			UserRole: "admin",
		},
	}
}

// fillMoveFile fills template for file move/relocation
func fillMoveFile(entities map[string]string) SemanticOutput {
	sourceFile := entities["source_file"]
	if sourceFile == "" {
		sourceFile = entities["file"] // fallback
	}
	if sourceFile == "" {
		sourceFile = "file.txt"
	}

	destFolder := entities["destination_folder"]
	if destFolder == "" {
		destFolder = entities["folder"] // fallback
	}
	if destFolder == "" {
		destFolder = "./"
	}

	return SemanticOutput{
		Operation: "Move",
		TargetResource: &Resource{
			Type: "Filesystem::File",
			Name: sourceFile,
			Properties: map[string]any{
				"destination": destFolder,
			},
		},
		Context: Context{
			UserRole: "admin",
		},
	}
}

// fillMoveFolder fills template for folder move/relocation
func fillMoveFolder(entities map[string]string) SemanticOutput {
	sourceFolder := entities["source_folder"]
	if sourceFolder == "" {
		sourceFolder = entities["folder"] // fallback to first folder found
	}
	if sourceFolder == "" {
		sourceFolder = "folder"
	}

	destFolder := entities["destination_folder"]
	if destFolder == "" {
		destFolder = "./"
	}

	return SemanticOutput{
		Operation: "Move",
		TargetResource: &Resource{
			Type: "Filesystem::Folder",
			Name: sourceFolder,
			Properties: map[string]any{
				"destination": destFolder,
			},
		},
		Context: Context{
			UserRole: "admin",
		},
	}
}

// fillModifyCode fills template for generic code modification
func fillModifyCode(entities map[string]string) SemanticOutput {
	componentName := entities["component"]
	if componentName == "" {
		componentName = "code"
	}

	location := entities["folder"]
	if location == "" {
		location = "./"
	}

	properties := map[string]any{
		"location": location,
	}

	// Add feature if present
	if feature, ok := entities["feature"]; ok {
		properties["feature"] = feature
	}

	return SemanticOutput{
		Operation: "Modify",
		TargetResource: &Resource{
			Type:       "Code::Component",
			Name:       componentName,
			Properties: properties,
		},
		Context: Context{
			UserRole: "admin",
		},
	}
}

// ===== Structured Command Filling Functions =====

// FillFromStructuredCommand creates a SemanticOutput from a StructuredCommand
func FillFromStructuredCommand(cmd *StructuredCommand) SemanticOutput {
	output := SemanticOutput{
		Operation: string(cmd.Action),
		Context: Context{
			UserRole: "admin",
		},
	}

	// Determine resource type
	resourceType := objectTypeToResourceType(cmd.ObjectType)

	// Build properties
	properties := make(map[string]any)
	if cmd.Path != "" {
		properties["path"] = cmd.Path
	} else {
		properties["path"] = "./"
	}

	// Add custom properties
	for key, value := range cmd.Properties {
		properties[key] = value
	}

	// Handle action-specific properties
	switch cmd.Action {
	case ActionRename:
		if cmd.ArgumentName != "" {
			properties["new_name"] = cmd.ArgumentName
		}
	case ActionMove:
		if cmd.ArgumentName != "" {
			properties["destination"] = cmd.ArgumentName
		}
	case ActionAdd:
		if feature, ok := cmd.Properties["feature"]; ok {
			properties["feature"] = feature
		}
		// Special handling for handlers
		if cmd.ObjectType == ObjectComponent && strings.Contains(strings.ToLower(cmd.Name), "handler") {
			properties["content"] = fmt.Sprintf(`package main

import (
	"fmt"
	"net/http"
)

func %s(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "Hello from %s!")
}
`, cmd.Name, cmd.Name)
		}
	}

	// Build target resource
	output.TargetResource = &Resource{
		Type:       resourceType,
		Name:       cmd.Name,
		Properties: properties,
	}

	// Handle secondary operations (nested resources)
	if cmd.HasSecondaryOperation() {
		childType := objectTypeToResourceType(cmd.ArgumentType)
		childProps := make(map[string]any)

		// Copy relevant properties to child
		if codeType, ok := cmd.Properties["code_type"]; ok {
			childProps["code_type"] = codeType
		}

		child := Resource{
			Type:       childType,
			Name:       cmd.ArgumentName,
			Properties: childProps,
		}

		output.TargetResource.Children = []Resource{child}
	}

	return output
}

// objectTypeToResourceType converts ObjectType to resource type string
func objectTypeToResourceType(objType ObjectType) string {
	switch objType {
	case ObjectFile:
		return "Filesystem::File"
	case ObjectFolder:
		return "Filesystem::Folder"
	case ObjectComponent:
		return "Code::Component"
	case ObjectCode:
		return "Code::Component"
	case ObjectFeature:
		return "Code::Feature"
	default:
		return "Unknown"
	}
}

// FillFromHierarchicalCommand creates a SemanticOutput from a HierarchicalCommand
// Supports recursive children and template content
func FillFromHierarchicalCommand(cmd *HierarchicalCommand) SemanticOutput {
	output := SemanticOutput{
		Operation: string(cmd.Action),
		Context: Context{
			UserRole: "admin",
		},
	}

	// Determine resource type
	resourceType := objectTypeToResourceType(cmd.ObjectType)

	// Build properties
	properties := make(map[string]any)
	if cmd.Path != "" {
		properties["path"] = cmd.Path
	} else {
		properties["path"] = "./"
	}

	// Add template if present
	if cmd.Template != "" {
		properties["template"] = cmd.Template
	}

	// Add custom properties
	for key, value := range cmd.Properties {
		properties[key] = value
	}

	// Build target resource
	output.TargetResource = &Resource{
		Type:       resourceType,
		Name:       cmd.Name,
		Properties: properties,
	}

	// Handle children recursively
	if cmd.HasChildren() {
		children := make([]Resource, 0, len(cmd.Children))

		for _, child := range cmd.Children {
			childResource := hierarchicalCommandToResource(child)
			children = append(children, childResource)
		}

		output.TargetResource.Children = children
	}

	return output
}

// hierarchicalCommandToResource converts a HierarchicalCommand to a Resource
// Used for recursive child conversion
func hierarchicalCommandToResource(cmd *HierarchicalCommand) Resource {
	resourceType := objectTypeToResourceType(cmd.ObjectType)

	properties := make(map[string]any)

	// Add content if present (for files with boilerplate)
	if content, ok := cmd.Properties["content"]; ok {
		properties["content"] = content
	}

	// Add other properties
	for key, value := range cmd.Properties {
		if key != "content" {
			properties[key] = value
		}
	}

	resource := Resource{
		Type:       resourceType,
		Name:       cmd.Name,
		Properties: properties,
	}

	// Recursively convert children
	if cmd.HasChildren() {
		children := make([]Resource, 0, len(cmd.Children))
		for _, child := range cmd.Children {
			childResource := hierarchicalCommandToResource(child)
			children = append(children, childResource)
		}
		resource.Children = children
	}

	return resource
}
