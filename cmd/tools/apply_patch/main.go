package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

type SemanticOutput struct {
	Operation      string          `json:"operation"`
	TargetResource *ResourceTarget `json:"target_resource"`
}

type ResourceTarget struct {
	Type       string                 `json:"type"`
	Name       string                 `json:"name"`
	Content    string                 `json:"content,omitempty"`
	Properties map[string]interface{} `json:"properties"`
}

func main() {
	targetFile := flag.String("target", "./main.go", "Target file path to apply code changes")
	validate := flag.Bool("validate", true, "Run validation steps")
	flag.Parse()

	inputBytes, err := io.ReadAll(os.Stdin)
	if err != nil {
		log.Fatalf("Error reading stdin: %v", err)
	}

	inputText := string(inputBytes)
	jsonStart := strings.Index(inputText, "=== Generated Semantic Output ===")
	if jsonStart != -1 {
		inputText = inputText[jsonStart+len("=== Generated Semantic Output ==="):]
	}
	jsonEnd := strings.Index(inputText, "=================================")
	if jsonEnd != -1 {
		inputText = inputText[:jsonEnd]
	}

	inputText = strings.TrimSpace(inputText)
	if inputText == "" {
		log.Fatal("No valid JSON found in stdin stream")
	}

	var output SemanticOutput
	if err := json.Unmarshal([]byte(inputText), &output); err != nil {
		log.Fatalf("Failed to parse JSON input: %v", err)
	}

	if output.TargetResource == nil {
		log.Fatal("JSON output missing target_resource")
	}

	if err := applyPatch(*targetFile, output); err != nil {
		log.Fatalf("   failed for %s: %v", *targetFile, err)
	}

	if *validate && strings.HasSuffix(*targetFile, ".go") {
		if err := validateGoCode(*targetFile); err != nil {
			log.Fatalf("validation failed for %s: %v", *targetFile, err)
		}
	}

	fmt.Printf("Successfully applied patch and validated %s\n", *targetFile)
}

func applyPatch(targetFile string, output SemanticOutput) error {
	if output.TargetResource == nil {
		return fmt.Errorf("no target resource")
	}

	content := output.TargetResource.Content
	if content == "" {
		content = getStringProp(output.TargetResource.Properties, "content")
	}

	if content == "" {
		return fmt.Errorf("no content to apply")
	}

	if _, err := os.Stat(targetFile); os.IsNotExist(err) {
		dir := filepath.Dir(targetFile)
		if err := os.MkdirAll(dir, 0755); err != nil {
			return err
		}
	}

	existingBytes, err := os.ReadFile(targetFile)
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	existingContent := string(existingBytes)

	if !strings.HasSuffix(targetFile, ".go") {
		if !strings.Contains(existingContent, content) {
			if !strings.HasSuffix(existingContent, "\n") {
				existingContent += "\n"
			}
			existingContent += "\n\n" + content + "\n"
		}
		if err := os.WriteFile(targetFile, []byte(existingContent), 0644); err != nil {
			return err
		}
		return nil
	}

	strippedContent := stripPackageClause(content)
	if strings.TrimSpace(strippedContent) == "" {
		return nil
	}

	if strings.Contains(existingContent, strippedContent) {
		return nil
	}

	if err := applyGoCode(targetFile, existingContent, strippedContent); err != nil {
		return err
	}

	return nil
}

func stripPackageClause(content string) string {
	lines := strings.Split(content, "\n")
	var result []string
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "package ") {
			continue
		}
		result = append(result, line)
	}
	return strings.TrimSpace(strings.Join(result, "\n"))
}

func applyGoCode(targetFile, existingContent, newContent string) error {
	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, targetFile, existingContent, parser.ParseComments)
	if err != nil {
		return fmt.Errorf("failed to parse existing file: %w", err)
	}

	src := fmt.Sprintf("package main\n\n%s", newContent)
	newFset := token.NewFileSet()
	newNode, err := parser.ParseFile(newFset, "", src, parser.ParseComments)
	if err != nil {
		return fmt.Errorf("failed to parse new content: %w", err)
	}

	existingPackage := ""
	if node.Name != nil {
		existingPackage = node.Name.Name
	}

	var newFuncs []*ast.FuncDecl
	var newGenDecls []*ast.GenDecl

	for _, decl := range newNode.Decls {
		switch d := decl.(type) {
		case *ast.FuncDecl:
			if d.Name != nil && d.Name.Name == "main" && existingPackage != "main" {
				continue
			}
			if d.Name != nil && d.Name.Name == "main" {
				found := false
				for _, existing := range node.Decls {
					if existingFn, ok := existing.(*ast.FuncDecl); ok && existingFn.Name != nil && existingFn.Name.Name == "main" {
						found = true
						break
					}
				}
				if found {
					continue
				}
			}
			newFuncs = append(newFuncs, d)
		case *ast.GenDecl:
			newGenDecls = append(newGenDecls, d)
		}
	}

	for _, d := range newGenDecls {
		if d.Tok == token.IMPORT {
			replaced := false
			for i, existing := range node.Decls {
				if existingGen, ok := existing.(*ast.GenDecl); ok && existingGen.Tok == token.IMPORT {
					node.Decls[i] = d
					replaced = true
					break
				}
			}
			if !replaced {
				node.Decls = append([]ast.Decl{d}, node.Decls...)
			}
		} else {
			replaced := false
			for i, existing := range node.Decls {
				if existingGen, ok := existing.(*ast.GenDecl); ok && existingGen.Tok == d.Tok {
					node.Decls[i] = d
					replaced = true
					break
				}
			}
			if !replaced {
				node.Decls = append(node.Decls, d)
			}
		}
	}

	for _, d := range newFuncs {
		node.Decls = append(node.Decls, d)
	}

	f, err := os.Create(targetFile)
	if err != nil {
		return err
	}
	defer f.Close()

	return format.Node(f, fset, node)
}

func formatFile(targetFile string) error {
	if !strings.HasSuffix(targetFile, ".go") {
		return nil
	}

	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, targetFile, nil, parser.ParseComments)
	if err != nil {
		return err
	}

	f, err := os.Create(targetFile)
	if err != nil {
		return err
	}
	defer f.Close()

	return format.Node(f, fset, node)
}

func validateGoCode(targetFile string) error {
	dir := filepath.Dir(targetFile)

	gofmtCmd := exec.Command("gofmt", "-w", targetFile)
	if out, err := gofmtCmd.CombinedOutput(); err != nil {
		return fmt.Errorf("gofmt failed: %s (%v)", string(out), err)
	}

	vetCmd := exec.Command("go", "vet", ".")
	vetCmd.Dir = dir
	if out, err := vetCmd.CombinedOutput(); err != nil {
		return fmt.Errorf("go vet failed: %s", string(out))
	}

	tmpOutput := filepath.Join(os.TempDir(), fmt.Sprintf("  _build_%d", os.Getpid()))
	defer os.Remove(tmpOutput)

	buildCmd := exec.Command("go", "build", "-o", tmpOutput, ".")
	buildCmd.Dir = dir
	if out, err := buildCmd.CombinedOutput(); err != nil {
		cleanOut := strings.TrimSpace(string(out))
		if cleanOut == "" {
			cleanOut = err.Error()
		}
		return fmt.Errorf("go build failed: %s", cleanOut)
	}

	return nil
}

func getStringProp(props map[string]interface{}, keys ...string) string {
	if props == nil {
		return ""
	}
	for _, key := range keys {
		if val, ok := props[key].(string); ok && val != "" {
			return val
		}
	}
	return ""
}
