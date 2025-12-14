package main

import (
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"io/ioutil"
	"os"
	"strings"
)

func main() {
	if len(os.Args) != 3 {
		fmt.Println("Usage: create_handler <handler_name> <url_path>")
		os.Exit(1)
	}

	handlerName := os.Args[1]
	urlPath := os.Args[2]

	// 1. Create the handler file
	err := createHandlerFile(handlerName)
	if err != nil {
		fmt.Printf("Error creating handler file: %v\n", err)
		os.Exit(1)
	}

	// 2. Add handler to main.go
	err = addHandlerToMain(handlerName, urlPath)
	if err != nil {
		fmt.Printf("Error updating main.go: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Successfully created handler '%s' and registered it to '%s'\n", handlerName, urlPath)
}

func createHandlerFile(handlerName string) error {
	capitalizedHandlerName := strings.Title(handlerName)
	fileName := strings.ToLower(handlerName) + ".go"

	if _, err := os.Stat(fileName); err == nil {
		fmt.Printf("File '%s' already exists. Overwriting.\n", fileName)
	}

	content := fmt.Sprintf(`package main

import (
	"fmt"
	"net/http"
)

func %sHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "Executing %sHandler! Request URL: %%s\n", r.URL.Path)
}
`, capitalizedHandlerName, capitalizedHandlerName)

	return ioutil.WriteFile(fileName, []byte(content), 0644)
}

func addHandlerToMain(handlerName, urlPath string) error {
	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, "main.go", nil, parser.ParseComments)
	if err != nil {
		return err
	}

	capitalizedHandlerName := strings.Title(handlerName)
	handlerFuncName := fmt.Sprintf("%sHandler", capitalizedHandlerName)
	handlerExists := false
	var mainFunc *ast.FuncDecl

	// Find the main function and check for existing handler
	for _, f := range node.Decls {
		fn, ok := f.(*ast.FuncDecl)
		if ok && fn.Name.Name == "main" {
			mainFunc = fn
			for _, stmt := range fn.Body.List {
				if call, ok := stmt.(*ast.ExprStmt); ok {
					if call, ok := call.X.(*ast.CallExpr); ok {
						if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
							if sel.X.(*ast.Ident).Name == "http" && sel.Sel.Name == "HandleFunc" {
								if len(call.Args) == 2 {
									if path, ok := call.Args[0].(*ast.BasicLit); ok {
										if path.Value == fmt.Sprintf("\"%s\"", urlPath) {
											handlerExists = true
											break
										}
									}
								}
							}
						}
					}
				}
			}
			if handlerExists {
				break
			}
		}
	}

	if handlerExists {
		fmt.Printf("Handler for path '%s' already exists in main.go. Nothing to do.\n", urlPath)
		return nil
	}

	if mainFunc == nil {
		return fmt.Errorf("main function not found in main.go")
	}

	// Create the new HandleFunc call
	newCall := &ast.ExprStmt{
		X: &ast.CallExpr{
			Fun: &ast.SelectorExpr{
				X:   ast.NewIdent("http"),
				Sel: ast.NewIdent("HandleFunc"),
			},
			Args: []ast.Expr{
				&ast.BasicLit{
					Kind:  token.STRING,
					Value: fmt.Sprintf("\"%s\"", urlPath),
				},
				ast.NewIdent(handlerFuncName),
			},
		},
	}

	// Insert the new call before the log.Println
	for i, stmt := range mainFunc.Body.List {
		if call, ok := stmt.(*ast.ExprStmt); ok {
			if call, ok := call.X.(*ast.CallExpr); ok {
				if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
					if sel.X.(*ast.Ident).Name == "log" && sel.Sel.Name == "Println" {
						// Insert before this statement
						mainFunc.Body.List = append(mainFunc.Body.List[:i], append([]ast.Stmt{newCall}, mainFunc.Body.List[i:]...)...)
						goto write_file
					}
				}
			}
		}
	}
	// If no log.Println is found, just append it before the last statement (usually log.Fatal)
	if len(mainFunc.Body.List) > 0 {
		lastStmt := mainFunc.Body.List[len(mainFunc.Body.List)-1]
		mainFunc.Body.List = append(mainFunc.Body.List[:len(mainFunc.Body.List)-1], newCall, lastStmt)
	} else {
		mainFunc.Body.List = append(mainFunc.Body.List, newCall)
	}


write_file:
	// Write the modified AST back to the file
	f, err := os.Create("main.go")
	if err != nil {
		return err
	}
	defer f.Close()

	return format.Node(f, fset, node)
}
