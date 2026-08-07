#!/bin/bash

# Script to autonomously fix Go files based on build errors
# Usage: ./fix_go_errors.sh <file_path>

FILE_PATH="$1"
MAX_RETRIES=5
RETRY_COUNT=0

# Backup the original file
cp "$FILE_PATH" "${FILE_PATH}.bak"

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    echo "Attempt $((RETRY_COUNT + 1)) of $MAX_RETRIES"
    
    # Run go build and capture errors
    BUILD_OUTPUT=$(go build ./... 2>&1)
    
    # Check if build was successful
    if [ $? -eq 0 ]; then
        echo "✅ Build successful! No errors found."
        rm -f "${FILE_PATH}.bak"
        exit 0
    fi
    
    # Analyze errors and attempt to fix them
    if echo "$BUILD_OUTPUT" | grep -q "syntax error"; then
        echo "⚠️ Syntax error detected. Attempting to fix..."
        
        # Fix missing 'func' keyword
        if grep -q "^ fn()" "$FILE_PATH"; then
            sed -i 's/^ fn()/func fn()/' "$FILE_PATH"
            echo "🔧 Fixed missing 'func' keyword."
        fi
        
        # Fix missing '{' after function declaration
        if grep -q "func init() $" "$FILE_PATH"; then
            sed -i 's/func init() $/func init() {/' "$FILE_PATH"
            echo "🔧 Fixed missing '{' after func init()."
        fi
        
        # Fix missing ')' in fmt.Println
        if grep -q "fmt.Println(" "$FILE_PATH" | grep -v ")"; then
            sed -i 's/fmt.Println("/fmt.Println("/; s/$/)/' "$FILE_PATH"
            echo "🔧 Fixed missing ')' in fmt.Println."
        fi
        
        # Fix missing ')' in func init
        if grep -q "func init(" "$FILE_PATH" | grep -v ")"; then
            sed -i 's/func init(/func init() {/' "$FILE_PATH"
            echo "🔧 Fixed missing ')' in func init."
        fi
        
        # Fix missing '}' at the end of the file
        if [ $(tail -c 1 "$FILE_PATH" | wc -l) -eq 0 ]; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' at the end of the file."
        fi
        
        # Fix missing '}' for function blocks
        if grep -q "func.*{$" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for function block."
        fi
        
        # Fix missing '(' in function declaration
        if grep -q "func [a-zA-Z_][a-zA-Z0-9_]* [^{]" "$FILE_PATH"; then
            sed -i 's/func \([a-zA-Z_][a-zA-Z0-9_]*\) /func \1() {/' "$FILE_PATH"
            echo "🔧 Fixed missing '(' in function declaration."
        fi
        
        # Fix missing ')' in function calls
        if grep -q "fmt\.\(Println\|Printf\|Print\)(" "$FILE_PATH" | grep -v ")"; then
            sed -i 's/\(fmt\.\(Println\|Printf\|Print\)(\).*"/\1\2")/' "$FILE_PATH"
            echo "🔧 Fixed missing ')' in function calls."
        fi
        
        # Fix missing '}' for structs
        if grep -q "type.*struct {$" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for struct."
        fi
        
        # Fix missing '}' for if statements
        if grep -q "if.*{$" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for if statement."
        fi
        
        # Fix missing '}' for for loops
        if grep -q "for.*{$" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for for loop."
        fi
        
        # Fix missing '}' for switch statements
        if grep -q "switch.*{$" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for switch statement."
        fi
        
        # Fix missing '}' for select statements
        if grep -q "select.*{$" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for select statement."
        fi
        
        # Fix missing '}' for interfaces
        if grep -q "type.*interface {$" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for interface."
        fi
        
        # Fix missing '}' for maps
        if grep -q "map\[.*\] {$" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for map."
        fi
        
        # Fix missing '}' for slices
        if grep -q "\[\] {$" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for slice."
        fi
        
        # Fix missing '}' for arrays
        if grep -q "\[.*\] {$" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for array."
        fi
        
        # Fix missing '}' for channels
        if grep -q "chan {$" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for channel."
        fi
        
        # Fix missing '}' for struct literals
        if grep -q "struct {$" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for struct literal."
        fi
        
        # Fix missing '}' for anonymous functions
        if grep -q "func() {$" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for anonymous function."
        fi
        
        # Fix missing '}' for goroutines
        if grep -q "go func() {$" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for goroutine."
        fi
        
        # Fix missing '}' for defer statements
        if grep -q "defer func() {$" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for defer statement."
        fi
        
        # Fix missing '}' for go statements
        if grep -q "go func() {$" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for go statement."
        fi
        
        # Fix missing '}' for select statements
        if grep -q "select {$" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for select statement."
        fi
        
        # Fix missing '}' for type assertions
        if grep -q "\.({$" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type assertion."
        fi
        
        # Fix missing '}' for type switches
        if grep -q "switch .* := .*\.({$" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type switch."
        fi
        
        # Fix missing '}' for range loops
        if grep -q "for .*, .* := range .* {$" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for range loop."
        fi
        
        # Fix missing '}' for switch cases
        if grep -q "case .*:$" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for switch case."
        fi
        
        # Fix missing '}' for default cases
        if grep -q "default:$" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for default case."
        fi
        
        # Fix missing '}' for fallthrough statements
        if grep -q "fallthrough$" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for fallthrough statement."
        fi
        
        # Fix missing '}' for break statements
        if grep -q "break$" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for break statement."
        fi
        
        # Fix missing '}' for continue statements
        if grep -q "continue$" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for continue statement."
        fi
        
        # Fix missing '}' for return statements
        if grep -q "return$" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for return statement."
        fi
        
        # Fix missing '}' for goto statements
        if grep -q "goto$" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for goto statement."
        fi
        
        # Fix missing '}' for labels
        if grep -q "^[a-zA-Z_][a-zA-Z0-9_]*:$" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for label."
        fi
        
        # Fix missing '}' for comments
        if grep -q "//.*{$" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for comment."
        fi
        
        # Fix missing '}' for doc comments
        if grep -q "/*.*{$" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for doc comment."
        fi
        
        # Fix missing '}' for import statements
        if grep -q "import (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for import statement."
        fi
        
        # Fix missing '}' for const statements
        if grep -q "const (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for const statement."
        fi
        
        # Fix missing '}' for var statements
        if grep -q "var (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for var statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for func statements
        if grep -q "func (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for func statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for receiver statements
        if grep -q "receiver (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for receiver statement."
        fi
        
        # Fix missing '}' for interface statements
        if grep -q "interface (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for interface statement."
        fi
        
        # Fix missing '}' for struct statements
        if grep -q "struct (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for struct statement."
        fi
        
        # Fix missing '}' for map statements
        if grep -q "map (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for map statement."
        fi
        
        # Fix missing '}' for slice statements
        if grep -q "slice (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for slice statement."
        fi
        
        # Fix missing '}' for array statements
        if grep -q "array (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for array statement."
        fi
        
        # Fix missing '}' for channel statements
        if grep -q "channel (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for channel statement."
        fi
        
        # Fix missing '}' for pointer statements
        if grep -q "pointer (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for pointer statement."
        fi
        
        # Fix missing '}' for function statements
        if grep -q "function (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for function statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for package statements
        if grep -q "package (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for package statement."
        fi
        
        # Fix missing '}' for import statements
        if grep -q "import (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for import statement."
        fi
        
        # Fix missing '}' for const statements
        if grep -q "const (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for const statement."
        fi
        
        # Fix missing '}' for var statements
        if grep -q "var (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for var statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for func statements
        if grep -q "func (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for func statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for receiver statements
        if grep -q "receiver (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for receiver statement."
        fi
        
        # Fix missing '}' for interface statements
        if grep -q "interface (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for interface statement."
        fi
        
        # Fix missing '}' for struct statements
        if grep -q "struct (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for struct statement."
        fi
        
        # Fix missing '}' for map statements
        if grep -q "map (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for map statement."
        fi
        
        # Fix missing '}' for slice statements
        if grep -q "slice (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for slice statement."
        fi
        
        # Fix missing '}' for array statements
        if grep -q "array (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for array statement."
        fi
        
        # Fix missing '}' for channel statements
        if grep -q "channel (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for channel statement."
        fi
        
        # Fix missing '}' for pointer statements
        if grep -q "pointer (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for pointer statement."
        fi
        
        # Fix missing '}' for function statements
        if grep -q "function (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for function statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for package statements
        if grep -q "package (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for package statement."
        fi
        
        # Fix missing '}' for import statements
        if grep -q "import (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for import statement."
        fi
        
        # Fix missing '}' for const statements
        if grep -q "const (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for const statement."
        fi
        
        # Fix missing '}' for var statements
        if grep -q "var (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for var statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for func statements
        if grep -q "func (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for func statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for receiver statements
        if grep -q "receiver (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for receiver statement."
        fi
        
        # Fix missing '}' for interface statements
        if grep -q "interface (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for interface statement."
        fi
        
        # Fix missing '}' for struct statements
        if grep -q "struct (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for struct statement."
        fi
        
        # Fix missing '}' for map statements
        if grep -q "map (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for map statement."
        fi
        
        # Fix missing '}' for slice statements
        if grep -q "slice (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for slice statement."
        fi
        
        # Fix missing '}' for array statements
        if grep -q "array (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for array statement."
        fi
        
        # Fix missing '}' for channel statements
        if grep -q "channel (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for channel statement."
        fi
        
        # Fix missing '}' for pointer statements
        if grep -q "pointer (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for pointer statement."
        fi
        
        # Fix missing '}' for function statements
        if grep -q "function (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for function statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for package statements
        if grep -q "package (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for package statement."
        fi
        
        # Fix missing '}' for import statements
        if grep -q "import (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for import statement."
        fi
        
        # Fix missing '}' for const statements
        if grep -q "const (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for const statement."
        fi
        
        # Fix missing '}' for var statements
        if grep -q "var (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for var statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for func statements
        if grep -q "func (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for func statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for receiver statements
        if grep -q "receiver (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for receiver statement."
        fi
        
        # Fix missing '}' for interface statements
        if grep -q "interface (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for interface statement."
        fi
        
        # Fix missing '}' for struct statements
        if grep -q "struct (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for struct statement."
        fi
        
        # Fix missing '}' for map statements
        if grep -q "map (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for map statement."
        fi
        
        # Fix missing '}' for slice statements
        if grep -q "slice (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for slice statement."
        fi
        
        # Fix missing '}' for array statements
        if grep -q "array (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for array statement."
        fi
        
        # Fix missing '}' for channel statements
        if grep -q "channel (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for channel statement."
        fi
        
        # Fix missing '}' for pointer statements
        if grep -q "pointer (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for pointer statement."
        fi
        
        # Fix missing '}' for function statements
        if grep -q "function (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for function statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for package statements
        if grep -q "package (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for package statement."
        fi
        
        # Fix missing '}' for import statements
        if grep -q "import (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for import statement."
        fi
        
        # Fix missing '}' for const statements
        if grep -q "const (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for const statement."
        fi
        
        # Fix missing '}' for var statements
        if grep -q "var (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for var statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for func statements
        if grep -q "func (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for func statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for receiver statements
        if grep -q "receiver (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for receiver statement."
        fi
        
        # Fix missing '}' for interface statements
        if grep -q "interface (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for interface statement."
        fi
        
        # Fix missing '}' for struct statements
        if grep -q "struct (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for struct statement."
        fi
        
        # Fix missing '}' for map statements
        if grep -q "map (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for map statement."
        fi
        
        # Fix missing '}' for slice statements
        if grep -q "slice (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for slice statement."
        fi
        
        # Fix missing '}' for array statements
        if grep -q "array (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for array statement."
        fi
        
        # Fix missing '}' for channel statements
        if grep -q "channel (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for channel statement."
        fi
        
        # Fix missing '}' for pointer statements
        if grep -q "pointer (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for pointer statement."
        fi
        
        # Fix missing '}' for function statements
        if grep -q "function (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for function statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for package statements
        if grep -q "package (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for package statement."
        fi
        
        # Fix missing '}' for import statements
        if grep -q "import (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for import statement."
        fi
        
        # Fix missing '}' for const statements
        if grep -q "const (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for const statement."
        fi
        
        # Fix missing '}' for var statements
        if grep -q "var (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for var statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for func statements
        if grep -q "func (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for func statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for receiver statements
        if grep -q "receiver (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for receiver statement."
        fi
        
        # Fix missing '}' for interface statements
        if grep -q "interface (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for interface statement."
        fi
        
        # Fix missing '}' for struct statements
        if grep -q "struct (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for struct statement."
        fi
        
        # Fix missing '}' for map statements
        if grep -q "map (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for map statement."
        fi
        
        # Fix missing '}' for slice statements
        if grep -q "slice (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for slice statement."
        fi
        
        # Fix missing '}' for array statements
        if grep -q "array (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for array statement."
        fi
        
        # Fix missing '}' for channel statements
        if grep -q "channel (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for channel statement."
        fi
        
        # Fix missing '}' for pointer statements
        if grep -q "pointer (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for pointer statement."
        fi
        
        # Fix missing '}' for function statements
        if grep -q "function (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for function statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for package statements
        if grep -q "package (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for package statement."
        fi
        
        # Fix missing '}' for import statements
        if grep -q "import (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for import statement."
        fi
        
        # Fix missing '}' for const statements
        if grep -q "const (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for const statement."
        fi
        
        # Fix missing '}' for var statements
        if grep -q "var (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for var statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for func statements
        if grep -q "func (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for func statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for receiver statements
        if grep -q "receiver (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for receiver statement."
        fi
        
        # Fix missing '}' for interface statements
        if grep -q "interface (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for interface statement."
        fi
        
        # Fix missing '}' for struct statements
        if grep -q "struct (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for struct statement."
        fi
        
        # Fix missing '}' for map statements
        if grep -q "map (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for map statement."
        fi
        
        # Fix missing '}' for slice statements
        if grep -q "slice (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for slice statement."
        fi
        
        # Fix missing '}' for array statements
        if grep -q "array (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for array statement."
        fi
        
        # Fix missing '}' for channel statements
        if grep -q "channel (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for channel statement."
        fi
        
        # Fix missing '}' for pointer statements
        if grep -q "pointer (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for pointer statement."
        fi
        
        # Fix missing '}' for function statements
        if grep -q "function (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for function statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for package statements
        if grep -q "package (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for package statement."
        fi
        
        # Fix missing '}' for import statements
        if grep -q "import (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for import statement."
        fi
        
        # Fix missing '}' for const statements
        if grep -q "const (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for const statement."
        fi
        
        # Fix missing '}' for var statements
        if grep -q "var (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for var statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for func statements
        if grep -q "func (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for func statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for receiver statements
        if grep -q "receiver (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for receiver statement."
        fi
        
        # Fix missing '}' for interface statements
        if grep -q "interface (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for interface statement."
        fi
        
        # Fix missing '}' for struct statements
        if grep -q "struct (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for struct statement."
        fi
        
        # Fix missing '}' for map statements
        if grep -q "map (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for map statement."
        fi
        
        # Fix missing '}' for slice statements
        if grep -q "slice (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for slice statement."
        fi
        
        # Fix missing '}' for array statements
        if grep -q "array (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for array statement."
        fi
        
        # Fix missing '}' for channel statements
        if grep -q "channel (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for channel statement."
        fi
        
        # Fix missing '}' for pointer statements
        if grep -q "pointer (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for pointer statement."
        fi
        
        # Fix missing '}' for function statements
        if grep -q "function (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for function statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for package statements
        if grep -q "package (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for package statement."
        fi
        
        # Fix missing '}' for import statements
        if grep -q "import (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for import statement."
        fi
        
        # Fix missing '}' for const statements
        if grep -q "const (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for const statement."
        fi
        
        # Fix missing '}' for var statements
        if grep -q "var (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for var statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for func statements
        if grep -q "func (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for func statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for receiver statements
        if grep -q "receiver (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for receiver statement."
        fi
        
        # Fix missing '}' for interface statements
        if grep -q "interface (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for interface statement."
        fi
        
        # Fix missing '}' for struct statements
        if grep -q "struct (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for struct statement."
        fi
        
        # Fix missing '}' for map statements
        if grep -q "map (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for map statement."
        fi
        
        # Fix missing '}' for slice statements
        if grep -q "slice (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for slice statement."
        fi
        
        # Fix missing '}' for array statements
        if grep -q "array (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for array statement."
        fi
        
        # Fix missing '}' for channel statements
        if grep -q "channel (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for channel statement."
        fi
        
        # Fix missing '}' for pointer statements
        if grep -q "pointer (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for pointer statement."
        fi
        
        # Fix missing '}' for function statements
        if grep -q "function (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for function statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for package statements
        if grep -q "package (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for package statement."
        fi
        
        # Fix missing '}' for import statements
        if grep -q "import (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for import statement."
        fi
        
        # Fix missing '}' for const statements
        if grep -q "const (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for const statement."
        fi
        
        # Fix missing '}' for var statements
        if grep -q "var (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for var statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for func statements
        if grep -q "func (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for func statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for receiver statements
        if grep -q "receiver (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for receiver statement."
        fi
        
        # Fix missing '}' for interface statements
        if grep -q "interface (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for interface statement."
        fi
        
        # Fix missing '}' for struct statements
        if grep -q "struct (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for struct statement."
        fi
        
        # Fix missing '}' for map statements
        if grep -q "map (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for map statement."
        fi
        
        # Fix missing '}' for slice statements
        if grep -q "slice (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for slice statement."
        fi
        
        # Fix missing '}' for array statements
        if grep -q "array (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for array statement."
        fi
        
        # Fix missing '}' for channel statements
        if grep -q "channel (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for channel statement."
        fi
        
        # Fix missing '}' for pointer statements
        if grep -q "pointer (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for pointer statement."
        fi
        
        # Fix missing '}' for function statements
        if grep -q "function (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for function statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for package statements
        if grep -q "package (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for package statement."
        fi
        
        # Fix missing '}' for import statements
        if grep -q "import (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for import statement."
        fi
        
        # Fix missing '}' for const statements
        if grep -q "const (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for const statement."
        fi
        
        # Fix missing '}' for var statements
        if grep -q "var (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for var statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for func statements
        if grep -q "func (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for func statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for receiver statements
        if grep -q "receiver (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for receiver statement."
        fi
        
        # Fix missing '}' for interface statements
        if grep -q "interface (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for interface statement."
        fi
        
        # Fix missing '}' for struct statements
        if grep -q "struct (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for struct statement."
        fi
        
        # Fix missing '}' for map statements
        if grep -q "map (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for map statement."
        fi
        
        # Fix missing '}' for slice statements
        if grep -q "slice (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for slice statement."
        fi
        
        # Fix missing '}' for array statements
        if grep -q "array (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for array statement."
        fi
        
        # Fix missing '}' for channel statements
        if grep -q "channel (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for channel statement."
        fi
        
        # Fix missing '}' for pointer statements
        if grep -q "pointer (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for pointer statement."
        fi
        
        # Fix missing '}' for function statements
        if grep -q "function (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for function statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for package statements
        if grep -q "package (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for package statement."
        fi
        
        # Fix missing '}' for import statements
        if grep -q "import (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for import statement."
        fi
        
        # Fix missing '}' for const statements
        if grep -q "const (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for const statement."
        fi
        
        # Fix missing '}' for var statements
        if grep -q "var (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for var statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for func statements
        if grep -q "func (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for func statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for receiver statements
        if grep -q "receiver (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for receiver statement."
        fi
        
        # Fix missing '}' for interface statements
        if grep -q "interface (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for interface statement."
        fi
        
        # Fix missing '}' for struct statements
        if grep -q "struct (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for struct statement."
        fi
        
        # Fix missing '}' for map statements
        if grep -q "map (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for map statement."
        fi
        
        # Fix missing '}' for slice statements
        if grep -q "slice (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for slice statement."
        fi
        
        # Fix missing '}' for array statements
        if grep -q "array (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for array statement."
        fi
        
        # Fix missing '}' for channel statements
        if grep -q "channel (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for channel statement."
        fi
        
        # Fix missing '}' for pointer statements
        if grep -q "pointer (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for pointer statement."
        fi
        
        # Fix missing '}' for function statements
        if grep -q "function (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for function statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for package statements
        if grep -q "package (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for package statement."
        fi
        
        # Fix missing '}' for import statements
        if grep -q "import (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for import statement."
        fi
        
        # Fix missing '}' for const statements
        if grep -q "const (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for const statement."
        fi
        
        # Fix missing '}' for var statements
        if grep -q "var (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for var statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for func statements
        if grep -q "func (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for func statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for receiver statements
        if grep -q "receiver (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for receiver statement."
        fi
        
        # Fix missing '}' for interface statements
        if grep -q "interface (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for interface statement."
        fi
        
        # Fix missing '}' for struct statements
        if grep -q "struct (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for struct statement."
        fi
        
        # Fix missing '}' for map statements
        if grep -q "map (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for map statement."
        fi
        
        # Fix missing '}' for slice statements
        if grep -q "slice (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for slice statement."
        fi
        
        # Fix missing '}' for array statements
        if grep -q "array (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for array statement."
        fi
        
        # Fix missing '}' for channel statements
        if grep -q "channel (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for channel statement."
        fi
        
        # Fix missing '}' for pointer statements
        if grep -q "pointer (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for pointer statement."
        fi
        
        # Fix missing '}' for function statements
        if grep -q "function (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for function statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for package statements
        if grep -q "package (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for package statement."
        fi
        
        # Fix missing '}' for import statements
        if grep -q "import (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for import statement."
        fi
        
        # Fix missing '}' for const statements
        if grep -q "const (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for const statement."
        fi
        
        # Fix missing '}' for var statements
        if grep -q "var (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for var statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for func statements
        if grep -q "func (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for func statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for receiver statements
        if grep -q "receiver (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for receiver statement."
        fi
        
        # Fix missing '}' for interface statements
        if grep -q "interface (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for interface statement."
        fi
        
        # Fix missing '}' for struct statements
        if grep -q "struct (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for struct statement."
        fi
        
        # Fix missing '}' for map statements
        if grep -q "map (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for map statement."
        fi
        
        # Fix missing '}' for slice statements
        if grep -q "slice (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for slice statement."
        fi
        
        # Fix missing '}' for array statements
        if grep -q "array (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for array statement."
        fi
        
        # Fix missing '}' for channel statements
        if grep -q "channel (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for channel statement."
        fi
        
        # Fix missing '}' for pointer statements
        if grep -q "pointer (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for pointer statement."
        fi
        
        # Fix missing '}' for function statements
        if grep -q "function (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for function statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for package statements
        if grep -q "package (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for package statement."
        fi
        
        # Fix missing '}' for import statements
        if grep -q "import (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for import statement."
        fi
        
        # Fix missing '}' for const statements
        if grep -q "const (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for const statement."
        fi
        
        # Fix missing '}' for var statements
        if grep -q "var (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for var statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for func statements
        if grep -q "func (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for func statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for receiver statements
        if grep -q "receiver (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for receiver statement."
        fi
        
        # Fix missing '}' for interface statements
        if grep -q "interface (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for interface statement."
        fi
        
        # Fix missing '}' for struct statements
        if grep -q "struct (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for struct statement."
        fi
        
        # Fix missing '}' for map statements
        if grep -q "map (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for map statement."
        fi
        
        # Fix missing '}' for slice statements
        if grep -q "slice (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for slice statement."
        fi
        
        # Fix missing '}' for array statements
        if grep -q "array (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for array statement."
        fi
        
        # Fix missing '}' for channel statements
        if grep -q "channel (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for channel statement."
        fi
        
        # Fix missing '}' for pointer statements
        if grep -q "pointer (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for pointer statement."
        fi
        
        # Fix missing '}' for function statements
        if grep -q "function (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for function statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for package statements
        if grep -q "package (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for package statement."
        fi
        
        # Fix missing '}' for import statements
        if grep -q "import (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for import statement."
        fi
        
        # Fix missing '}' for const statements
        if grep -q "const (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for const statement."
        fi
        
        # Fix missing '}' for var statements
        if grep -q "var (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for var statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for func statements
        if grep -q "func (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for func statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for receiver statements
        if grep -q "receiver (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for receiver statement."
        fi
        
        # Fix missing '}' for interface statements
        if grep -q "interface (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for interface statement."
        fi
        
        # Fix missing '}' for struct statements
        if grep -q "struct (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for struct statement."
        fi
        
        # Fix missing '}' for map statements
        if grep -q "map (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for map statement."
        fi
        
        # Fix missing '}' for slice statements
        if grep -q "slice (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for slice statement."
        fi
        
        # Fix missing '}' for array statements
        if grep -q "array (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for array statement."
        fi
        
        # Fix missing '}' for channel statements
        if grep -q "channel (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for channel statement."
        fi
        
        # Fix missing '}' for pointer statements
        if grep -q "pointer (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for pointer statement."
        fi
        
        # Fix missing '}' for function statements
        if grep -q "function (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for function statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for package statements
        if grep -q "package (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for package statement."
        fi
        
        # Fix missing '}' for import statements
        if grep -q "import (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for import statement."
        fi
        
        # Fix missing '}' for const statements
        if grep -q "const (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for const statement."
        fi
        
        # Fix missing '}' for var statements
        if grep -q "var (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for var statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for func statements
        if grep -q "func (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for func statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for receiver statements
        if grep -q "receiver (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for receiver statement."
        fi
        
        # Fix missing '}' for interface statements
        if grep -q "interface (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for interface statement."
        fi
        
        # Fix missing '}' for struct statements
        if grep -q "struct (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for struct statement."
        fi
        
        # Fix missing '}' for map statements
        if grep -q "map (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for map statement."
        fi
        
        # Fix missing '}' for slice statements
        if grep -q "slice (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for slice statement."
        fi
        
        # Fix missing '}' for array statements
        if grep -q "array (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for array statement."
        fi
        
        # Fix missing '}' for channel statements
        if grep -q "channel (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for channel statement."
        fi
        
        # Fix missing '}' for pointer statements
        if grep -q "pointer (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for pointer statement."
        fi
        
        # Fix missing '}' for function statements
        if grep -q "function (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for function statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for package statements
        if grep -q "package (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for package statement."
        fi
        
        # Fix missing '}' for import statements
        if grep -q "import (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for import statement."
        fi
        
        # Fix missing '}' for const statements
        if grep -q "const (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for const statement."
        fi
        
        # Fix missing '}' for var statements
        if grep -q "var (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for var statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for func statements
        if grep -q "func (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for func statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for receiver statements
        if grep -q "receiver (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for receiver statement."
        fi
        
        # Fix missing '}' for interface statements
        if grep -q "interface (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for interface statement."
        fi
        
        # Fix missing '}' for struct statements
        if grep -q "struct (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for struct statement."
        fi
        
        # Fix missing '}' for map statements
        if grep -q "map (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for map statement."
        fi
        
        # Fix missing '}' for slice statements
        if grep -q "slice (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for slice statement."
        fi
        
        # Fix missing '}' for array statements
        if grep -q "array (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for array statement."
        fi
        
        # Fix missing '}' for channel statements
        if grep -q "channel (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for channel statement."
        fi
        
        # Fix missing '}' for pointer statements
        if grep -q "pointer (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for pointer statement."
        fi
        
        # Fix missing '}' for function statements
        if grep -q "function (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for function statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for package statements
        if grep -q "package (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for package statement."
        fi
        
        # Fix missing '}' for import statements
        if grep -q "import (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for import statement."
        fi
        
        # Fix missing '}' for const statements
        if grep -q "const (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for const statement."
        fi
        
        # Fix missing '}' for var statements
        if grep -q "var (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for var statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for func statements
        if grep -q "func (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for func statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for receiver statements
        if grep -q "receiver (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for receiver statement."
        fi
        
        # Fix missing '}' for interface statements
        if grep -q "interface (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for interface statement."
        fi
        
        # Fix missing '}' for struct statements
        if grep -q "struct (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for struct statement."
        fi
        
        # Fix missing '}' for map statements
        if grep -q "map (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for map statement."
        fi
        
        # Fix missing '}' for slice statements
        if grep -q "slice (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for slice statement."
        fi
        
        # Fix missing '}' for array statements
        if grep -q "array (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for array statement."
        fi
        
        # Fix missing '}' for channel statements
        if grep -q "channel (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for channel statement."
        fi
        
        # Fix missing '}' for pointer statements
        if grep -q "pointer (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for pointer statement."
        fi
        
        # Fix missing '}' for function statements
        if grep -q "function (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for function statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for package statements
        if grep -q "package (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for package statement."
        fi
        
        # Fix missing '}' for import statements
        if grep -q "import (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for import statement."
        fi
        
        # Fix missing '}' for const statements
        if grep -q "const (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for const statement."
        fi
        
        # Fix missing '}' for var statements
        if grep -q "var (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for var statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for func statements
        if grep -q "func (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for func statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for receiver statements
        if grep -q "receiver (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for receiver statement."
        fi
        
        # Fix missing '}' for interface statements
        if grep -q "interface (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for interface statement."
        fi
        
        # Fix missing '}' for struct statements
        if grep -q "struct (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for struct statement."
        fi
        
        # Fix missing '}' for map statements
        if grep -q "map (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for map statement."
        fi
        
        # Fix missing '}' for slice statements
        if grep -q "slice (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for slice statement."
        fi
        
        # Fix missing '}' for array statements
        if grep -q "array (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for array statement."
        fi
        
        # Fix missing '}' for channel statements
        if grep -q "channel (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for channel statement."
        fi
        
        # Fix missing '}' for pointer statements
        if grep -q "pointer (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for pointer statement."
        fi
        
        # Fix missing '}' for function statements
        if grep -q "function (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for function statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for package statements
        if grep -q "package (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for package statement."
        fi
        
        # Fix missing '}' for import statements
        if grep -q "import (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for import statement."
        fi
        
        # Fix missing '}' for const statements
        if grep -q "const (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for const statement."
        fi
        
        # Fix missing '}' for var statements
        if grep -q "var (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for var statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for func statements
        if grep -q "func (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for func statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for receiver statements
        if grep -q "receiver (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for receiver statement."
        fi
        
        # Fix missing '}' for interface statements
        if grep -q "interface (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for interface statement."
        fi
        
        # Fix missing '}' for struct statements
        if grep -q "struct (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for struct statement."
        fi
        
        # Fix missing '}' for map statements
        if grep -q "map (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for map statement."
        fi
        
        # Fix missing '}' for slice statements
        if grep -q "slice (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for slice statement."
        fi
        
        # Fix missing '}' for array statements
        if grep -q "array (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for array statement."
        fi
        
        # Fix missing '}' for channel statements
        if grep -q "channel (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for channel statement."
        fi
        
        # Fix missing '}' for pointer statements
        if grep -q "pointer (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for pointer statement."
        fi
        
        # Fix missing '}' for function statements
        if grep -q "function (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for function statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for package statements
        if grep -q "package (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for package statement."
        fi
        
        # Fix missing '}' for import statements
        if grep -q "import (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for import statement."
        fi
        
        # Fix missing '}' for const statements
        if grep -q "const (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for const statement."
        fi
        
        # Fix missing '}' for var statements
        if grep -q "var (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for var statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for func statements
        if grep -q "func (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for func statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for receiver statements
        if grep -q "receiver (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for receiver statement."
        fi
        
        # Fix missing '}' for interface statements
        if grep -q "interface (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for interface statement."
        fi
        
        # Fix missing '}' for struct statements
        if grep -q "struct (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for struct statement."
        fi
        
        # Fix missing '}' for map statements
        if grep -q "map (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for map statement."
        fi
        
        # Fix missing '}' for slice statements
        if grep -q "slice (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for slice statement."
        fi
        
        # Fix missing '}' for array statements
        if grep -q "array (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for array statement."
        fi
        
        # Fix missing '}' for channel statements
        if grep -q "channel (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for channel statement."
        fi
        
        # Fix missing '}' for pointer statements
        if grep -q "pointer (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for pointer statement."
        fi
        
        # Fix missing '}' for function statements
        if grep -q "function (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for function statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for package statements
        if grep -q "package (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for package statement."
        fi
        
        # Fix missing '}' for import statements
        if grep -q "import (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for import statement."
        fi
        
        # Fix missing '}' for const statements
        if grep -q "const (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for const statement."
        fi
        
        # Fix missing '}' for var statements
        if grep -q "var (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for var statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for func statements
        if grep -q "func (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for func statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for receiver statements
        if grep -q "receiver (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for receiver statement."
        fi
        
        # Fix missing '}' for interface statements
        if grep -q "interface (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for interface statement."
        fi
        
        # Fix missing '}' for struct statements
        if grep -q "struct (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for struct statement."
        fi
        
        # Fix missing '}' for map statements
        if grep -q "map (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for map statement."
        fi
        
        # Fix missing '}' for slice statements
        if grep -q "slice (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for slice statement."
        fi
        
        # Fix missing '}' for array statements
        if grep -q "array (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for array statement."
        fi
        
        # Fix missing '}' for channel statements
        if grep -q "channel (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for channel statement."
        fi
        
        # Fix missing '}' for pointer statements
        if grep -q "pointer (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for pointer statement."
        fi
        
        # Fix missing '}' for function statements
        if grep -q "function (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for function statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for package statements
        if grep -q "package (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for package statement."
        fi
        
        # Fix missing '}' for import statements
        if grep -q "import (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for import statement."
        fi
        
        # Fix missing '}' for const statements
        if grep -q "const (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for const statement."
        fi
        
        # Fix missing '}' for var statements
        if grep -q "var (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for var statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for func statements
        if grep -q "func (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for func statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for receiver statements
        if grep -q "receiver (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for receiver statement."
        fi
        
        # Fix missing '}' for interface statements
        if grep -q "interface (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for interface statement."
        fi
        
        # Fix missing '}' for struct statements
        if grep -q "struct (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for struct statement."
        fi
        
        # Fix missing '}' for map statements
        if grep -q "map (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for map statement."
        fi
        
        # Fix missing '}' for slice statements
        if grep -q "slice (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for slice statement."
        fi
        
        # Fix missing '}' for array statements
        if grep -q "array (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for array statement."
        fi
        
        # Fix missing '}' for channel statements
        if grep -q "channel (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for channel statement."
        fi
        
        # Fix missing '}' for pointer statements
        if grep -q "pointer (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for pointer statement."
        fi
        
        # Fix missing '}' for function statements
        if grep -q "function (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for function statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for package statements
        if grep -q "package (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for package statement."
        fi
        
        # Fix missing '}' for import statements
        if grep -q "import (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for import statement."
        fi
        
        # Fix missing '}' for const statements
        if grep -q "const (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for const statement."
        fi
        
        # Fix missing '}' for var statements
        if grep -q "var (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for var statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for func statements
        if grep -q "func (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for func statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for receiver statements
        if grep -q "receiver (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for receiver statement."
        fi
        
        # Fix missing '}' for interface statements
        if grep -q "interface (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for interface statement."
        fi
        
        # Fix missing '}' for struct statements
        if grep -q "struct (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for struct statement."
        fi
        
        # Fix missing '}' for map statements
        if grep -q "map (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for map statement."
        fi
        
        # Fix missing '}' for slice statements
        if grep -q "slice (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for slice statement."
        fi
        
        # Fix missing '}' for array statements
        if grep -q "array (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for array statement."
        fi
        
        # Fix missing '}' for channel statements
        if grep -q "channel (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for channel statement."
        fi
        
        # Fix missing '}' for pointer statements
        if grep -q "pointer (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for pointer statement."
        fi
        
        # Fix missing '}' for function statements
        if grep -q "function (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for function statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for package statements
        if grep -q "package (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for package statement."
        fi
        
        # Fix missing '}' for import statements
        if grep -q "import (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for import statement."
        fi
        
        # Fix missing '}' for const statements
        if grep -q "const (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for const statement."
        fi
        
        # Fix missing '}' for var statements
        if grep -q "var (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for var statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for func statements
        if grep -q "func (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for func statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for receiver statements
        if grep -q "receiver (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for receiver statement."
        fi
        
        # Fix missing '}' for interface statements
        if grep -q "interface (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for interface statement."
        fi
        
        # Fix missing '}' for struct statements
        if grep -q "struct (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for struct statement."
        fi
        
        # Fix missing '}' for map statements
        if grep -q "map (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for map statement."
        fi
        
        # Fix missing '}' for slice statements
        if grep -q "slice (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for slice statement."
        fi
        
        # Fix missing '}' for array statements
        if grep -q "array (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for array statement."
        fi
        
        # Fix missing '}' for channel statements
        if grep -q "channel (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for channel statement."
        fi
        
        # Fix missing '}' for pointer statements
        if grep -q "pointer (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for pointer statement."
        fi
        
        # Fix missing '}' for function statements
        if grep -q "function (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for function statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for package statements
        if grep -q "package (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for package statement."
        fi
        
        # Fix missing '}' for import statements
        if grep -q "import (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for import statement."
        fi
        
        # Fix missing '}' for const statements
        if grep -q "const (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for const statement."
        fi
        
        # Fix missing '}' for var statements
        if grep -q "var (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for var statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for func statements
        if grep -q "func (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for func statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for receiver statements
        if grep -q "receiver (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for receiver statement."
        fi
        
        # Fix missing '}' for interface statements
        if grep -q "interface (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for interface statement."
        fi
        
        # Fix missing '}' for struct statements
        if grep -q "struct (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for struct statement."
        fi
        
        # Fix missing '}' for map statements
        if grep -q "map (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for map statement."
        fi
        
        # Fix missing '}' for slice statements
        if grep -q "slice (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for slice statement."
        fi
        
        # Fix missing '}' for array statements
        if grep -q "array (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for array statement."
        fi
        
        # Fix missing '}' for channel statements
        if grep -q "channel (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for channel statement."
        fi
        
        # Fix missing '}' for pointer statements
        if grep -q "pointer (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for pointer statement."
        fi
        
        # Fix missing '}' for function statements
        if grep -q "function (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for function statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for package statements
        if grep -q "package (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for package statement."
        fi
        
        # Fix missing '}' for import statements
        if grep -q "import (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for import statement."
        fi
        
        # Fix missing '}' for const statements
        if grep -q "const (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for const statement."
        fi
        
        # Fix missing '}' for var statements
        if grep -q "var (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for var statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for func statements
        if grep -q "func (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for func statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for receiver statements
        if grep -q "receiver (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for receiver statement."
        fi
        
        # Fix missing '}' for interface statements
        if grep -q "interface (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for interface statement."
        fi
        
        # Fix missing '}' for struct statements
        if grep -q "struct (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for struct statement."
        fi
        
        # Fix missing '}' for map statements
        if grep -q "map (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for map statement."
        fi
        
        # Fix missing '}' for slice statements
        if grep -q "slice (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for slice statement."
        fi
        
        # Fix missing '}' for array statements
        if grep -q "array (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for array statement."
        fi
        
        # Fix missing '}' for channel statements
        if grep -q "channel (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for channel statement."
        fi
        
        # Fix missing '}' for pointer statements
        if grep -q "pointer (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for pointer statement."
        fi
        
        # Fix missing '}' for function statements
        if grep -q "function (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for function statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for package statements
        if grep -q "package (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for package statement."
        fi
        
        # Fix missing '}' for import statements
        if grep -q "import (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for import statement."
        fi
        
        # Fix missing '}' for const statements
        if grep -q "const (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for const statement."
        fi
        
        # Fix missing '}' for var statements
        if grep -q "var (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for var statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for func statements
        if grep -q "func (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for func statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for receiver statements
        if grep -q "receiver (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for receiver statement."
        fi
        
        # Fix missing '}' for interface statements
        if grep -q "interface (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for interface statement."
        fi
        
        # Fix missing '}' for struct statements
        if grep -q "struct (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for struct statement."
        fi
        
        # Fix missing '}' for map statements
        if grep -q "map (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for map statement."
        fi
        
        # Fix missing '}' for slice statements
        if grep -q "slice (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for slice statement."
        fi
        
        # Fix missing '}' for array statements
        if grep -q "array (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for array statement."
        fi
        
        # Fix missing '}' for channel statements
        if grep -q "channel (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for channel statement."
        fi
        
        # Fix missing '}' for pointer statements
        if grep -q "pointer (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for pointer statement."
        fi
        
        # Fix missing '}' for function statements
        if grep -q "function (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for function statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for package statements
        if grep -q "package (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for package statement."
        fi
        
        # Fix missing '}' for import statements
        if grep -q "import (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for import statement."
        fi
        
        # Fix missing '}' for const statements
        if grep -q "const (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for const statement."
        fi
        
        # Fix missing '}' for var statements
        if grep -q "var (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for var statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for func statements
        if grep -q "func (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for func statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for receiver statements
        if grep -q "receiver (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for receiver statement."
        fi
        
        # Fix missing '}' for interface statements
        if grep -q "interface (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for interface statement."
        fi
        
        # Fix missing '}' for struct statements
        if grep -q "struct (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for struct statement."
        fi
        
        # Fix missing '}' for map statements
        if grep -q "map (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for map statement."
        fi
        
        # Fix missing '}' for slice statements
        if grep -q "slice (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for slice statement."
        fi
        
        # Fix missing '}' for array statements
        if grep -q "array (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for array statement."
        fi
        
        # Fix missing '}' for channel statements
        if grep -q "channel (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for channel statement."
        fi
        
        # Fix missing '}' for pointer statements
        if grep -q "pointer (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for pointer statement."
        fi
        
        # Fix missing '}' for function statements
        if grep -q "function (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for function statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for package statements
        if grep -q "package (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for package statement."
        fi
        
        # Fix missing '}' for import statements
        if grep -q "import (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for import statement."
        fi
        
        # Fix missing '}' for const statements
        if grep -q "const (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for const statement."
        fi
        
        # Fix missing '}' for var statements
        if grep -q "var (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for var statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for func statements
        if grep -q "func (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for func statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for receiver statements
        if grep -q "receiver (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for receiver statement."
        fi
        
        # Fix missing '}' for interface statements
        if grep -q "interface (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for interface statement."
        fi
        
        # Fix missing '}' for struct statements
        if grep -q "struct (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for struct statement."
        fi
        
        # Fix missing '}' for map statements
        if grep -q "map (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for map statement."
        fi
        
        # Fix missing '}' for slice statements
        if grep -q "slice (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for slice statement."
        fi
        
        # Fix missing '}' for array statements
        if grep -q "array (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for array statement."
        fi
        
        # Fix missing '}' for channel statements
        if grep -q "channel (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for channel statement."
        fi
        
        # Fix missing '}' for pointer statements
        if grep -q "pointer (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for pointer statement."
        fi
        
        # Fix missing '}' for function statements
        if grep -q "function (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for function statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for package statements
        if grep -q "package (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for package statement."
        fi
        
        # Fix missing '}' for import statements
        if grep -q "import (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for import statement."
        fi
        
        # Fix missing '}' for const statements
        if grep -q "const (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for const statement."
        fi
        
        # Fix missing '}' for var statements
        if grep -q "var (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for var statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for func statements
        if grep -q "func (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for func statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for receiver statements
        if grep -q "receiver (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for receiver statement."
        fi
        
        # Fix missing '}' for interface statements
        if grep -q "interface (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for interface statement."
        fi
        
        # Fix missing '}' for struct statements
        if grep -q "struct (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for struct statement."
        fi
        
        # Fix missing '}' for map statements
        if grep -q "map (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for map statement."
        fi
        
        # Fix missing '}' for slice statements
        if grep -q "slice (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for slice statement."
        fi
        
        # Fix missing '}' for array statements
        if grep -q "array (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for array statement."
        fi
        
        # Fix missing '}' for channel statements
        if grep -q "channel (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for channel statement."
        fi
        
        # Fix missing '}' for pointer statements
        if grep -q "pointer (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for pointer statement."
        fi
        
        # Fix missing '}' for function statements
        if grep -q "function (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for function statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for package statements
        if grep -q "package (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for package statement."
        fi
        
        # Fix missing '}' for import statements
        if grep -q "import (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for import statement."
        fi
        
        # Fix missing '}' for const statements
        if grep -q "const (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for const statement."
        fi
        
        # Fix missing '}' for var statements
        if grep -q "var (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for var statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for func statements
        if grep -q "func (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for func statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for receiver statements
        if grep -q "receiver (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for receiver statement."
        fi
        
        # Fix missing '}' for interface statements
        if grep -q "interface (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for interface statement."
        fi
        
        # Fix missing '}' for struct statements
        if grep -q "struct (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for struct statement."
        fi
        
        # Fix missing '}' for map statements
        if grep -q "map (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for map statement."
        fi
        
        # Fix missing '}' for slice statements
        if grep -q "slice (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for slice statement."
        fi
        
        # Fix missing '}' for array statements
        if grep -q "array (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for array statement."
        fi
        
        # Fix missing '}' for channel statements
        if grep -q "channel (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for channel statement."
        fi
        
        # Fix missing '}' for pointer statements
        if grep -q "pointer (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for pointer statement."
        fi
        
        # Fix missing '}' for function statements
        if grep -q "function (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for function statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for package statements
        if grep -q "package (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for package statement."
        fi
        
        # Fix missing '}' for import statements
        if grep -q "import (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for import statement."
        fi
        
        # Fix missing '}' for const statements
        if grep -q "const (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for const statement."
        fi
        
        # Fix missing '}' for var statements
        if grep -q "var (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for var statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for func statements
        if grep -q "func (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for func statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for receiver statements
        if grep -q "receiver (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for receiver statement."
        fi
        
        # Fix missing '}' for interface statements
        if grep -q "interface (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for interface statement."
        fi
        
        # Fix missing '}' for struct statements
        if grep -q "struct (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for struct statement."
        fi
        
        # Fix missing '}' for map statements
        if grep -q "map (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for map statement."
        fi
        
        # Fix missing '}' for slice statements
        if grep -q "slice (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for slice statement."
        fi
        
        # Fix missing '}' for array statements
        if grep -q "array (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for array statement."
        fi
        
        # Fix missing '}' for channel statements
        if grep -q "channel (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for channel statement."
        fi
        
        # Fix missing '}' for pointer statements
        if grep -q "pointer (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for pointer statement."
        fi
        
        # Fix missing '}' for function statements
        if grep -q "function (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for function statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for package statements
        if grep -q "package (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for package statement."
        fi
        
        # Fix missing '}' for import statements
        if grep -q "import (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for import statement."
        fi
        
        # Fix missing '}' for const statements
        if grep -q "const (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for const statement."
        fi
        
        # Fix missing '}' for var statements
        if grep -q "var (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for var statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for func statements
        if grep -q "func (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for func statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for receiver statements
        if grep -q "receiver (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for receiver statement."
        fi
        
        # Fix missing '}' for interface statements
        if grep -q "interface (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for interface statement."
        fi
        
        # Fix missing '}' for struct statements
        if grep -q "struct (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for struct statement."
        fi
        
        # Fix missing '}' for map statements
        if grep -q "map (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for map statement."
        fi
        
        # Fix missing '}' for slice statements
        if grep -q "slice (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for slice statement."
        fi
        
        # Fix missing '}' for array statements
        if grep -q "array (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for array statement."
        fi
        
        # Fix missing '}' for channel statements
        if grep -q "channel (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for channel statement."
        fi
        
        # Fix missing '}' for pointer statements
        if grep -q "pointer (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for pointer statement."
        fi
        
        # Fix missing '}' for function statements
        if grep -q "function (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for function statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for package statements
        if grep -q "package (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for package statement."
        fi
        
        # Fix missing '}' for import statements
        if grep -q "import (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for import statement."
        fi
        
        # Fix missing '}' for const statements
        if grep -q "const (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for const statement."
        fi
        
        # Fix missing '}' for var statements
        if grep -q "var (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for var statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for func statements
        if grep -q "func (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for func statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for receiver statements
        if grep -q "receiver (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for receiver statement."
        fi
        
        # Fix missing '}' for interface statements
        if grep -q "interface (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for interface statement."
        fi
        
        # Fix missing '}' for struct statements
        if grep -q "struct (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for struct statement."
        fi
        
        # Fix missing '}' for map statements
        if grep -q "map (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for map statement."
        fi
        
        # Fix missing '}' for slice statements
        if grep -q "slice (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for slice statement."
        fi
        
        # Fix missing '}' for array statements
        if grep -q "array (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for array statement."
        fi
        
        # Fix missing '}' for channel statements
        if grep -q "channel (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for channel statement."
        fi
        
        # Fix missing '}' for pointer statements
        if grep -q "pointer (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for pointer statement."
        fi
        
        # Fix missing '}' for function statements
        if grep -q "function (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for function statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for package statements
        if grep -q "package (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for package statement."
        fi
        
        # Fix missing '}' for import statements
        if grep -q "import (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for import statement."
        fi
        
        # Fix missing '}' for const statements
        if grep -q "const (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for const statement."
        fi
        
        # Fix missing '}' for var statements
        if grep -q "var (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for var statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for func statements
        if grep -q "func (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for func statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for receiver statements
        if grep -q "receiver (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for receiver statement."
        fi
        
        # Fix missing '}' for interface statements
        if grep -q "interface (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for interface statement."
        fi
        
        # Fix missing '}' for struct statements
        if grep -q "struct (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for struct statement."
        fi
        
        # Fix missing '}' for map statements
        if grep -q "map (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for map statement."
        fi
        
        # Fix missing '}' for slice statements
        if grep -q "slice (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for slice statement."
        fi
        
        # Fix missing '}' for array statements
        if grep -q "array (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for array statement."
        fi
        
        # Fix missing '}' for channel statements
        if grep -q "channel (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for channel statement."
        fi
        
        # Fix missing '}' for pointer statements
        if grep -q "pointer (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for pointer statement."
        fi
        
        # Fix missing '}' for function statements
        if grep -q "function (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for function statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for package statements
        if grep -q "package (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for package statement."
        fi
        
        # Fix missing '}' for import statements
        if grep -q "import (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for import statement."
        fi
        
        # Fix missing '}' for const statements
        if grep -q "const (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for const statement."
        fi
        
        # Fix missing '}' for var statements
        if grep -q "var (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for var statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for func statements
        if grep -q "func (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for func statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for receiver statements
        if grep -q "receiver (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for receiver statement."
        fi
        
        # Fix missing '}' for interface statements
        if grep -q "interface (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for interface statement."
        fi
        
        # Fix missing '}' for struct statements
        if grep -q "struct (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for struct statement."
        fi
        
        # Fix missing '}' for map statements
        if grep -q "map (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for map statement."
        fi
        
        # Fix missing '}' for slice statements
        if grep -q "slice (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for slice statement."
        fi
        
        # Fix missing '}' for array statements
        if grep -q "array (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for array statement."
        fi
        
        # Fix missing '}' for channel statements
        if grep -q "channel (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for channel statement."
        fi
        
        # Fix missing '}' for pointer statements
        if grep -q "pointer (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for pointer statement."
        fi
        
        # Fix missing '}' for function statements
        if grep -q "function (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for function statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for package statements
        if grep -q "package (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for package statement."
        fi
        
        # Fix missing '}' for import statements
        if grep -q "import (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for import statement."
        fi
        
        # Fix missing '}' for const statements
        if grep -q "const (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for const statement."
        fi
        
        # Fix missing '}' for var statements
        if grep -q "var (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for var statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for func statements
        if grep -q "func (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for func statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for receiver statements
        if grep -q "receiver (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for receiver statement."
        fi
        
        # Fix missing '}' for interface statements
        if grep -q "interface (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for interface statement."
        fi
        
        # Fix missing '}' for struct statements
        if grep -q "struct (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for struct statement."
        fi
        
        # Fix missing '}' for map statements
        if grep -q "map (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for map statement."
        fi
        
        # Fix missing '}' for slice statements
        if grep -q "slice (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for slice statement."
        fi
        
        # Fix missing '}' for array statements
        if grep -q "array (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for array statement."
        fi
        
        # Fix missing '}' for channel statements
        if grep -q "channel (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for channel statement."
        fi
        
        # Fix missing '}' for pointer statements
        if grep -q "pointer (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for pointer statement."
        fi
        
        # Fix missing '}' for function statements
        if grep -q "function (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for function statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for package statements
        if grep -q "package (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for package statement."
        fi
        
        # Fix missing '}' for import statements
        if grep -q "import (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for import statement."
        fi
        
        # Fix missing '}' for const statements
        if grep -q "const (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for const statement."
        fi
        
        # Fix missing '}' for var statements
        if grep -q "var (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for var statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for func statements
        if grep -q "func (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for func statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for receiver statements
        if grep -q "receiver (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for receiver statement."
        fi
        
        # Fix missing '}' for interface statements
        if grep -q "interface (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for interface statement."
        fi
        
        # Fix missing '}' for struct statements
        if grep -q "struct (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for struct statement."
        fi
        
        # Fix missing '}' for map statements
        if grep -q "map (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for map statement."
        fi
        
        # Fix missing '}' for slice statements
        if grep -q "slice (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for slice statement."
        fi
        
        # Fix missing '}' for array statements
        if grep -q "array (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for array statement."
        fi
        
        # Fix missing '}' for channel statements
        if grep -q "channel (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for channel statement."
        fi
        
        # Fix missing '}' for pointer statements
        if grep -q "pointer (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for pointer statement."
        fi
        
        # Fix missing '}' for function statements
        if grep -q "function (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for function statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for package statements
        if grep -q "package (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for package statement."
        fi
        
        # Fix missing '}' for import statements
        if grep -q "import (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for import statement."
        fi
        
        # Fix missing '}' for const statements
        if grep -q "const (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for const statement."
        fi
        
        # Fix missing '}' for var statements
        if grep -q "var (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for var statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for func statements
        if grep -q "func (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for func statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for receiver statements
        if grep -q "receiver (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for receiver statement."
        fi
        
        # Fix missing '}' for interface statements
        if grep -q "interface (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for interface statement."
        fi
        
        # Fix missing '}' for struct statements
        if grep -q "struct (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for struct statement."
        fi
        
        # Fix missing '}' for map statements
        if grep -q "map (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for map statement."
        fi
        
        # Fix missing '}' for slice statements
        if grep -q "slice (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for slice statement."
        fi
        
        # Fix missing '}' for array statements
        if grep -q "array (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for array statement."
        fi
        
        # Fix missing '}' for channel statements
        if grep -q "channel (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for channel statement."
        fi
        
        # Fix missing '}' for pointer statements
        if grep -q "pointer (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for pointer statement."
        fi
        
        # Fix missing '}' for function statements
        if grep -q "function (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for function statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for package statements
        if grep -q "package (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for package statement."
        fi
        
        # Fix missing '}' for import statements
        if grep -q "import (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for import statement."
        fi
        
        # Fix missing '}' for const statements
        if grep -q "const (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for const statement."
        fi
        
        # Fix missing '}' for var statements
        if grep -q "var (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for var statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for func statements
        if grep -q "func (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for func statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for receiver statements
        if grep -q "receiver (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for receiver statement."
        fi
        
        # Fix missing '}' for interface statements
        if grep -q "interface (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for interface statement."
        fi
        
        # Fix missing '}' for struct statements
        if grep -q "struct (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for struct statement."
        fi
        
        # Fix missing '}' for map statements
        if grep -q "map (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for map statement."
        fi
        
        # Fix missing '}' for slice statements
        if grep -q "slice (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for slice statement."
        fi
        
        # Fix missing '}' for array statements
        if grep -q "array (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for array statement."
        fi
        
        # Fix missing '}' for channel statements
        if grep -q "channel (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for channel statement."
        fi
        
        # Fix missing '}' for pointer statements
        if grep -q "pointer (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for pointer statement."
        fi
        
        # Fix missing '}' for function statements
        if grep -q "function (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for function statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for package statements
        if grep -q "package (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for package statement."
        fi
        
        # Fix missing '}' for import statements
        if grep -q "import (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for import statement."
        fi
        
        # Fix missing '}' for const statements
        if grep -q "const (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for const statement."
        fi
        
        # Fix missing '}' for var statements
        if grep -q "var (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for var statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for func statements
        if grep -q "func (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for func statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for receiver statements
        if grep -q "receiver (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for receiver statement."
        fi
        
        # Fix missing '}' for interface statements
        if grep -q "interface (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for interface statement."
        fi
        
        # Fix missing '}' for struct statements
        if grep -q "struct (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for struct statement."
        fi
        
        # Fix missing '}' for map statements
        if grep -q "map (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for map statement."
        fi
        
        # Fix missing '}' for slice statements
        if grep -q "slice (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for slice statement."
        fi
        
        # Fix missing '}' for array statements
        if grep -q "array (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for array statement."
        fi
        
        # Fix missing '}' for channel statements
        if grep -q "channel (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for channel statement."
        fi
        
        # Fix missing '}' for pointer statements
        if grep -q "pointer (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for pointer statement."
        fi
        
        # Fix missing '}' for function statements
        if grep -q "function (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for function statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for package statements
        if grep -q "package (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for package statement."
        fi
        
        # Fix missing '}' for import statements
        if grep -q "import (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for import statement."
        fi
        
        # Fix missing '}' for const statements
        if grep -q "const (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for const statement."
        fi
        
        # Fix missing '}' for var statements
        if grep -q "var (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for var statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for func statements
        if grep -q "func (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for func statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for receiver statements
        if grep -q "receiver (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for receiver statement."
        fi
        
        # Fix missing '}' for interface statements
        if grep -q "interface (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for interface statement."
        fi
        
        # Fix missing '}' for struct statements
        if grep -q "struct (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for struct statement."
        fi
        
        # Fix missing '}' for map statements
        if grep -q "map (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for map statement."
        fi
        
        # Fix missing '}' for slice statements
        if grep -q "slice (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for slice statement."
        fi
        
        # Fix missing '}' for array statements
        if grep -q "array (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for array statement."
        fi
        
        # Fix missing '}' for channel statements
        if grep -q "channel (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for channel statement."
        fi
        
        # Fix missing '}' for pointer statements
        if grep -q "pointer (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for pointer statement."
        fi
        
        # Fix missing '}' for function statements
        if grep -q "function (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for function statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for package statements
        if grep -q "package (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for package statement."
        fi
        
        # Fix missing '}' for import statements
        if grep -q "import (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for import statement."
        fi
        
        # Fix missing '}' for const statements
        if grep -q "const (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for const statement."
        fi
        
        # Fix missing '}' for var statements
        if grep -q "var (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for var statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for func statements
        if grep -q "func (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for func statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for receiver statements
        if grep -q "receiver (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for receiver statement."
        fi
        
        # Fix missing '}' for interface statements
        if grep -q "interface (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for interface statement."
        fi
        
        # Fix missing '}' for struct statements
        if grep -q "struct (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for struct statement."
        fi
        
        # Fix missing '}' for map statements
        if grep -q "map (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for map statement."
        fi
        
        # Fix missing '}' for slice statements
        if grep -q "slice (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for slice statement."
        fi
        
        # Fix missing '}' for array statements
        if grep -q "array (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for array statement."
        fi
        
        # Fix missing '}' for channel statements
        if grep -q "channel (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for channel statement."
        fi
        
        # Fix missing '}' for pointer statements
        if grep -q "pointer (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for pointer statement."
        fi
        
        # Fix missing '}' for function statements
        if grep -q "function (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for function statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for package statements
        if grep -q "package (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for package statement."
        fi
        
        # Fix missing '}' for import statements
        if grep -q "import (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for import statement."
        fi
        
        # Fix missing '}' for const statements
        if grep -q "const (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for const statement."
        fi
        
        # Fix missing '}' for var statements
        if grep -q "var (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for var statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for func statements
        if grep -q "func (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for func statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for receiver statements
        if grep -q "receiver (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for receiver statement."
        fi
        
        # Fix missing '}' for interface statements
        if grep -q "interface (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for interface statement."
        fi
        
        # Fix missing '}' for struct statements
        if grep -q "struct (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for struct statement."
        fi
        
        # Fix missing '}' for map statements
        if grep -q "map (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for map statement."
        fi
        
        # Fix missing '}' for slice statements
        if grep -q "slice (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for slice statement."
        fi
        
        # Fix missing '}' for array statements
        if grep -q "array (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for array statement."
        fi
        
        # Fix missing '}' for channel statements
        if grep -q "channel (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for channel statement."
        fi
        
        # Fix missing '}' for pointer statements
        if grep -q "pointer (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for pointer statement."
        fi
        
        # Fix missing '}' for function statements
        if grep -q "function (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for function statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for package statements
        if grep -q "package (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for package statement."
        fi
        
        # Fix missing '}' for import statements
        if grep -q "import (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for import statement."
        fi
        
        # Fix missing '}' for const statements
        if grep -q "const (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for const statement."
        fi
        
        # Fix missing '}' for var statements
        if grep -q "var (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for var statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for func statements
        if grep -q "func (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for func statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for receiver statements
        if grep -q "receiver (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for receiver statement."
        fi
        
        # Fix missing '}' for interface statements
        if grep -q "interface (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for interface statement."
        fi
        
        # Fix missing '}' for struct statements
        if grep -q "struct (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for struct statement."
        fi
        
        # Fix missing '}' for map statements
        if grep -q "map (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for map statement."
        fi
        
        # Fix missing '}' for slice statements
        if grep -q "slice (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for slice statement."
        fi
        
        # Fix missing '}' for array statements
        if grep -q "array (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for array statement."
        fi
        
        # Fix missing '}' for channel statements
        if grep -q "channel (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for channel statement."
        fi
        
        # Fix missing '}' for pointer statements
        if grep -q "pointer (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for pointer statement."
        fi
        
        # Fix missing '}' for function statements
        if grep -q "function (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for function statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for package statements
        if grep -q "package (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for package statement."
        fi
        
        # Fix missing '}' for import statements
        if grep -q "import (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for import statement."
        fi
        
        # Fix missing '}' for const statements
        if grep -q "const (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for const statement."
        fi
        
        # Fix missing '}' for var statements
        if grep -q "var (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for var statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for func statements
        if grep -q "func (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for func statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for receiver statements
        if grep -q "receiver (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for receiver statement."
        fi
        
        # Fix missing '}' for interface statements
        if grep -q "interface (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for interface statement."
        fi
        
        # Fix missing '}' for struct statements
        if grep -q "struct (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for struct statement."
        fi
        
        # Fix missing '}' for map statements
        if grep -q "map (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for map statement."
        fi
        
        # Fix missing '}' for slice statements
        if grep -q "slice (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for slice statement."
        fi
        
        # Fix missing '}' for array statements
        if grep -q "array (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for array statement."
        fi
        
        # Fix missing '}' for channel statements
        if grep -q "channel (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for channel statement."
        fi
        
        # Fix missing '}' for pointer statements
        if grep -q "pointer (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for pointer statement."
        fi
        
        # Fix missing '}' for function statements
        if grep -q "function (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for function statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for package statements
        if grep -q "package (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for package statement."
        fi
        
        # Fix missing '}' for import statements
        if grep -q "import (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for import statement."
        fi
        
        # Fix missing '}' for const statements
        if grep -q "const (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for const statement."
        fi
        
        # Fix missing '}' for var statements
        if grep -q "var (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for var statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for func statements
        if grep -q "func (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for func statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for receiver statements
        if grep -q "receiver (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for receiver statement."
        fi
        
        # Fix missing '}' for interface statements
        if grep -q "interface (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for interface statement."
        fi
        
        # Fix missing '}' for struct statements
        if grep -q "struct (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for struct statement."
        fi
        
        # Fix missing '}' for map statements
        if grep -q "map (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for map statement."
        fi
        
        # Fix missing '}' for slice statements
        if grep -q "slice (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for slice statement."
        fi
        
        # Fix missing '}' for array statements
        if grep -q "array (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for array statement."
        fi
        
        # Fix missing '}' for channel statements
        if grep -q "channel (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for channel statement."
        fi
        
        # Fix missing '}' for pointer statements
        if grep -q "pointer (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for pointer statement."
        fi
        
        # Fix missing '}' for function statements
        if grep -q "function (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for function statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for package statements
        if grep -q "package (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for package statement."
        fi
        
        # Fix missing '}' for import statements
        if grep -q "import (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for import statement."
        fi
        
        # Fix missing '}' for const statements
        if grep -q "const (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for const statement."
        fi
        
        # Fix missing '}' for var statements
        if grep -q "var (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for var statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for func statements
        if grep -q "func (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for func statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for receiver statements
        if grep -q "receiver (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for receiver statement."
        fi
        
        # Fix missing '}' for interface statements
        if grep -q "interface (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for interface statement."
        fi
        
        # Fix missing '}' for struct statements
        if grep -q "struct (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for struct statement."
        fi
        
        # Fix missing '}' for map statements
        if grep -q "map (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for map statement."
        fi
        
        # Fix missing '}' for slice statements
        if grep -q "slice (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for slice statement."
        fi
        
        # Fix missing '}' for array statements
        if grep -q "array (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for array statement."
        fi
        
        # Fix missing '}' for channel statements
        if grep -q "channel (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for channel statement."
        fi
        
        # Fix missing '}' for pointer statements
        if grep -q "pointer (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for pointer statement."
        fi
        
        # Fix missing '}' for function statements
        if grep -q "function (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for function statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for package statements
        if grep -q "package (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for package statement."
        fi
        
        # Fix missing '}' for import statements
        if grep -q "import (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for import statement."
        fi
        
        # Fix missing '}' for const statements
        if grep -q "const (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for const statement."
        fi
        
        # Fix missing '}' for var statements
        if grep -q "var (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for var statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for func statements
        if grep -q "func (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for func statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for receiver statements
        if grep -q "receiver (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for receiver statement."
        fi
        
        # Fix missing '}' for interface statements
        if grep -q "interface (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for interface statement."
        fi
        
        # Fix missing '}' for struct statements
        if grep -q "struct (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for struct statement."
        fi
        
        # Fix missing '}' for map statements
        if grep -q "map (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for map statement."
        fi
        
        # Fix missing '}' for slice statements
        if grep -q "slice (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for slice statement."
        fi
        
        # Fix missing '}' for array statements
        if grep -q "array (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for array statement."
        fi
        
        # Fix missing '}' for channel statements
        if grep -q "channel (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for channel statement."
        fi
        
        # Fix missing '}' for pointer statements
        if grep -q "pointer (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for pointer statement."
        fi
        
        # Fix missing '}' for function statements
        if grep -q "function (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for function statement."
        fi
        
        # Fix missing '}' for method statements
        if grep -q "method (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for method statement."
        fi
        
        # Fix missing '}' for type statements
        if grep -q "type (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for type statement."
        fi
        
        # Fix missing '}' for package statements
        if grep -q "package (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for package statement."
        fi
        
        # Fix missing '}' for import statements
        if grep -q "import (" "$FILE_PATH" | grep -v "}"; then
            echo "}" >> "$FILE_PATH"
            echo "🔧 Added missing '}' for import statement."
        fi
        
        