#!/usr/bin/env bash
set -e

HOOK_DIR=".git/hooks"
HOOK_FILE="$HOOK_DIR/pre-commit"

if [ ! -d "$HOOK_DIR" ]; then
    mkdir -p "$HOOK_DIR"
fi

cat << 'EOF' > "$HOOK_FILE"
#!/usr/bin/env bash
set -e

echo "🔍 Running Gollemer Pre-Commit AI Validation & Self-Healing..."

STAGED_GO_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep '\.go$' || true)

if [ -z "$STAGED_GO_FILES" ]; then
    echo "✅ No staged Go files to validate."
    exit 0
fi

# Build gollemer binary if missing
if [ ! -f "./bin/gollemer" ]; then
    echo "🔨 Building gollemer binary..."
    go build -o ./bin/gollemer ./cmd/gollemer
fi

FAILED=0

for FILE in $STAGED_GO_FILES; do
    if [ -f "$FILE" ]; then
        echo "Checking $FILE..."
        gofmt -w "$FILE"
        git add "$FILE"
        
        # Run vet and build check
        DIR=$(dirname "$FILE")
        if ! (cd "$DIR" && go vet . >/dev/null 2>&1 && go build -o /tmp/precommit_check . >/dev/null 2>&1); then
            echo "⚠️ Validation issue in $FILE. Attempting Gollemer self-healing patch..."
            rm -f /tmp/precommit_check
            if ./bin/gollemer patch "fix syntax and compilation errors" -target="$FILE"; then
                echo "✅ Self-healing succeeded for $FILE"
                git add "$FILE"
            else
                echo "❌ Self-healing failed for $FILE"
                FAILED=1
            fi
        fi
        rm -f /tmp/precommit_check
    fi
done

if [ $FAILED -ne 0 ]; then
    echo "❌ Pre-commit validation failed. Please fix remaining errors before committing."
    exit 1
fi

echo "✅ All staged Go files passed Gollemer pre-commit validation!"
exit 0
EOF

chmod +x "$HOOK_FILE"
echo "✅ Gollemer Git pre-commit hook installed successfully at $HOOK_FILE"
