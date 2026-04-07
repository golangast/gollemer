# Set Environment
$env:CGO_ENABLED = "1"
$LocalMinGW = "$PSScriptRoot\tools\mingw64\bin"

if (Test-Path $LocalMinGW) {
    $env:PATH = "$LocalMinGW;" + $env:PATH
    Write-Host "Using local MinGW compiler: $LocalMinGW"
} else {
    Write-Host "Local MinGW not found at $LocalMinGW."
}

# Execute Training Tool
if ($args.Count -eq 0) {
    Write-Host "Running default training: -train-chat -gpu -dry-run"
    go run ./cmd/tools/train_moe/main.go -train-chat -gpu -dry-run
} else {
    Write-Host "Running with arguments: $args"
    go run ./cmd/tools/train_moe/main.go $args
}
