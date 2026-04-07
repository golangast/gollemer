# Gollemer MoE Training Pipeline (Windows PowerShell - Clean ASCII Version)

# --- 1. Environment & Path Setup ---
$ProjectRoot = Get-Location
$Env:GOEXPERIMENT = "simd"
$Env:CGO_ENABLED = "0"

# Ensure critical directories exist
$Dirs = @("logs", "research_logs", "data/models/checkpoints", "data/models/gob_models")
foreach ($Dir in $Dirs) {
    if (-not (Test-Path $Dir)) {
        New-Item -ItemType Directory -Path $Dir | Out-Null
    }
}

if (-not (Test-Path "logs/training.csv")) {
    New-Item -ItemType File -Path "logs/training.csv" | Out-Null
}

# --- 2. Housekeeping ---
$OldModels = @(
    "data/models/gob_models/seq2seq_output_vocab.gob",
    "data/models/gob_models/moe_classification_model.gob",
    "data/models/gob_models/moe_classification_model_best.gob"
)
foreach ($Model in $OldModels) {
    if (Test-Path $Model) {
        Remove-Item -Path $Model -ErrorAction SilentlyContinue
    }
}

$W2VPath = "data/models/gob_models/word2vec_model.gob"
if (-not (Test-Path $W2VPath)) {
    Write-Host "[WAIT] Word2Vec Dictionary missing. Regenerating from data..." -ForegroundColor Yellow
    go run cmd/tools/train_moe/main.go -train-word2vec
}

# --- 3. Audit & Trend Analysis ---
Write-Host "`r`n[AUDIT] Scanning Gollemer Evolution and Commitment Trends..." -ForegroundColor Cyan
Write-Host ("| {0,-20} | {1,-7} | {2,-12} | {3,-7} |" -f "File", "Steps", "Commitment", "Trend")
Write-Host "----------------------|---------|--------------|---------"

$PrevIQ = 0.0
$BestScore = 0.0
$BestFile = ""
$CheckpointDir = "data/models/checkpoints"

if (Test-Path $CheckpointDir) {
    $Checkpoints = Get-ChildItem "$CheckpointDir/*.gob" | Sort-Object LastWriteTime

    foreach ($f in $Checkpoints) {
        # Run the inspector to export JSON
        go run cmd/inspect/inspect_model.go --export $f.FullName | Out-Null
        $JsonFile = "$($f.FullName).json"
        
        if (Test-Path $JsonFile) {
            try {
                $RawJson = Get-Content $JsonFile -Raw
                $Data = $RawJson | ConvertFrom-Json
                $Steps = $Data.StepCount
                $IQRaw = $Data.Commitment
                $IQ = [math]::Round($IQRaw * 100, 2)

                $Trend = "SAME"
                if ($IQ -gt $PrevIQ) { $Trend = "UP" }
                elseif ($IQ -lt $PrevIQ) { $Trend = "DOWN" }

                if ($IQ -gt $BestScore) {
                    $BestScore = $IQ
                    $BestFile = $f.FullName
                }

                Write-Host ("| {0,-20} | {1,-7} | {2,-12} | {3,-7} |" -f $f.Name, $Steps, "$IQ%", $Trend)
                $PrevIQ = $IQ
                Move-Item -Path $JsonFile -Destination "research_logs/" -Force
            } catch {
                Write-Host ("| {0,-20} | {1,-7} | {2,-12} | {3,-7} |" -f $f.Name, "ERR", "ERR", "ERR") -ForegroundColor Red
                if (Test-Path $JsonFile) { Remove-Item $JsonFile }
            }
        } else {
            Write-Host ("| {0,-20} | {1,-7} | {2,-12} | {3,-7} |" -f $f.Name, "???", "???", "???") -ForegroundColor Gray
        }
    }
}

# --- 4. Disk Pruning ---
if ($BestFile -and $Checkpoints) {
    $LatestFile = $Checkpoints | Select-Object -Last 1
    Write-Host "`n[BEST] Top Performer: $(Split-Path $BestFile -Leaf) ($BestScore%)" -ForegroundColor Green
    foreach ($f in $Checkpoints) {
        if ($f.FullName -ne $BestFile -and $f.FullName -ne $LatestFile.FullName) {
            Remove-Item $f.FullName -ErrorAction SilentlyContinue
        }
    }
}

# --- 5. Promotion ---
$Threshold = 2.0
if ($BestFile) {
    if ($BestScore -ge $Threshold) {
        Write-Host "[PROMOTION] Threshold Met ($BestScore% >= $Threshold%). Promoting..." -ForegroundColor Green
        Copy-Item -Path $BestFile -Destination "data/models/gob_models/moe_classification_model_best.gob" -Force
        Copy-Item -Path "data/models/gob_models/moe_classification_model_best.gob" -Destination "data/models/gob_models/moe_active.gob" -Force
    } else {
        Write-Host "[INFO] Threshold NOT Met ($BestScore%)." -ForegroundColor Yellow
    }
}

# --- 6. Launch Training ---
Write-Host "`n[TRAIN] Starting Gollemer Training (float32)..." -ForegroundColor Cyan

$TrainLog = "logs/training_full.log"
if (Test-Path $TrainLog) { Remove-Item $TrainLog }

$LR = "0.0005"
$MaxGradNorm = "10.0"

# Using 'go run' directly to ensure late fixes are applied without re-compiling the binary manually
go run cmd/tools/train_moe/main.go `
    -train-chat `
    -rebalance `
    -auto-heal `
    -wd 0.01 `
    -lr $LR `
    -max_grad_norm $MaxGradNorm | Tee-Object -FilePath $TrainLog

# --- 7. Stability Audit ---
Write-Host "`n[AUDIT] Scanning training log for stability issues..." -ForegroundColor Blue
if (Test-Path $TrainLog) {
    $Content = Get-Content $TrainLog
    if ($Content -match "NaN|Inf|loss exploded") {
        Write-Host "[ALERT] STABILITY ISSUE DETECTED in $TrainLog" -ForegroundColor Red
        Write-Host "   -> Try lowering --lr or tightening the gradient clip."
    } else {
        Write-Host "[OK] No NaN/Inf detected." -ForegroundColor Green
    }
}

# --- 8. Build WASM & Server ---
Write-Host "`n[BUILD] Finalizing Build Artifacts..." -ForegroundColor Cyan
Write-Host "Compiling WASM Dashboard..."
$Env:GOOS = "js"
$Env:GOARCH = "wasm"
go build -o static/main.wasm ./examples/learningfolder/wasm
$Env:GOOS = "windows"
$Env:GOARCH = "amd64"

Write-Host "Compiling Go Server (SIMD)..."
go build -o "gollemer_server.exe" ./examples/learningfolder

Write-Host "Compiling Main Gollemer binary..."
go build -o "train_moe.exe" ./cmd/tools/train_moe

Write-Host "`n[DONE] System Live at http://localhost:5500" -ForegroundColor Green
