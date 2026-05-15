# llama.cpp + TriAttention + TurboQuant Server
# Qwen3.6-27B

$ErrorActionPreference = "Continue"

# === Config ===
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$SERVER_EXE = Join-Path $SCRIPT_DIR "build\bin\Release\llama-server.exe"
$MODEL_PATH = "C:\Users\linla\Downloads\Qwen3.6-27B-NEO-CODE-HERE-2T-OT-IQ4_XS.gguf"
$MMPROJ_PATH = "C:\Users\linla\Downloads\mmproj-F16.gguf"

# === Port ===
$PORT = 8080

# === GPU Layers (27B IQ4_XS ~14.3GB, RTX 4070 12GB can offload most) ===
$GPU_LAYERS = 30

# === Context ===
$CONTEXT_SIZE = 131072

# === CUDA DLL Path ===
$env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\x64;" + $env:PATH

# === Check files ===
if (-not (Test-Path $SERVER_EXE)) {
    Write-Host "ERROR: $SERVER_EXE not found" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $MODEL_PATH)) {
    Write-Host "ERROR: Model $MODEL_PATH not found" -ForegroundColor Red
    exit 1
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  llama.cpp + TriAttention + TurboQuant" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Model: $(Split-Path $MODEL_PATH -Leaf)" -ForegroundColor Green
Write-Host "GPU Layers: $GPU_LAYERS" -ForegroundColor Green
Write-Host "Context: $CONTEXT_SIZE" -ForegroundColor Green
Write-Host "Port: http://127.0.0.1:$PORT" -ForegroundColor Green
Write-Host ""

# === Clean old process ===
$existing = netstat -ano | Select-String ":$PORT " 
if ($existing) {
    $oldPid = ($existing -split '\s+')[-1]
    taskkill /F /PID $oldPid 2>$null
    Write-Host "Cleaned old process PID $oldPid" -ForegroundColor Yellow
    Start-Sleep 1
}

# === Start ===
$args = @(
    "-m", $MODEL_PATH,
    "--mmproj", $MMPROJ_PATH,
    "--host", "127.0.0.1",
    "--port", $PORT,
    "-ngl", $GPU_LAYERS,
    "-c", $CONTEXT_SIZE,
    "-fa", "on",
    "--metrics",
    "--no-warmup",
    "--cache-ram", "0"
)

Write-Host "Args: $args" -ForegroundColor DarkGray

Start-Process -FilePath $SERVER_EXE -ArgumentList $args -WindowStyle Normal
Write-Host "Server started!" -ForegroundColor Green
