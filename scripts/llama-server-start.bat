@echo off
setlocal

REM llama.cpp-triattention-mtp server startup template.
REM Set MODEL_PATH before running, or edit the defaults below.

set "EXE=%~dp0..\releases\llama-server-triattention-mtp.exe"

if "%MODEL_PATH%"=="" (
  set "MODEL_PATH=C:\Models\Qwen3.6-27B-Q4_K_M.gguf"
)

REM Optional multimodal projector. Leave empty for text-only use.
set "MMPROJ_ARGS="
if not "%MMPROJ_PATH%"=="" (
  set MMPROJ_ARGS=--mmproj "%MMPROJ_PATH%"
)

REM Adjust this if CUDA is installed somewhere else.
if "%CUDA_BIN%"=="" (
  set "CUDA_BIN=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\x64"
)

if exist "%CUDA_BIN%" (
  set "PATH=%CUDA_BIN%;%PATH%"
)

echo ========================================
echo   llama.cpp-triattention-mtp
echo   TriAttention + MTP
echo ========================================
echo.
echo Model: %MODEL_PATH%
echo.

"%EXE%" ^
  -m "%MODEL_PATH%" ^
  %MMPROJ_ARGS% ^
  --host 127.0.0.1 ^
  --port 8080 ^
  -ngl 37 ^
  -c 131072 ^
  --cache-type-k q4_0 ^
  --cache-type-v f16 ^
  --spec-type draft-mtp ^
  --spec-draft-n-max 2 ^
  -fa on ^
  --parallel 1 ^
  -t 12 ^
  -tb 24 ^
  --metrics

echo.
echo Server stopped.
pause
