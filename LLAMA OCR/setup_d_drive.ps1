# Store Ollama models on the D drive (more space) instead of C:.
#
# Ollama reads the OLLAMA_MODELS environment variable when the server starts and
# downloads/keeps all models there. This script sets that variable persistently
# for your user account and creates the folder.
#
# Usage (PowerShell):
#   ./setup_d_drive.ps1                       # uses D:\OllamaModels
#   ./setup_d_drive.ps1 -ModelsPath "D:\AI\Ollama"
#
# After running it, RESTART Ollama (quit it from the tray / `ollama serve`) so it
# picks up the new location, then pull the vision model:
#   ollama pull llama3.2-vision

param(
    [string]$ModelsPath = "D:\OllamaModels",
    [string]$Model = "llama3.2-vision"
)

Write-Host "Configuring Ollama to store models at: $ModelsPath" -ForegroundColor Cyan

# Create the directory if it does not exist.
if (-not (Test-Path $ModelsPath)) {
    New-Item -ItemType Directory -Path $ModelsPath -Force | Out-Null
    Write-Host "Created $ModelsPath"
} else {
    Write-Host "$ModelsPath already exists"
}

# Persist OLLAMA_MODELS for the current user (survives reboots).
[Environment]::SetEnvironmentVariable("OLLAMA_MODELS", $ModelsPath, "User")
# Also set it for the current shell session.
$env:OLLAMA_MODELS = $ModelsPath
Write-Host "Set OLLAMA_MODELS = $ModelsPath (user environment)" -ForegroundColor Green

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Fully quit Ollama (system tray) or stop 'ollama serve'."
Write-Host "  2. Start Ollama again so it reads the new model path."
Write-Host "  3. Pull the vision model (downloads to D: now):"
Write-Host "       ollama pull $Model" -ForegroundColor White
Write-Host ""
Write-Host "Verify with:  ollama list   and check that files land in $ModelsPath"
