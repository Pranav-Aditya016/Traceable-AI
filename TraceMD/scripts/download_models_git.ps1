param(
    [string]$Token = "",
    [string[]]$Models = @("all")
)

$ErrorActionPreference = "Stop"

function Test-Command([string]$Name) {
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

if (-not (Test-Command "git")) {
    throw "git is required. Install Git first."
}

if (-not (Test-Command "git-lfs")) {
    throw "git-lfs is required. Install Git LFS first."
}

& git lfs install | Out-Null

$projectRoot = Split-Path -Parent $PSScriptRoot
$modelsDir = Join-Path $projectRoot "models"
New-Item -ItemType Directory -Path $modelsDir -Force | Out-Null

$catalog = @{
    "trocr" = @{
        Repo = "microsoft/trocr-large-handwritten"
        Folder = "trocr-large-handwritten"
        Gated = $false
    }
    "sd15" = @{
        Repo = "runwayml/stable-diffusion-v1-5"
        Folder = "stable-diffusion-v1-5"
        Gated = $false
    }
    "paligemma" = @{
        Repo = "google/paligemma2-3b-pt-896"
        Folder = "paligemma2-3b-pt-896"
        Gated = $true
    }
    "medgemma" = @{
        Repo = "google/medgemma-4b-it"
        Folder = "medgemma-4b-it"
        Gated = $true
    }
}

$selected = @()
if ($Models -contains "all") {
    $selected = $catalog.Keys
} else {
    $selected = $Models
}

foreach ($name in $selected) {
    if (-not $catalog.ContainsKey($name)) {
        Write-Warning "Unknown model key '$name'. Valid keys: all, trocr, sd15, paligemma, medgemma"
        continue
    }

    $entry = $catalog[$name]
    $target = Join-Path $modelsDir $entry.Folder

    if (Test-Path $target) {
        Write-Host "[SKIP] $name already exists at $target"
        continue
    }

    $repoUrl = "https://huggingface.co/$($entry.Repo)"

    if ($entry.Gated -and [string]::IsNullOrWhiteSpace($Token)) {
        Write-Warning "[SKIP] $name is gated and requires token access: $($entry.Repo)"
        continue
    }

    Write-Host "[CLONE] $name -> $target"

    if ([string]::IsNullOrWhiteSpace($Token)) {
        & git clone $repoUrl $target
    } else {
        & git -c "http.extraHeader=Authorization: Bearer $Token" clone $repoUrl $target
    }

    if ($LASTEXITCODE -ne 0) {
        if (Test-Path $target) {
            Remove-Item -Recurse -Force $target
        }
        Write-Warning "[FAIL] Could not clone $($entry.Repo)."
    } else {
        Write-Host "[DONE] $name"
    }
}

Write-Host ""
Write-Host "Model download attempt complete."
Write-Host "TraceMD auto-loads local folders under: $modelsDir"
Write-Host "No extra env vars are required if folders exist and are non-empty."
