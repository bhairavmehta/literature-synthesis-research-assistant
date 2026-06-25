<#
  One-shot Azure App Service deployment for the LSRA web UI.

  Prerequisites (done once, by YOU, because they need your interactive sign-in):
    1. Azure CLI installed:  winget install --id Microsoft.AzureCLI -e
    2. Sign in to the target account:
         az login                       # opens a browser; sign in as bhairavmehta2026@outlook.com
    3. That account must have an active (e.g. pay-as-you-go or free-trial) subscription.

  Then just run:
    pwsh ./deploy/azure-deploy.ps1
  or with overrides:
    pwsh ./deploy/azure-deploy.ps1 -AppName lsra-bhairav -Location eastus -Sku F1

  The F1 (Free) tier costs nothing. B1 (~$13/mo) is steadier if F1 quota is exhausted.
#>
param(
  [string]$ResourceGroup = "lsra-rg",
  [string]$AppName       = "lsra-$(Get-Random -Minimum 1000 -Maximum 9999)",
  [string]$PlanName      = "lsra-plan",
  [string]$Location      = "eastus",
  [string]$Sku           = "F1"
)

$ErrorActionPreference = "Stop"
$projRoot = Split-Path -Parent $PSScriptRoot   # the project dir containing app.py

Write-Host "== LSRA Azure deploy ==" -ForegroundColor Cyan
Write-Host "  account:" (az account show --query user.name -o tsv)
Write-Host "  subscription:" (az account show --query name -o tsv)
Write-Host "  resource group: $ResourceGroup  app: $AppName  sku: $Sku  loc: $Location"

az group create --name $ResourceGroup --location $Location --output none
Write-Host "[1/4] resource group ready" -ForegroundColor Green

az appservice plan create --name $PlanName --resource-group $ResourceGroup `
  --is-linux --sku $Sku --location $Location --output none
Write-Host "[2/4] linux app service plan ready" -ForegroundColor Green

az webapp create --name $AppName --resource-group $ResourceGroup `
  --plan $PlanName --runtime "PYTHON:3.12" --output none
Write-Host "[3/4] web app created" -ForegroundColor Green

# Oryx runs `pip install -r requirements.txt`; we just point the startup at app.py.
az webapp config set --name $AppName --resource-group $ResourceGroup `
  --startup-file "python app.py" --output none
az webapp config appsettings set --name $AppName --resource-group $ResourceGroup `
  --settings SCM_DO_BUILD_DURING_DEPLOYMENT=1 PYTHONIOENCODING=utf-8 --output none

# Package only what the app needs (exclude tests/eval/zip/caches) and zip-deploy.
$zip = Join-Path $env:TEMP "lsra-deploy.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }
Push-Location $projRoot
try {
  $items = Get-ChildItem -Force | Where-Object {
    $_.Name -notin @('.git','__pycache__','.pytest_cache','deploy','eval','tests') -and
    $_.Extension -ne '.zip'
  }
  Compress-Archive -Path $items.FullName -DestinationPath $zip -Force
} finally { Pop-Location }

az webapp deploy --name $AppName --resource-group $ResourceGroup `
  --src-path $zip --type zip --output none
Write-Host "[4/4] code deployed" -ForegroundColor Green

$url = "https://$AppName.azurewebsites.net"
Write-Host "`nLive at: $url" -ForegroundColor Cyan
Write-Host "(first request may take ~30s while the container cold-starts)"
