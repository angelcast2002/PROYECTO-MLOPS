Param(
    [string]$Python = "python",
    [switch]$UpgradePip
)

$ErrorActionPreference = "Stop"

function Invoke-Step {
    Param(
        [string]$Command,
        [string]$WorkingDirectory = $null
    )
    if ($WorkingDirectory) { Push-Location $WorkingDirectory }
    try {
        Write-Host "→ $Command" -ForegroundColor Cyan
        iex $Command
    }
    finally {
        if ($WorkingDirectory) { Pop-Location }
    }
}

Write-Host "== Proyecto MLOps: instalación de paquetes en modo editable ==" -ForegroundColor Green

try {
    & $Python --version | Out-Null
} catch {
    Write-Error "Python no encontrado. Pasa la ruta con -Python o instala Python 3.8+."
    exit 1
}

# Asegurar pip disponible
$pipOk = $false
& $Python -m pip --version | Out-Null
if ($LASTEXITCODE -eq 0) {
    $pipOk = $true
} else {
    Write-Host "pip no encontrado. Intentando bootstrapping con ensurepip..." -ForegroundColor DarkYellow
    & $Python -m ensurepip --upgrade | Out-Null
    & $Python -m pip --version | Out-Null
    if ($LASTEXITCODE -eq 0) {
        $pipOk = $true
    } else {
        Write-Host "Intentando con 'py -m pip'..." -ForegroundColor DarkYellow
        py -m pip --version | Out-Null
        if ($LASTEXITCODE -eq 0) {
            $Python = "py"
            $pipOk = $true
        } else {
            Write-Error "No se pudo inicializar pip. Instala pip manualmente y reintenta."
            exit 1
        }
    }
}

if ($UpgradePip -and $pipOk) {
    # Evitar problemas de permisos globales: intentar en usuario actual
    Invoke-Step "$Python -m pip install --upgrade --user pip"
}

# Raíz del repo = carpeta padre de este script
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")

$Packages = @(
    "proyecto-core",
    "proyecto-bu",
    "proyecto-du",
    "proyecto-dp",
    "proyecto-modeling",
    "proyecto-eval",
    "proyecto-deploy",
    "proyecto-final"
)

foreach ($pkg in $Packages) {
    $pkgPath = Join-Path $RepoRoot "packages\$pkg"
    if (Test-Path $pkgPath) {
        Write-Host "Instalando $pkg en modo editable..." -ForegroundColor Yellow
        Invoke-Step "$Python -m pip install -e `"$pkgPath`""
    } else {
        Write-Host "(omitido) $pkg no encontrado en $pkgPath" -ForegroundColor DarkYellow
    }
}

# Opcional: instalar paquete monolítico para compatibilidad
if ( (Test-Path (Join-Path $RepoRoot "setup.py")) -or (Test-Path (Join-Path $RepoRoot "pyproject.toml")) ) {
    Write-Host "Instalando paquete monolítico (compatibilidad) en modo editable..." -ForegroundColor Yellow
    Invoke-Step "$Python -m pip install -e `"$RepoRoot`""
}

Write-Host "✔ Instalación completada. Ejecuta 'pytest tests -v' para validar." -ForegroundColor Green
