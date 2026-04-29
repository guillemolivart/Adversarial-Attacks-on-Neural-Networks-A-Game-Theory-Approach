@echo off
setlocal enabledelayedexpansion

if exist .env (
    for /f "tokens=*" %%i in (.env) do set %%i
) else (
    echo ERROR: File .env not found
    pause
    exit /b
)

echo 1. Uploading code AND existing matrices...
:: Pugem el codi
scp -P %PORT% -o ServerAliveInterval=60 *.py %HOST%:/workspace/
:: IMPORTANT: Pugem les matrius que ja tens perquè el Python sàpiga on s'ha quedat
if exist "..\matrices" scp -P %PORT% -o ServerAliveInterval=60 "..\matrices\*.npy" %HOST%:/output/matrices/

echo 2. Setting up environment and executing training...
:: Hem corregit els mkdir perquè coincideixin amb el que vol el Python (/output/...)
ssh -p %PORT% -o ServerAliveInterval=60 %HOST% "mkdir -p /workspace/data /output/matrices /output/models && pip install torch torchvision numpy tqdm && cd /workspace && python3 game.py"

echo 3. Downloading results...
if not exist "..\matrices" mkdir "..\matrices"
if not exist "..\models" mkdir "..\models"

:: ARA SI: Busquem a /output/matrices i /output/models
scp -P %PORT% -o ServerAliveInterval=60 %HOST%:/output/matrices/*.npy "..\matrices"
scp -P %PORT% -o ServerAliveInterval=60 %HOST%:/output/models/*.pth "..\models"

echo ✅ Process completed!
pause