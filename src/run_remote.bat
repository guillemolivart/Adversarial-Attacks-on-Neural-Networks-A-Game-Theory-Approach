@echo off

setlocal enabledelayedexpansion

:: .env file should contain:
:: PORT=YOUR_PORT_HERE
:: HOST=root@LYOUR_IP_HERE

if exist .env (
    for /f "tokens=*" %%i in (.env) do set %%i
) else (
    echo ERROR: File .env not found
    pause
    exit /b
)

echo 1. Uploading your TFG code...
scp -P %PORT% -o ServerAliveInterval=60 *.py %HOST%:/workspace/

echo 2. Setting up environment and executing training...
ssh -p %PORT% -o ServerAliveInterval=60 %HOST% "mkdir -p /workspace/data /matrices /models && pip install torch torchvision numpy tqdm && [ -f /workspace/data/cifar-10-python.tar.gz ] || wget -O /workspace/data/cifar-10-python.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz && cd /workspace && python3 game.py"

echo 3. Downloading results...
if not exist "..\matrices" mkdir "..\matrices"
if not exist "..\models" mkdir "..\models"
scp -P %PORT% -o ServerAliveInterval=60 %HOST%:/matrices/*.npy "..\matrices"
scp -P %PORT% -o ServerAliveInterval=60 %HOST%:/models/*.pth "..\models"

echo ✅ Process completed!
pause