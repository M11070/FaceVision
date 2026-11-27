@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  py -3.11 -m venv .venv
  if errorlevel 1 goto fail
)
".venv\Scripts\python.exe" -c "import sys,cv2,numpy,importlib.metadata as m; assert sys.version_info[:2]==(3,11) and sys.maxsize>2**32 and m.version('opencv-python')=='4.14.0.94' and numpy.__version__=='2.4.6' and m.version('mediapipe')=='0.10.31'" >nul 2>&1
if errorlevel 1 (
  ".venv\Scripts\python.exe" -m pip install --only-binary=:all: --require-hashes -r requirements-gestures.txt
  if errorlevel 1 goto fail
  ".venv\Scripts\python.exe" -m pip check
  if errorlevel 1 goto fail
)
".venv\Scripts\python.exe" -c "import sys; assert sys.version_info[:2]==(3,11) and sys.maxsize>2**32"
if errorlevel 1 goto fail
".venv\Scripts\python.exe" app.py %*
if errorlevel 1 goto fail
exit /b 0
:fail
echo Pokretanje nije uspelo. Proveri poruku iznad.
pause
exit /b 1
