@echo off
setlocal
cd /d "%~dp0"
if not exist data\owner.npy (
  echo Prvo pokreni START.bat i registruj svoje lice tasterom E.
  pause
  exit /b 1
)
for %%D in (0.5 1 2) do (
  echo Postavi lice na %%D m od objektiva. Ostani sam u kadru do naredne poruke.
  pause
  for %%S in (640x480 1280x720 1920x1080) do (
    for %%M in (base exp) do (
      call START.bat --size %%S --mode %%M --distance %%D --warmup 3 --seconds 10
      if errorlevel 1 exit /b 1
    )
  )
)
echo Gotovo. Rezultati su u fascikli results.
pause
