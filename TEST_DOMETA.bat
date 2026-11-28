@echo off
setlocal
cd /d "%~dp0"
if not exist data\owner.npy (
  echo First run START.bat --developer and enroll your face with E.
  pause
  exit /b 1
)
for %%D in (0.5 1 2) do (
  echo Position your face %%D m from the camera. Stay alone in view until the next prompt.
  pause
  for %%S in (640x480 1280x720 1920x1080) do (
    for %%M in (base exp) do (
      call START.bat --size %%S --mode %%M --distance %%D --warmup 3 --seconds 10
      if errorlevel 1 exit /b 1
    )
  )
)
echo Finished. Results are in the results folder.
pause
