@echo off
REM run_local_service.bat
REM ======================
REM Double-click this to start the local stand-in for Modules 1, 2 & 3.
REM It sets the two environment variables the service needs, then starts it.
REM Leave this window open while you use the Streamlit app — closing it
REM stops the service.
REM
REM If your folder layout changes (Image-Processing moves, or you get a
REM newer weights file from a teammate), just edit the two "set" lines below.

setlocal

set "PCB_WORKSPACE=C:\Users\leekeezhan\Image-Processing"
set "PCB_WEIGHTS=C:\Users\leekeezhan\Image-Processing\Student3-Defect Detection\models\rtdetr_l_pcb.pt"

REM Always run from the folder this .bat file lives in, no matter where it
REM was double-clicked from.
cd /d "%~dp0"

echo Starting PCB inspection local service...
echo   Folder        : %cd%
echo   PCB_WORKSPACE = %PCB_WORKSPACE%
echo   PCB_WEIGHTS   = %PCB_WEIGHTS%
echo.
echo Once you see "Uvicorn running on http://127.0.0.1:8000", leave this
echo window open and switch to the Streamlit app. Press Ctrl+C here to stop.
echo.

uvicorn local_service.serve:app --host 127.0.0.1 --port 8000

echo.
echo The service stopped. Press any key to close this window.
pause >nul
