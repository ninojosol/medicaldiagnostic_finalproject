@echo off
REM =============================================================================
REM daily_push.bat — Double-click launcher for the safe daily Git sync
REM =============================================================================
REM Always runs from this project root so Explorer double-clicks work.
REM Keeps the window open so you can read success or error messages.
REM =============================================================================

cd /d "%~dp0"

echo.
echo Launching scripts\daily_push.ps1 ...
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\daily_push.ps1"
set "EXITCODE=%ERRORLEVEL%"

echo.
if %EXITCODE% neq 0 (
  echo Sync finished with errors. Exit code: %EXITCODE%
) else (
  echo Sync finished.
)

echo.
pause
exit /b %EXITCODE%
