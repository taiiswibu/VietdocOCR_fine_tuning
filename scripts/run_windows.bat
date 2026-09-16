@echo off
cd /d "%~dp0.."
if not exist .venv\Scripts\python.exe (
  echo Chua co .venv. Hay lam theo README.md de cai dat.
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m vietdoc.cli serve
pause
