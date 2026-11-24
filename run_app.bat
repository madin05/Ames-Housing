@echo off
cd /d "%~dp0"
echo Membuka Streamlit Dashboard Ames Housing (dashboard.py)...
".venv\Scripts\streamlit.exe" run dashboard.py
pause
