@echo off
cd /d "%~dp0"
echo Membuka Streamlit Dashboard (app.py)...
"..\.venv\Scripts\streamlit.exe" run app.py
pause
