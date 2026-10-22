@echo off
echo Starting Gemini Video Watermark Remover...
cd /d "%~dp0"
call .\venv\Scripts\activate
python server.py
pause
