@echo off
chcp 65001 > nul
echo Запуск Assistant + Open-LLM-VTuber...
cd /d "%~dp0Open-LLM-VTuber"
python launcher.py
pause