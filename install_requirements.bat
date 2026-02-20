@echo off
chcp 65001 >nul
echo Установка зависимостей ассистента Мэй...
echo.

:: Проверяем, установлен ли Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Python не найден в системе.
    echo Запускаю установщик Python из папки packages...
    start /wait packages\python-3.11.9-amd64.exe
    echo После установки Python закройте это окно и запустите install_requirements.bat снова.
    pause
    exit /b
)

:: Устанавливаем пакеты из локальной папки packages
pip install --no-index --find-links=packages -r requirements.txt

echo.
echo Все зависимости установлены.
echo Теперь вы можете запустить ассистента командой: python main.py
pause