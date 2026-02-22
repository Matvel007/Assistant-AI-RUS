#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import time
import signal
import atexit
import socket
import requests
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
ROOT_DIR = BASE_DIR.parent.resolve()
API_SERVER = ROOT_DIR / "api_server.py"
VTUBER_SERVER = BASE_DIR / "run_server.py"
ELECTRON_EXE = BASE_DIR / "frontend" / "release" / "1.2.1" / "win-unpacked" / "open-llm-vtuber-electron.exe"

if not ELECTRON_EXE.exists():
    print(f"⚠️ Electron не найден по пути {ELECTRON_EXE}. Будет открыт браузер.")
    ELECTRON_EXE = None

processes = []
electron_proc = None

def cleanup():
    for proc in processes:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    if electron_proc and electron_proc.poll() is None:
        os.system(f"taskkill /F /PID {electron_proc.pid}")
    print("\nВсе процессы завершены.")

def start_api_server():
    print("🚀 Запуск API-сервера ассистента...")
    proc = subprocess.Popen(
        [sys.executable, str(API_SERVER)],
        cwd=str(ROOT_DIR)
    )
    processes.append(proc)
    return proc

def start_vtuber_server():
    print("🚀 Запуск сервера Open-LLM-VTuber...")
    proc = subprocess.Popen(
        [sys.executable, str(VTUBER_SERVER)],
        cwd=str(BASE_DIR)
    )
    processes.append(proc)
    return proc

def wait_for_port(host, port, timeout=30):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except (socket.timeout, ConnectionRefusedError):
            time.sleep(1)
    return False

def wait_for_api_ready(host='127.0.0.1', port=8001, timeout=60):
    """Ждёт, пока API-сервер ответит на /health."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            r = requests.get(f"http://{host}:{port}/health", timeout=2)
            if r.status_code == 200 and r.json().get("assistant_ready"):
                print(f"✅ API-сервер готов (ассистент загружен)")
                return True
        except:
            pass
        print("⏳ Ожидание готовности API-сервера...")
        time.sleep(2)
    return False

def open_vtuber_client():
    global electron_proc
    # Ждём порт VTuber
    if not wait_for_port('127.0.0.1', 12393, timeout=30):
        print("❌ Сервер VTuber не запустился")
        import webbrowser
        webbrowser.open("http://localhost:12393")
        return

    # Ждём готовность API
    if not wait_for_api_ready(timeout=60):
        print("❌ API-сервер не ответил")
        import webbrowser
        webbrowser.open("http://localhost:12393")
        return

    if ELECTRON_EXE:
        print(f"🌐 Запуск Electron из {ELECTRON_EXE}...")
        try:
            electron_proc = subprocess.Popen(
                [str(ELECTRON_EXE)],
                creationflags=subprocess.CREATE_NEW_CONSOLE,
                shell=True
            )
            processes.append(electron_proc)
            print("✅ Electron-клиент запущен.")
        except Exception as e:
            print(f"⚠️ Ошибка запуска Electron: {e}")
            import webbrowser
            webbrowser.open("http://localhost:12393")
    else:
        print("🌐 Electron не найден, открываю браузер")
        import webbrowser
        webbrowser.open("http://localhost:12393")

def main():
    atexit.register(cleanup)
    signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))

    api_proc = start_api_server()
    vtuber_proc = start_vtuber_server()

    # Даём серверам время на первичный запуск
    time.sleep(3)
    open_vtuber_client()

    try:
        while True:
            time.sleep(1)
            if api_proc.poll() is not None:
                print("❌ API-сервер неожиданно завершился.")
                break
            if vtuber_proc.poll() is not None:
                print("❌ Сервер VTuber неожиданно завершился.")
                break
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()

if __name__ == "__main__":
    main()