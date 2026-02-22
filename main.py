#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Голосовой ассистент с настраиваемым именем.
"""

import os
import sys
import importlib.util
import json
import time
import asyncio
import threading
import queue
import torch
import silero_vad
import re
import onnx_asr
import onnxruntime
import subprocess
import warnings
from pathlib import Path
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import scrolledtext, messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
# ------------------------------------------------------------
#  Глобальные импорты зависимостей
# ------------------------------------------------------------
try:
    import requests
    import yaml
    import chromadb
    from sentence_transformers import SentenceTransformer
    import edge_tts
    import pystray
    from PIL import Image, ImageDraw
    import pyaudio
    import numpy as np
    from openai import OpenAI
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    print("Пожалуйста, установите все зависимости командой:")
    print("  pip install edge-tts chromadb sentence-transformers requests pyyaml pystray pillow tavily-python pyaudio ttkbootstrap openai")
    sys.exit(1)

# Для Silero TTS
try:
    import torch
    import silero
    SILERO_AVAILABLE = True
except ImportError:
    SILERO_AVAILABLE = False
    print("⚠️ Silero TTS не установлен. Для использования установите: pip install silero torch")

# Для преобразования чисел в слова
try:
    from num2words import num2words
    NUM2WORDS_AVAILABLE = True
except ImportError:
    NUM2WORDS_AVAILABLE = False
    print("⚠️ num2words не установлен. Числа могут не озвучиваться. Установите: pip install num2words")

CONFIG_FILE = "config.yaml"

# ------------------------------------------------------------
#  Функции очистки ответов и преобразования чисел
# ------------------------------------------------------------
def clean_response(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub(r'', text)
    text = re.sub(r'(\d+)\s?м/с', r'\1 метров в секунду', text, flags=re.IGNORECASE)
    text = re.sub(r'(\d+)\s?км/ч', r'\1 километров в час', text, flags=re.IGNORECASE)
    text = re.sub(r'(\d+)\s?мм/ч', r'\1 миллиметров в час', text, flags=re.IGNORECASE)
    text = re.sub(r'[*_]{2,}', '', text)
    text = re.sub(r'[*_]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_for_tts(text):
    # Удаление URL
    text = re.sub(r'https?://\S+', '', text)

    # Замена минуса/тире на " минус ", но только если это отдельный символ перед числом
    text = re.sub(r'(?<=\s)[−\-–—](?=\s*\d)', ' минус ', text)
    text = re.sub(r'^[−\-–—](?=\s*\d)', 'минус ', text)

    # Замена символа градуса
    text = re.sub(r'°', ' градусов', text)

    # Замена знаков +, /, <, > на слова
    text = re.sub(r'\+', ' плюс ', text)
    text = re.sub(r'/', ' на ', text)
    text = re.sub(r'<', ' меньше ', text)
    text = re.sub(r'>', ' больше ', text)

    text = re.sub(r'\s+', ' ', text).strip()
    return text

def replace_numbers_with_words(text, lang='ru'):
    if not NUM2WORDS_AVAILABLE:
        return text

    def repl_range(match):
        num1, num2 = match.groups()
        return f"от {num1} до {num2}"

    text = re.sub(r'(\d+)\s*[−\-–—]\s*(\d+)', repl_range, text)

    def repl_number(match):
        num_str = match.group(0)
        normalized = num_str.replace('−', '-').replace('–', '-').replace('—', '-')
        try:
            num = int(normalized)
            if num < 0:
                word = "минус " + num2words(abs(num), lang=lang)
            else:
                word = num2words(num, lang=lang)
            return word
        except ValueError:
            return num_str

    pattern = r'[−\-–—]?\d+'
    text = re.sub(pattern, repl_number, text)
    return text

# ------------------------------------------------------------
#  Класс EdgeTTS
# ------------------------------------------------------------
class EdgeTTSEngine:
    def __init__(self, config):
        self.voice = config.get('voice', 'ru-RU-SvetlanaNeural')
        self.rate = config.get('rate', '+0%')
        self.volume = config.get('volume', '+0%')
        self.use_pygame = importlib.util.find_spec('pygame') is not None
        if self.use_pygame:
            import pygame
            pygame.mixer.init()
            self.pygame = pygame
        import tempfile
        self.temp_dir = tempfile.gettempdir()
        self.current_thread = None

    def speak(self, text, callback=None):
        def _run():
            asyncio.set_event_loop(asyncio.new_event_loop())
            loop = asyncio.get_event_loop()
            try:
                loop.run_until_complete(self._speak_async(text))
            finally:
                loop.close()
            if callback:
                callback()
        self.current_thread = threading.Thread(target=_run, daemon=True)
        self.current_thread.start()

    async def _speak_async(self, text):
        import uuid
        filename = os.path.join(self.temp_dir, f"speech_{uuid.uuid4().hex}.mp3")
        communicate = edge_tts.Communicate(text, self.voice, rate=self.rate, volume=self.volume)
        await communicate.save(filename)

        if self.use_pygame:
            self.pygame.mixer.music.load(filename)
            self.pygame.mixer.music.play()
            while self.pygame.mixer.music.get_busy():
                await asyncio.sleep(0.1)
            self.pygame.mixer.music.unload()
        else:
            if sys.platform == "win32":
                proc = await asyncio.create_subprocess_exec("start", filename, shell=True)
            else:
                proc = await asyncio.create_subprocess_exec("mpg123", filename)
            await proc.wait()

        try:
            os.remove(filename)
        except Exception as e:
            print(f"Не удалось удалить {filename}: {e}")

    def stop(self):
        """Останавливает текущее воспроизведение (если используется pygame)."""
        if self.use_pygame:
            self.pygame.mixer.music.stop()
        else:
            print("⚠️ Остановка внешнего плеера не поддерживается")

# ------------------------------------------------------------
#  Обёртка для pyttsx3
# ------------------------------------------------------------
class Pyttsx3Wrapper:
    def __init__(self, config):
        import pyttsx3
        self.engine = pyttsx3.init()
        voices = self.engine.getProperty('voices')
        if config.get('voice_id') is not None and len(voices) > config['voice_id']:
            self.engine.setProperty('voice', voices[config['voice_id']].id)
        self.engine.setProperty('rate', config.get('pyttsx3_rate', 180))
        self.engine.setProperty('volume', config.get('volume', 1.0))

    def speak(self, text, callback=None):
        self.engine.say(text)
        self.engine.runAndWait()
        if callback:
            callback()

    def stop(self):
        """Останавливает текущее произношение."""
        try:
            self.engine.stop()
        except:
            pass

# ------------------------------------------------------------
#  Класс Silero TTS (исправленная версия)
# ------------------------------------------------------------
class SileroTTSEngine:
    def __init__(self, config):
        self.config = config
        self.language = config.get('silero_language', 'ru')
        self.speaker = config.get('silero_speaker', 'v5_ru')
        self.device = config.get('silero_device', 'cuda')
        self.sample_rate = 48000
        self.model = None
        self.ready = False
        self.max_chars = 800  # немного меньше 1000 для запаса

        if not SILERO_AVAILABLE:
            print("❌ Silero TTS не установлен. Установите: pip install silero torch")
            return

        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self):
        try:
            print(f"🔄 Загрузка Silero TTS (speaker: {self.speaker})...")
            from silero import silero_tts
            if self.device == 'cuda' and torch.cuda.is_available():
                device = torch.device('cuda')
                print("✅ CUDA доступна, загружаем на GPU")
            else:
                device = torch.device('cpu')
                if self.device == 'cuda':
                    print("⚠️ CUDA недоступна, загружаем на CPU")
                else:
                    print("Загружаем на CPU")
            self.model, self.example_text = silero_tts(
                language=self.language,
                speaker=self.speaker,
                device=device
            )
            self.ready = True
            print(f"✅ Silero TTS загружена. Пример текста: {self.example_text}")
        except Exception as e:
            print(f"❌ Ошибка загрузки Silero TTS: {e}")
            self.ready = False

    def _split_text(self, text):
        """Разбивает длинный текст на части по предложениям, не превышая max_chars."""
        sentences = re.split(r'(?<=[.!?…])\s+', text)
        parts = []
        current = ""
        for sent in sentences:
            if len(current) + len(sent) <= self.max_chars:
                current += sent + " "
            else:
                if current:
                    parts.append(current.strip())
                current = sent + " "
        if current:
            parts.append(current.strip())
        # Если предложение длиннее max_chars (редко) – принудительно режем
        final_parts = []
        for p in parts:
            if len(p) <= self.max_chars:
                final_parts.append(p)
            else:
                # режем по символам
                for i in range(0, len(p), self.max_chars):
                    final_parts.append(p[i:i+self.max_chars])
        return final_parts

    def speak(self, text, callback=None):
        if not self.ready:
            print("⏳ Модель Silero ещё не загружена, пропускаю озвучивание.")
            if callback:
                callback()
            return

        parts = self._split_text(text)
        if len(parts) > 1:
            print(f"🔊 Текст слишком длинный, разбит на {len(parts)} частей.")

        def _run():
            for part in parts:
                try:
                    import soundfile as sf
                    import tempfile
                    import torch
                    import numpy as np

                    audio_output = self.model.apply_tts(part)

                    if isinstance(audio_output, torch.Tensor):
                        audio_np = audio_output.cpu().numpy()
                    elif isinstance(audio_output, list):
                        if len(audio_output) == 1 and isinstance(audio_output[0], torch.Tensor):
                            audio_np = audio_output[0].cpu().numpy()
                        elif len(audio_output) == 1 and isinstance(audio_output[0], np.ndarray):
                            audio_np = audio_output[0]
                        else:
                            audio_np = np.array(audio_output)
                    elif isinstance(audio_output, np.ndarray):
                        audio_np = audio_output
                    else:
                        raise TypeError(f"Неизвестный тип аудиовыхода: {type(audio_output)}")

                    audio_np = np.squeeze(audio_np).astype(np.float32)

                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
                        temp_filename = f.name
                    sf.write(temp_filename, audio_np, self.sample_rate)

                    if importlib.util.find_spec('pygame') is not None:
                        import pygame
                        pygame.mixer.init()
                        pygame.mixer.music.load(temp_filename)
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy():
                            time.sleep(0.1)
                        pygame.mixer.music.unload()
                    else:
                        if sys.platform == "win32":
                            os.startfile(temp_filename)
                        else:
                            subprocess.call(["aplay", temp_filename])

                    os.unlink(temp_filename)
                except Exception as e:
                    print(f"❌ Ошибка при озвучивании части Silero TTS: {e}")
            if callback:
                callback()

        threading.Thread(target=_run, daemon=True).start()

# ------------------------------------------------------------
#  Класс векторной памяти (с очисткой старых записей)
# ------------------------------------------------------------
class VectorMemory:
    def __init__(self, persist_directory, collection_name, embed_model_name):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embed_model_name = embed_model_name
        self._embedder = None
        self._client = None
        self._collection = None
        self.lock = threading.Lock()

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self._embedder = SentenceTransformer(
                self.embed_model_name,
                device=device,
                trust_remote_code=True
            )
            print(f"✅ Embedder загружен на {device}")
        return self._embedder

    def _get_client(self):
        if self._client is None:
            import chromadb
            self._client = chromadb.PersistentClient(path=self.persist_directory)
        return self._client

    def _get_collection(self):
        if self._collection is None:
            client = self._get_client()
            try:
                self._collection = client.get_collection(name=self.collection_name)
            except:
                self._collection = client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
        return self._collection

    @property
    def collection(self):
        return self._get_collection()

    def add_message(self, role, text):
        if not text or len(text.strip()) == 0:
            return
        embedder = self._get_embedder()
        embedding = embedder.encode(text).tolist()
        message_id = f"{role}_{datetime.now().timestamp()}"
        collection = self._get_collection()
        with self.lock:
            collection.add(
                embeddings=[embedding],
                documents=[text],
                metadatas=[{"role": role, "timestamp": datetime.now().isoformat()}],
                ids=[message_id]
            )
        preview = text[:50] + "..." if len(text) > 50 else text
        print(f"📝 [Память] Сохранено {role}: {preview}")

    def search_similar(self, query, top_k=5):
        if not query or len(query.strip()) == 0:
            return []
        embedder = self._get_embedder()
        query_emb = embedder.encode(query).tolist()
        collection = self._get_collection()
        try:
            results = collection.query(query_embeddings=[query_emb], n_results=top_k)
            docs = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    docs.append({
                        "text": results['documents'][0][i],
                        "role": results['metadatas'][0][i]['role'] if results['metadatas'] else "unknown"
                    })
            return docs
        except Exception as e:
            print(f"Ошибка при поиске в памяти: {e}")
            return []

    def cleanup_old_memory(self, days=30):
        """Удаляет записи старше days дней, кроме ролей summary и user_profile."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        collection = self._get_collection()
        results = collection.get()
        ids_to_delete = []
        for doc_id, meta in zip(results['ids'], results['metadatas']):
            role = meta.get('role', '')
            timestamp = meta.get('timestamp', '')
            if role not in ['summary', 'user_profile'] and timestamp < cutoff:
                ids_to_delete.append(doc_id)
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
            print(f"🧹 Удалено {len(ids_to_delete)} старых записей")

# ------------------------------------------------------------
#  Класс DeepSeekClient (с поддержкой Tavily)
# ------------------------------------------------------------
class DeepSeekClient:
    def __init__(self, config, tavily_client=None, full_config=None):
        self.tavily_client = tavily_client
        self.provider = config.get('provider', 'official')
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 800)
        self.full_config = full_config or {}

        if self.provider == 'official':
            api_key = config.get('api_key_official')
            base_url = "https://api.deepseek.com/v1"
            if not api_key:
                raise ValueError("Не указан API-ключ для официального DeepSeek")
        elif self.provider == 'aitunnel':
            api_key = config.get('api_key_aitunnel')
            base_url = "https://api.aitunnel.ru/v1/"
            if not api_key:
                raise ValueError("Не указан API-ключ для AITUNNEL")
        else:
            raise ValueError(f"Неизвестный провайдер DeepSeek: {self.provider}")

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.tools = self._get_tools()
        print(f"✅ DeepSeek клиент инициализирован (провайдер: {self.provider})")

    def _get_tools(self):
        return [{
            "type": "function",
            "function": {
                "name": "search_internet",
                "description": "Ищет информацию в интернете по запросу. Используй для новостей, погоды, актуальных данных.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Поисковый запрос, максимально точный и релевантный."
                        }
                    },
                    "required": ["query"]
                }
            }
        }]

    def chat(self, messages, tools=None, tool_choice="auto"):
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response
        except Exception as e:
            print(f"❌ Ошибка при вызове DeepSeek: {e}")
            return None

    def process(self, user_query, system_prompt=None, status_callback=None):
        def update_status(msg):
            if status_callback:
                status_callback(msg)
            print(f"🔧 DeepSeek: {msg}")

        if system_prompt is None:
            # Нейтральный запасной промпт
            system_prompt = (
                "Ты — Ассистент, созданный пользователем. Вы общаетесь на равных, как друзья. "
                "Отвечай кратко, по делу, без лишней информации. Если нужна актуальная информация (новости, погода), "
                "используй инструмент search_internet. Для общих фактов можешь отвечать из своих знаний.\n\n"
                "При использовании результатов поиска для ответа о погоде всегда выбирай данные, которые соответствуют текущему моменту (сегодня, сейчас). Обращай внимание на наличие слов 'сейчас', 'сегодня' в заголовках или тексте. Игнорируй прогнозы на другие дни, если пользователь не просит конкретно о них."
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]

        update_status("Анализирую запрос...")
        response = self.chat(messages, tools=self.tools, tool_choice="auto")
        if response is None:
            return None

        message = response.choices[0].message
        tool_calls = message.tool_calls

        if not tool_calls:
            update_status("Отвечаю из своих знаний.")
            return message.content

        update_status("Принял решение искать в интернете.")
        messages.append(message)

        for tool_call in tool_calls:
            if tool_call.function.name != "search_internet":
                update_status(f"⚠️ Неизвестный инструмент: {tool_call.function.name}")
                continue

            args = json.loads(tool_call.function.arguments)
            query = args.get("query", user_query)
            update_status(f"Формулирую поисковый запрос: '{query}'")

            query_lower = query.lower()
            domains = None
            if "погод" in query_lower or "weather" in query_lower:
                domains = self.full_config.get('weather_sources')
            elif "новост" in query_lower or "news" in query_lower:
                news = self.full_config.get('news_sources', {})
                domains = news.get('default', []) + news.get('international', [])
            elif "стих" in query_lower or "поэт" in query_lower or "poem" in query_lower:
                domains = self.full_config.get('literature_sources')

            if self.tavily_client:
                update_status("Обращаюсь к Tavily...")
                search_result = self._perform_search(query, domains)
                update_status("Получил результаты от Tavily.")
            else:
                search_result = {"error": "Tavily не настроен"}
                update_status("⚠️ Tavily не настроен, поиск невозможен.")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(search_result, ensure_ascii=False)
            })

        update_status("Анализирую результаты поиска и формирую ответ...")
        final_response = self.chat(messages, tools=self.tools, tool_choice="none")
        if final_response is None:
            return None

        update_status("Готово.")
        return final_response.choices[0].message.content

    def _perform_search(self, query, preferred_domains=None):
        if not self.tavily_client:
            return {"error": "Tavily не настроен"}

        try:
            base_params = {
                "query": query,
                "search_depth": "advanced",
                "max_results": 5,
                "include_raw_content": False
            }

            results = []
            if preferred_domains:
                domains = [d for d in preferred_domains if d]
                if domains:
                    params_with_domains = base_params.copy()
                    params_with_domains["include_domains"] = domains
                    params_with_domains["max_results"] = 3
                    print(f"🔍 Поиск с приоритетными доменами: {domains}")
                    response = self.tavily_client.search(**params_with_domains)
                    results = response.get('results', [])
                    if results:
                        print(f"✅ Найдено {len(results)} результатов на приоритетных доменах")

            if not results:
                print("🔍 Обычный поиск (без ограничений по доменам)")
                response = self.tavily_client.search(**base_params)
                results = response.get('results', [])

            formatted_results = []
            for res in results:
                formatted_results.append({
                    'title': res.get('title', ''),
                    'url': res.get('url', ''),
                    'content': res.get('content', '')
                })
            return formatted_results
        except Exception as e:
            print(f"❌ Ошибка при поиске Tavily: {e}")
            return {"error": str(e)}

    def extract_facts(self, text, extraction_prompt):
        messages = [
            {"role": "system", "content": extraction_prompt},
            {"role": "user", "content": text}
        ]
        response = self.chat(messages, tools=None, tool_choice="none")
        if response is None:
            return []
        content = response.choices[0].message.content
        if content.strip().lower().rstrip('.,!?;:') == "нет":
            return []
        facts = [line.strip().lstrip('- ').strip() for line in content.split('\n') if line.strip()]
        return facts

    def summarize(self, dialogue, summary_prompt):
        messages = [
            {"role": "system", "content": summary_prompt},
            {"role": "user", "content": dialogue}
        ]
        response = self.chat(messages, tools=None, tool_choice="none")
        if response is None:
            return None
        return response.choices[0].message.content

#  Класс для записи аудио с микрофона
# ------------------------------------------------------------
class VADProcessor:
    def __init__(self):
        self.vad = VadIterator()
        self.audio_buffer = []
        self.is_speech = False
        self.speech_buffer = bytearray()

    def process_audio(self, audio_bytes):
        # Передаём байты в VAD
        speech_probs = self.vad.process_chunk(audio_bytes)
        if speech_probs > 0.5: # Если это речь
            if not self.is_speech:
                self.is_speech = True
                self.speech_buffer = bytearray()
            self.speech_buffer.extend(audio_bytes)
        else:
            if self.is_speech:
                # Речь закончилась, отправляем буфер на распознавание
                self.is_speech = False
                # Здесь твой вызов self.asr_recognizer.recognize(bytes(self.speech_buffer))
                # и дальше отправка сообщения
                self.speech_buffer = bytearray()

# ------------------------------------------------------------
#  Класс распознавания речи через GigaAM (onnx-asr)
# ------------------------------------------------------------
class GigaAMRecognizer:
    def __init__(self, model_name='gigaam-v3-e2e-ctc', device='cuda'):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.ready = False
        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self):
        try:
            import onnx_asr
            self.model = onnx_asr.load_model(self.model_name)
            self.ready = True
            print(f"✅ GigaAM модель '{self.model_name}' загружена (device: {self.device})")
        except Exception as e:
            print(f"❌ Ошибка загрузки GigaAM: {e}")

    def recognize(self, audio_bytes, sample_rate=16000):
        if not self.ready:
            return None
        import tempfile, os, soundfile as sf, numpy as np
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
            temp_filename = f.name
        try:
            sf.write(temp_filename, audio_np, sample_rate)
            result = self.model.recognize(temp_filename)
            return result.strip() if result else None
        except Exception as e:
            print(f"❌ Ошибка распознавания: {e}")
            return None
        finally:
            os.unlink(temp_filename)

# ------------------------------------------------------------
#  Класс Assistant (основной, с улучшенной RAG) – добавлен api_mode
# ------------------------------------------------------------
class Assistant:
    def __init__(self, config, api_mode=False):
        self.config = config
        self.api_mode = api_mode
        self.memory = VectorMemory(
            config['memory']['vector_store_path'],
            config['memory']['collection_name'],
            config['memory']['embed_model']
        )
        self.memory.cleanup_old_memory(days=30)

        # В API-режиме TTS не создаём
        if not api_mode:
            tts_config = config['tts']
            if tts_config.get('engine') == 'edge':
                self.tts = EdgeTTSEngine(tts_config)
            elif tts_config.get('engine') == 'pyttsx3':
                self.tts = Pyttsx3Wrapper(tts_config)
            elif tts_config.get('engine') == 'silero':
                self.tts = SileroTTSEngine(tts_config)
            else:
                self.tts = None
                print("⚠️ Неизвестный TTS-движок, голосовой вывод отключён.")
        else:
            self.tts = None
            print("ℹ️ API-режим: TTS отключён.")

        self.ignore_phrases = config.get('ignore_phrases', [])
        self.memory_keywords = ["запомни", "запомни, что", "важно:", "запиши", "сохрани"]

        self.min_message_length = config.get('min_message_length', 30)
        self.summary_interval = config.get('summary_interval', 5)
        self.extraction_prompt = config.get('extraction_prompt')
        self.summary_prompt = config.get('summary_prompt')
        self.min_fact_length = config.get('min_fact_length', 10)
        self.sensitive_keywords = config.get('sensitive_keywords', [])
        self.allow_personal_teasing = config.get('allow_personal_teasing', False)

        self.tavily_client = None
        tavily_config = config.get('tools', {}).get('tavily', {})
        if tavily_config.get('enabled') and tavily_config.get('api_key'):
            try:
                from tavily import TavilyClient
                self.tavily_client = TavilyClient(tavily_config['api_key'])
                print("✅ Tavily клиент инициализирован")
            except ImportError:
                print("⚠️ Tavily не установлен. Установите: pip install tavily-python")
            except Exception as e:
                print(f"⚠️ Ошибка инициализации Tavily: {e}")

        self.deepseek_client = None
        deepseek_config = config.get('tools', {}).get('deepseek', {})
        if deepseek_config.get('enabled'):
            try:
                self.deepseek_client = DeepSeekClient(
                    config=deepseek_config,
                    tavily_client=self.tavily_client,
                    full_config=self.config
                )
                print("✅ DeepSeek клиент инициализирован")
            except Exception as e:
                print(f"❌ Ошибка инициализации DeepSeek: {e}")

        self.default_city_enabled = config.get('default_city', {}).get('enabled', False)
        self.default_city_name = config.get('default_city', {}).get('name', '')

        self.status_callback = None
        self.history_callback = None
        self.file_list_queue = queue.Queue()
        self.recent_messages = []
        self.message_counter = 0

        self.last_search_query = None
        self.profile_update_counter = 0

        self._is_speaking = False

        # Подставляем имя ассистента в системный промпт
        self._update_system_prompt_with_name()

    def _update_system_prompt_with_name(self):
        """Заменяет плейсхолдер {assistant_name} в системном промпте на имя из конфига."""
        raw_prompt = self.config['llm']['system_prompt']
        name = self.config.get('system', {}).get('assistant_name', 'Ассистент')
        self.config['llm']['system_prompt'] = raw_prompt.replace('{assistant_name}', name)

    def set_callbacks(self, status_callback=None, history_callback=None):
        self.status_callback = status_callback
        self.history_callback = history_callback

    def update_status(self, text):
        if self.status_callback:
            self.status_callback(text)
        print(f"[Статус] {text}")

    def add_history(self, speaker, text):
        if self.history_callback:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.history_callback(f"[{timestamp}] {speaker}: {text}")
        else:
            print(f"⚠️ history_callback не установлен, сообщение не будет отображено: {speaker}: {text}")

    def warmup(self):
        pass

    def preload_embedder(self):
        self.memory._get_embedder()

    def _is_sensitive(self, fact):
        fact_lower = fact.lower()
        return any(keyword in fact_lower for keyword in self.sensitive_keywords)

    def _extract_facts(self, text):
        if len(text) < self.min_message_length or not self.deepseek_client:
            return []
        facts_text = self.deepseek_client.extract_facts(text, self.extraction_prompt)
        if not facts_text:
            return []
        facts = [line.strip().lstrip('- ').strip() for line in facts_text 
                 if line.strip() and line.strip().lower().rstrip('.,!?;:') != 'нет']
        filtered = []
        for f in facts:
            if len(f) >= self.min_fact_length and not self._is_sensitive(f):
                filtered.append(f)
            else:
                print(f"⛔ Чувствительный или короткий факт пропущен: {f}")
        return filtered

    def _summarize_recent(self):
        if len(self.recent_messages) < 2 or not self.deepseek_client:
            return None
        dialogue = ""
        for role, text, ts in self.recent_messages[-self.summary_interval:]:
            speaker = "Пользователь" if role == "user" else "Ассистент"
            dialogue += f"{speaker}: {text}\n"
        summary = self.deepseek_client.summarize(dialogue, self.summary_prompt)
        return summary if summary else None

    def _process_after_response(self, user_text, assistant_text):
        def task():
            self.recent_messages.append(("user", user_text, time.time()))
            self.recent_messages.append(("assistant", assistant_text, time.time()))
            self.message_counter += 1
            self.profile_update_counter += 1
            if self.message_counter % self.summary_interval == 0:
                summary = self._summarize_recent()
                if summary:
                    self._async_add_message("summary", summary)
                    print(f"📋 [Резюме] Сохранено: {summary}")
            facts = self._extract_facts(user_text)
            for fact in facts:
                self._async_add_message("fact", fact)
                print(f"🧠 [Факт] Сохранено: {fact}")

            if self.profile_update_counter >= 5:
                self.update_user_profile()
                self.profile_update_counter = 0
        threading.Thread(target=task, daemon=True).start()

    def _async_add_message(self, role, text):
        def task():
            self.memory.add_message(role, text)
        threading.Thread(target=task, daemon=True).start()

    def update_user_profile(self):
        if not self.deepseek_client:
            return
        try:
            results = self.memory.collection.get()
            docs = results['documents']
            metas = results['metadatas']
            if not docs:
                return
            context = "Последние сообщения:\n"
            count = 0
            for doc, meta in zip(reversed(docs), reversed(metas)):
                if meta.get('role') in ['user', 'assistant']:
                    role = "Пользователь" if meta['role'] == 'user' else "Ассистент"
                    context += f"{role}: {doc}\n"
                    count += 1
                    if count >= 20:
                        break
            facts = [doc for doc, meta in zip(docs, metas) if meta.get('role') == 'fact']
            if facts:
                context += "\nСохранённые факты:\n" + "\n".join(f"- {f}" for f in facts[-10:])
            current_profile = None
            for doc, meta in zip(docs, metas):
                if meta.get('role') == 'user_profile':
                    current_profile = doc
                    break
            if current_profile:
                context += f"\nТекущий профиль пользователя:\n{current_profile}"
            prompt = (
                "На основе следующих сообщений составь краткий профиль пользователя. "
                "Укажи имя (если известно), возраст, город, интересы, предпочтения. "
                "Ответь одним-двумя предложениями.\n\n"
                f"{context}"
            )
            new_profile = self.deepseek_client.summarize(prompt, "Ты — система анализа личности.")
            if new_profile and new_profile.strip():
                old = self.memory.collection.get(where={"role": "user_profile"})
                if old['ids']:
                    self.memory.collection.delete(ids=old['ids'])
                self.memory.add_message("user_profile", new_profile.strip())
                print(f"👤 Профиль пользователя обновлён: {new_profile[:100]}...")
        except Exception as e:
            print(f"❌ Ошибка при обновлении профиля: {e}")

    def _refine_search_query(self, user_text, last_query):
        if not last_query:
            return user_text
        if any(word in user_text.lower() for word in ['ещё', 'другой', 'другое', 'еще']):
            prompt = (
                f"Предыдущий поисковый запрос: '{last_query}'. "
                f"Пользователь сказал: '{user_text}'. Сформулируй новый поисковый запрос, который будет "
                f"продолжением или вариацией предыдущего. Например, если просят «ещё», найди другую тему, "
                f"если «другой» — измени ключевые слова. Ответь только строкой запроса."
            )
            try:
                response = self.deepseek_client.chat(
                    messages=[{"role": "user", "content": prompt}],
                    tools=None,
                    tool_choice="none"
                )
                if response and response.choices:
                    new_query = response.choices[0].message.content.strip()
                    if new_query:
                        return new_query
            except:
                pass
            return f"{last_query} другое"
        return user_text

    def process_input(self, user_text):
        if not user_text.strip():
            return "Пожалуйста, скажите что-нибудь.", 0

        self.add_history("Пользователь", user_text)
        self.update_status("Думаю...")

        start = time.time()

        if self.last_search_query and any(word in user_text.lower() for word in ['ещё', 'другой', 'другое', 'еще']):
            refined_query = self._refine_search_query(user_text, self.last_search_query)
            self.last_search_query = refined_query
            context = f"Предыдущий поисковый запрос: '{self.last_search_query}'. Пользователь хочет продолжить: {user_text}. "
            full_prompt = context + user_text
        else:
            full_prompt = user_text

        similar = self.memory.search_similar(user_text, top_k=self.config['memory']['top_k'])
        memory_context = ""
        if similar:
            memory_context = "Ранее обсуждали:\n" + "\n".join(
                f"{'Пользователь' if item['role']=='user' else 'Ассистент'}: {item['text']}"
                for item in similar
            ) + "\nУчитывай это.\n"

        profile_text = ""
        try:
            profile_result = self.memory.collection.get(where={"role": "user_profile"})
            if profile_result['documents']:
                profile_text = f"Профиль пользователя: {profile_result['documents'][0]}\n"
        except:
            pass

        final_prompt = user_text
        if memory_context or profile_text:
            final_prompt = (profile_text + memory_context + "Теперь ответь на запрос:\n" + user_text).strip()

        system_prompt = self.config['llm']['system_prompt']

        if self.deepseek_client is None:
            self.update_status("Ошибка: DeepSeek не настроен.")
            return "DeepSeek не настроен. Пожалуйста, добавьте API-ключ в настройках.", 0

        def status_cb(msg):
            self.update_status(msg)

        deepseek_response = self.deepseek_client.process(
            final_prompt,
            system_prompt=system_prompt,
            status_callback=status_cb
        )

        elapsed = time.time() - start

        if deepseek_response is None:
            error_msg = "Не удалось получить ответ от DeepSeek."
            self.add_history("Ассистент", error_msg)
            self.update_status("Готов к работе")
            return error_msg, elapsed

        final_response_clean = clean_response(deepseek_response)

        self.add_history("Ассистент", final_response_clean)

        final_response_with_words = replace_numbers_with_words(final_response_clean, lang='ru')
        final_response_tts = clean_for_tts(final_response_with_words)

        print(f"🤖 {self.config.get('system', {}).get('assistant_name', 'Ассистент')}: {final_response_clean} ({elapsed:.2f} сек)")
        print(f"🔊 Текст для TTS (после обработки): {final_response_tts}")

        # В API-режиме пропускаем озвучивание
        if not self.api_mode and final_response_tts and self.tts and not self._is_speaking:
            self._is_speaking = True
            self.tts.speak(final_response_tts, callback=self._on_speak_finished)

        self.update_status("Готов к работе")

        # Фоновая обработка (память)
        self._process_after_response(user_text, final_response_clean)

        return final_response_clean, elapsed

    def _on_speak_finished(self):
        self._is_speaking = False
        # Микрофон не трогаем – он управляется кнопкой в GUI

    def stop_speaking(self):
        if self.tts and hasattr(self.tts, 'stop'):
            self.tts.stop()
        self._is_speaking = False
#  Главный GUI с отложенной загрузкой – финальная версия с silero-vad
# ------------------------------------------------------------
#  Простой аудио-рекордер с чанками по 32 мс (512 сэмплов для 16 кГц)
# ------------------------------------------------------------
class SimpleAudioRecorder:
    def __init__(self, sample_rate=16000, chunk_duration_ms=32):
        self.sample_rate = sample_rate
        self.chunk_bytes = int(sample_rate * 2 * chunk_duration_ms / 1000)  # 16-bit
        self.is_recording = False
        self.p = pyaudio.PyAudio()
        self.stream = None

    def start(self):
        if self.is_recording:
            return
        self.is_recording = True
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_bytes,
            stream_callback=None
        )
        print("🎤 Запись начата...")

    def read_chunk(self):
        if not self.is_recording:
            return None
        try:
            data = self.stream.read(self.chunk_bytes, exception_on_overflow=False)
            return data
        except Exception as e:
            print(f"❌ Ошибка чтения аудио: {e}")
            return None

    def stop(self):
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        self.p.terminate()
        print("🎤 Запись остановлена")


# ------------------------------------------------------------
#  Главный GUI с отложенной загрузкой – финальная версия с silero-vad
# ------------------------------------------------------------
# ------------------------------------------------------------
#  Простой аудио-рекордер с чанками по 32 мс (512 сэмплов для 16 кГц)
# ------------------------------------------------------------
class SimpleAudioRecorder:
    def __init__(self, sample_rate=16000, chunk_duration_ms=32):
        self.sample_rate = sample_rate
        self.chunk_bytes = int(sample_rate * 2 * chunk_duration_ms / 1000)  # 16-bit
        self.is_recording = False
        self.p = pyaudio.PyAudio()
        self.stream = None

    def start(self):
        if self.is_recording:
            return
        self.is_recording = True
        try:
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_bytes,
                stream_callback=None
            )
            print("🎤 Запись начата...")
        except Exception as e:
            print(f"❌ Ошибка открытия микрофона: {e}")
            self.is_recording = False
            raise

    def read_chunk(self):
        if not self.is_recording:
            return None
        try:
            data = self.stream.read(self.chunk_bytes, exception_on_overflow=False)
            return data
        except Exception as e:
            print(f"❌ Ошибка чтения аудио: {e}")
            return None

    def stop(self):
        self.is_recording = False
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
            self.stream = None
        print("🎤 Запись остановлена")

    def terminate(self):
        self.stop()
        self.p.terminate()


# ------------------------------------------------------------
#  Главный GUI с отложенной загрузкой – финальная версия с silero-vad
# ------------------------------------------------------------
class AssistantGUI:
    def __init__(self, assistant):
        self.assistant = assistant
        self.ready = False
        self.init_thread = None

        # Получаем имя ассистента для заголовка
        assistant_name = assistant.config.get('system', {}).get('assistant_name', 'Ассистент')
        self.root = ttk.Window(title=f"Голосовой ассистент {assistant_name}", themename="darkly")
        self.root.geometry("1000x700")
        self.root.minsize(800, 400)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.minimize_to_tray = assistant.config.get('system', {}).get('minimize_to_tray', False)
        self.tray_icon = None
        self.create_tray_icon()

        top_frame = ttk.Frame(self.root)
        top_frame.pack(padx=10, pady=5, fill=tk.X)

        ttk.Button(top_frame, text="📚 Память", command=self.open_memory, bootstyle="info").pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="⚙️ Настройки", command=self.open_settings, bootstyle="secondary").pack(side=tk.LEFT, padx=5)

        self.mic_button = ttk.Button(top_frame, text="🎤 Вкл", bootstyle="success", width=10, command=self.toggle_microphone)
        self.mic_button.pack(side=tk.RIGHT, padx=10)

        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(top_frame, textvariable=self.search_var, width=25)
        self.search_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.search_entry.insert(0, "🔍 Поиск...")
        self.search_var.trace_add("write", lambda *args: self._filter_history())
        self.search_entry.bind("<FocusIn>", lambda e: self.search_entry.delete(0, tk.END) if self.search_entry.get() == "🔍 Поиск..." else None)
        self.search_entry.bind("<FocusOut>", lambda e: self.search_entry.insert(0, "🔍 Поиск...") if not self.search_entry.get() else None)

        self.clear_search_btn = ttk.Button(top_frame, text="✖", command=self._clear_search, width=3, bootstyle="secondary")
        self.clear_search_btn.pack(side=tk.LEFT, padx=5)

        self.chat_frame = ttk.Frame(self.root)
        self.chat_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.history = scrolledtext.ScrolledText(self.chat_frame, state='disabled', height=20, wrap=tk.WORD)
        self.history.pack(fill=tk.BOTH, expand=True)
        self.all_history_lines = []

        self.loading_frame = ttk.Frame(self.chat_frame)
        self.loading_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        self.progress = ttk.Progressbar(self.loading_frame, mode='indeterminate', length=200)
        self.progress.pack(pady=10)
        self.loading_label = ttk.Label(self.loading_frame, text="Загрузка компонентов...", font=('Arial', 10))
        self.loading_label.pack()
        self.progress.start()

        self.context_menu = tk.Menu(self.history, tearoff=0)
        self.context_menu.add_command(label="Копировать", command=self.copy_text)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Выделить всё", command=self.select_all)
        self.history.bind("<Button-3>", self.show_context_menu)

        bottom_frame = ttk.Frame(self.root)
        bottom_frame.pack(padx=10, pady=(0,10), fill=tk.X)

        self.send_button = ttk.Button(bottom_frame, text="Отправить", command=self.send_message, bootstyle="primary")
        self.send_button.pack(side=tk.RIGHT, padx=(5,0))
        self.send_button.config(state='disabled')

        self.input_field = scrolledtext.ScrolledText(
            bottom_frame,
            wrap=tk.WORD,
            height=1,
            font=('Arial', 10),
            relief=tk.SUNKEN,
            borderwidth=2
        )
        self.input_field.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,5))
        self.input_field.bind("<Return>", self._on_enter_pressed)
        self.input_field.bind("<Shift-Return>", self._on_shift_enter)
        self.input_field.bind("<KeyRelease>", self._adjust_input_height)
        self.input_field.config(state='disabled')

        self.input_context_menu = tk.Menu(self.input_field, tearoff=0)
        self.input_context_menu.add_command(label="Копировать", command=lambda: self.input_field.event_generate("<<Copy>>"))
        self.input_context_menu.add_command(label="Вставить", command=lambda: self.input_field.event_generate("<<Paste>>"))
        self.input_context_menu.add_command(label="Вырезать", command=lambda: self.input_field.event_generate("<<Cut>>"))
        self.input_context_menu.add_separator()
        self.input_context_menu.add_command(label="Выделить всё", command=self._select_all_in_input)
        self.input_field.bind("<Button-3>", self._show_input_context_menu)

        self.status_var = tk.StringVar()
        self.status_var.set("Загрузка...")
        ttk.Label(self.root, textvariable=self.status_var, bootstyle="inverse-secondary", anchor=tk.W).pack(fill=tk.X)

        self.queue = queue.Queue()
        self.process_queue()

        self.assistant.set_callbacks(
            status_callback=lambda s: self.queue.put(("status", s)),
            history_callback=lambda t: self.queue.put(("history", t))
        )

        # ---------- Инициализация аудио и VAD ----------
        asr_config = self.assistant.config.get('asr', {})
        if asr_config.get('enabled', True):
            model_name = asr_config.get('model_name', 'gigaam-v3-e2e-ctc')
            device = asr_config.get('device', 'cuda')
            self.asr_recognizer = GigaAMRecognizer(model_name=model_name, device=device)

            import torch
            import numpy as np
            self.vad_model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                trust_repo=True,
                onnx=False
            )
            self.audio_recorder = SimpleAudioRecorder()
            self.is_recording = False
            self.speech_buffer = bytearray()
            self.sample_rate = 16000
        else:
            self.asr_recognizer = None
            self.vad_model = None
            self.audio_recorder = None
            self.is_recording = False

        self.init_thread = threading.Thread(target=self._background_init, daemon=True)
        self.init_thread.start()

    def _background_init(self):
        try:
            self._update_status("Загрузка модели эмбеддингов...")
            self.assistant.preload_embedder()
            self._update_status("Готов к работе")
            self.root.after(0, self._set_ready)
        except Exception as e:
            self._update_status(f"Ошибка инициализации: {e}")
            print(f"❌ Ошибка в фоновой инициализации: {e}")

    def _update_status(self, text):
        self.root.after(0, lambda: self.loading_label.config(text=text))

    def _set_ready(self):
        self.ready = True
        self.loading_frame.place_forget()
        self.progress.stop()
        self.input_field.config(state='normal')
        self.send_button.config(state='normal')
        self.status_var.set("Готов к работе")
        self.input_field.focus_set()

    def toggle_microphone(self):
        if not self.ready:
            return
        if not self.asr_recognizer or not self.audio_recorder:
            self.status_var.set("ASR не настроен")
            return

        if not self.is_recording:
            try:
                self.audio_recorder.start()
            except Exception as e:
                self.status_var.set("Ошибка микрофона")
                return
            self.is_recording = True
            self.mic_button.config(text="⏹️ Стоп", bootstyle="danger")
            self.status_var.set("Слушаю... (говорите)")
            threading.Thread(target=self._vad_loop, daemon=True).start()
        else:
            self._stop_mic()

    def _stop_mic(self):
        if self.is_recording:
            self.audio_recorder.stop()
            self.is_recording = False
            self.mic_button.config(text="🎤 Вкл", bootstyle="success")
            self.status_var.set("Готов к работе")
            self.speech_buffer = bytearray()

    def _vad_loop(self):
        silence_threshold_sec = 1.0
        frame_duration_ms = 32
        required_silence_frames = int(silence_threshold_sec / (frame_duration_ms / 1000.0))
        speech_active = False
        silence_frames = 0

        while self.is_recording:
            try:
                audio_chunk = self.audio_recorder.read_chunk()
                if audio_chunk is None:
                    break

                audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16)  # всегда 512 сэмплов
                audio_float32 = audio_int16.astype(np.float32) / 32768.0
                audio_tensor = torch.from_numpy(audio_float32)

                with torch.no_grad():
                    speech_prob = self.vad_model(audio_tensor, self.sample_rate).item()

                if speech_prob > 0.5:
                    if not speech_active:
                        speech_active = True
                        self.speech_buffer = bytearray()
                    self.speech_buffer.extend(audio_chunk)
                    silence_frames = 0
                else:
                    if speech_active:
                        self.speech_buffer.extend(audio_chunk)
                        silence_frames += 1
                        if silence_frames >= required_silence_frames:
                            audio_data = bytes(self.speech_buffer)
                            self.root.after(0, lambda: self._process_vad_recording(audio_data))
                            self.root.after(0, self._stop_mic)
                            break
                    else:
                        pass
                time.sleep(0.01)
            except Exception as e:
                print(f"❌ Ошибка в VAD цикле: {e}")
                self.root.after(0, self._stop_mic)
                break

    def _process_vad_recording(self, audio_bytes):
        if len(audio_bytes) > 8000:  # примерно 0.5 сек
            text = self.asr_recognizer.recognize(audio_bytes)
            if text:
                threading.Thread(target=self._process_message, args=(text,), daemon=True).start()
            else:
                self.status_var.set("Не удалось распознать")
        else:
            self.status_var.set("Слишком короткая запись")

    # ------------------------------------------------------------
    #  Отправка сообщений
    # ------------------------------------------------------------
    def send_message(self, event=None):
        if not self.ready:
            return
        user_text = self.input_field.get("1.0", "end-1c").strip()
        if not user_text:
            return
        self.input_field.delete("1.0", tk.END)
        self._adjust_input_height()
        self.status_var.set("Думаю...")
        self.assistant.stop_speaking()
        threading.Thread(target=self._process_message, args=(user_text,), daemon=True).start()

    def _process_message(self, user_text):
        response, elapsed = self.assistant.process_input(user_text)
        if not self.assistant.file_list_queue.empty():
            self.queue.put(("file_list", self.assistant.file_list_queue.get()))
        self.queue.put(("response", response, elapsed))

    def process_queue(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                if msg[0] == "history":
                    self._add_to_history(msg[1])
                elif msg[0] == "file_list":
                    self._add_to_history(msg[1])
                elif msg[0] == "status":
                    self.status_var.set(msg[1])
                elif msg[0] == "response":
                    response, elapsed = msg[1], msg[2]
                    self._add_to_history(f"      ⏱ {elapsed:.2f} сек")
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_queue)

    def _add_to_history(self, text):
        self.all_history_lines.append(text)
        self._filter_history()

    # ------------------------------------------------------------
    #  Вспомогательные методы (без изменений)
    # ------------------------------------------------------------
    def _on_enter_pressed(self, event):
        self.send_message()
        return "break"

    def _on_shift_enter(self, event):
        self.input_field.insert(tk.INSERT, "\n")
        self._adjust_input_height()
        return "break"

    def _adjust_input_height(self, event=None):
        content = self.input_field.get("1.0", "end-1c")
        lines = content.count("\n") + 1
        new_height = min(max(lines, 1), 5)
        self.input_field.config(height=new_height)

    def _show_input_context_menu(self, event):
        self.input_context_menu.tk_popup(event.x_root, event.y_root)
        return "break"

    def _select_all_in_input(self):
        self.input_field.tag_add(tk.SEL, "1.0", tk.END)
        self.input_field.mark_set(tk.INSERT, "1.0")
        self.input_field.see(tk.INSERT)
        return "break"

    def _clear_search(self):
        self.search_var.set("")
        self.search_entry.delete(0, tk.END)
        self.search_entry.insert(0, "🔍 Поиск...")
        self._filter_history()

    def _filter_history(self, event=None):
        if not hasattr(self, 'all_history_lines'):
            return
        search_text = self.search_var.get().strip().lower()
        self.history.config(state='normal')
        self.history.delete("1.0", tk.END)
        if not search_text or search_text == "🔍 поиск...":
            for line in self.all_history_lines:
                self.history.insert(tk.END, line + "\n")
        else:
            for line in self.all_history_lines:
                if search_text in line.lower():
                    self.history.insert(tk.END, line + "\n")
        self.history.config(state='disabled')
        self.history.see(tk.END)

    def create_tray_icon(self):
        image = Image.new('RGB', (64, 64), color='blue')
        draw = ImageDraw.Draw(image)
        name = self.assistant.config.get('system', {}).get('assistant_name', 'Ассистент')
        first_letter = name[0].upper() if name else 'А'
        draw.text((20, 10), first_letter, fill='white')
        menu = pystray.Menu(
            pystray.MenuItem("Показать", self.show_window),
            pystray.MenuItem("Выход", self.quit_app)
        )
        self.tray_icon = pystray.Icon("assistant", image, name, menu)

    def show_window(self, icon=None, item=None):
        if self.tray_icon:
            self.tray_icon.stop()
            self.tray_icon = None
        self.root.deiconify()
        self.root.lift()

    def hide_window(self):
        self.root.withdraw()
        if self.tray_icon is None:
            self.create_tray_icon()
        threading.Thread(target=self.tray_icon.run, daemon=True).start()

    def on_closing(self):
        if self.minimize_to_tray:
            self.hide_window()
        else:
            self.quit_app()

    def quit_app(self, icon=None, item=None):
        if self.tray_icon:
            self.tray_icon.stop()
        self.root.quit()
        self.root.destroy()

    def copy_text(self):
        try:
            self.history.clipboard_clear()
            self.history.clipboard_append(self.history.selection_get())
        except:
            pass

    def select_all(self):
        self.history.tag_add(tk.SEL, "1.0", tk.END)
        self.history.mark_set(tk.INSERT, "1.0")

    def show_context_menu(self, event):
        self.context_menu.post(event.x_root, event.y_root)

    def open_settings(self):
        SettingsWindow(self.root, self.assistant.config, self.assistant)

    def open_memory(self):
        MemoryWindow(self.root, self.assistant)
# ------------------------------------------------------------
#  Простой аудио-рекордер с чанками по 32 мс (512 сэмплов для 16 кГц)
# ------------------------------------------------------------
class SimpleAudioRecorder:
    def __init__(self, sample_rate=16000, chunk_duration_ms=32):
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * chunk_duration_ms / 1000)  # 512 сэмплов
        self.is_recording = False
        self.p = pyaudio.PyAudio()
        self.stream = None

    def start(self):
        if self.is_recording:
            return
        self.is_recording = True
        try:
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.frame_size,  # важно: количество фреймов
                stream_callback=None
            )
            print("🎤 Запись начата...")
        except Exception as e:
            print(f"❌ Ошибка открытия микрофона: {e}")
            self.is_recording = False
            raise

    def read_chunk(self):
        if not self.is_recording:
            return None
        try:
            data = self.stream.read(self.frame_size)  # читаем ровно 512 фреймов
            return data
        except Exception as e:
            print(f"❌ Ошибка чтения аудио: {e}")
            return None

    def stop(self):
        self.is_recording = False
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
            self.stream = None
        print("🎤 Запись остановлена")

    def terminate(self):
        self.stop()
        self.p.terminate()

# ------------------------------------------------------------
#  Окно настроек (ttkbootstrap) – без изменений
# ------------------------------------------------------------
class SettingsWindow(ttk.Toplevel):
    def __init__(self, parent, config, assistant):
        super().__init__(parent)
        self.config = config
        self.assistant = assistant
        self.title("Настройки ассистента")
        self.geometry("950x700")
        self.minsize(950, 700)
        self.transient(parent)
        self.grab_set()

        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self._create_rag_tab(notebook)
        self._create_tts_tab(notebook)
        self._create_prompt_tab(notebook)
        self._create_tools_tab(notebook)
        self._create_system_tab(notebook)

    def _create_rag_tab(self, notebook):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="RAG / Память")

        row = 0
        ttk.Label(frame, text="Количество похожих сообщений (top_k):").grid(row=row, column=0, sticky="w", pady=5, padx=10)
        self.top_k_var = tk.IntVar(value=self.config['memory']['top_k'])
        ttk.Entry(frame, textvariable=self.top_k_var, width=5).grid(row=row, column=1, sticky="w", pady=5)
        row += 1

        ttk.Label(frame, text="Модель эмбеддингов:").grid(row=row, column=0, sticky="w", pady=5, padx=10)
        self.embed_model_var = tk.StringVar(value=self.config['memory']['embed_model'])
        ttk.Entry(frame, textvariable=self.embed_model_var, width=40).grid(row=row, column=1, sticky="w", pady=5)
        row += 1

        ttk.Button(frame, text="Применить", command=self.apply_rag, bootstyle="primary").grid(row=row, column=0, columnspan=2, pady=20)

    def _create_tts_tab(self, notebook):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Голос")

        row = 0
        ttk.Label(frame, text="Движок TTS:").grid(row=row, column=0, sticky="w", pady=5, padx=10)
        self.tts_engine_var = tk.StringVar(value=self.config['tts']['engine'])
        ttk.Combobox(frame, textvariable=self.tts_engine_var, values=["edge", "pyttsx3", "silero"], width=20).grid(row=row, column=1, sticky="w", pady=5)
        row += 1

        self.edge_frame = ttk.Frame(frame)
        self.edge_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=5)
        ttk.Label(self.edge_frame, text="Голос Edge:").grid(row=0, column=0, sticky="w", padx=10)
        self.edge_voice_var = tk.StringVar(value=self.config['tts']['voice'])
        ttk.Entry(self.edge_frame, textvariable=self.edge_voice_var, width=30).grid(row=0, column=1, sticky="w", padx=5)
        row += 1

        self.pyttsx3_frame = ttk.Frame(frame)
        self.pyttsx3_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=5)
        ttk.Label(self.pyttsx3_frame, text="ID голоса pyttsx3:").grid(row=0, column=0, sticky="w", padx=10)
        self.pyttsx3_voice_id_var = tk.IntVar(value=self.config['tts']['voice_id'])
        ttk.Entry(self.pyttsx3_frame, textvariable=self.pyttsx3_voice_id_var, width=10).grid(row=0, column=1, sticky="w", padx=5)
        row += 1

        self.silero_frame = ttk.Frame(frame)
        self.silero_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=5)
        ttk.Label(self.silero_frame, text="Язык Silero:").grid(row=0, column=0, sticky="w", padx=10)
        self.silero_language_var = tk.StringVar(value=self.config['tts'].get('silero_language', 'ru'))
        ttk.Combobox(self.silero_frame, textvariable=self.silero_language_var,
                     values=["ru", "en", "de", "es", "fr", "uk", "uz", "cyrillic"],
                     width=20).grid(row=0, column=1, sticky="w", padx=5)
        row += 1

        ttk.Label(self.silero_frame, text="Голос Silero (speaker):").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        self.silero_speaker_var = tk.StringVar(value=self.config['tts'].get('silero_speaker', 'v5_ru'))
        ttk.Combobox(self.silero_frame, textvariable=self.silero_speaker_var,
                     values=["v5_ru", "v4_ru", "aidar", "baya", "kseniya", "irina", "natasha", "ruslan"],
                     width=20).grid(row=1, column=1, sticky="w", padx=5)
        row += 1

        ttk.Label(self.silero_frame, text="Устройство:").grid(row=2, column=0, sticky="w", padx=10, pady=5)
        self.silero_device_var = tk.StringVar(value=self.config['tts'].get('silero_device', 'cuda'))
        ttk.Combobox(self.silero_frame, textvariable=self.silero_device_var, values=["cpu", "cuda"], width=10).grid(row=2, column=1, sticky="w", padx=5)
        row += 1

        ttk.Button(frame, text="Применить", command=self.apply_tts, bootstyle="primary").grid(row=row, column=0, columnspan=2, pady=20)

    def _create_prompt_tab(self, notebook):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Системный промпт")

        self.prompt_text = scrolledtext.ScrolledText(frame, wrap=tk.WORD, height=25)
        self.prompt_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.prompt_text.insert("1.0", self.config['llm']['system_prompt'])

        ttk.Button(frame, text="Применить", command=self.apply_prompt, bootstyle="primary").pack(pady=10)

    def _create_tools_tab(self, notebook):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Инструменты")

        row = 0
        ttk.Label(frame, text="Tavily API (поиск в интернете)", font=('', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky="w", pady=(10,0), padx=10)
        row += 1

        self.tavily_enabled_var = tk.BooleanVar(value=self.config.get('tools', {}).get('tavily', {}).get('enabled', False))
        ttk.Checkbutton(frame, text="Включить Tavily", variable=self.tavily_enabled_var, bootstyle="round-toggle").grid(row=row, column=0, sticky="w", pady=5, padx=10)
        row += 1

        ttk.Label(frame, text="API-ключ Tavily:").grid(row=row, column=0, sticky="w", pady=5, padx=10)
        self.tavily_api_key_var = tk.StringVar(value=self.config.get('tools', {}).get('tavily', {}).get('api_key', ''))
        ttk.Entry(frame, textvariable=self.tavily_api_key_var, width=60, show="*").grid(row=row, column=1, sticky="w", padx=5)
        row += 1

        ttk.Label(frame, text="DeepSeek API (основная модель)", font=('', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky="w", pady=(20,0), padx=10)
        row += 1

        self.deepseek_enabled_var = tk.BooleanVar(value=self.config.get('tools', {}).get('deepseek', {}).get('enabled', False))
        ttk.Checkbutton(frame, text="Включить DeepSeek", variable=self.deepseek_enabled_var, bootstyle="round-toggle").grid(row=row, column=0, sticky="w", pady=5, padx=10)
        row += 1

        ttk.Label(frame, text="Провайдер:").grid(row=row, column=0, sticky="w", pady=5, padx=10)
        self.deepseek_provider_var = tk.StringVar(value=self.config.get('tools', {}).get('deepseek', {}).get('provider', 'official'))
        provider_combo = ttk.Combobox(frame, textvariable=self.deepseek_provider_var, values=["official", "aitunnel"], width=20)
        provider_combo.grid(row=row, column=1, sticky="w", padx=5)
        row += 1

        ttk.Label(frame, text="Ключ DeepSeek (official):").grid(row=row, column=0, sticky="w", pady=5, padx=10)
        self.deepseek_api_key_official_var = tk.StringVar(value=self.config.get('tools', {}).get('deepseek', {}).get('api_key_official', ''))
        ttk.Entry(frame, textvariable=self.deepseek_api_key_official_var, width=60, show="*").grid(row=row, column=1, sticky="w", padx=5)
        row += 1

        ttk.Label(frame, text="Ключ AITUNNEL:").grid(row=row, column=0, sticky="w", pady=5, padx=10)
        self.deepseek_api_key_aitunnel_var = tk.StringVar(value=self.config.get('tools', {}).get('deepseek', {}).get('api_key_aitunnel', ''))
        ttk.Entry(frame, textvariable=self.deepseek_api_key_aitunnel_var, width=60, show="*").grid(row=row, column=1, sticky="w", padx=5)
        row += 1

        ttk.Label(frame, text="Примечание: official использует api.deepseek.com, aitunnel использует api.aitunnel.ru.", font=('', 9), foreground="gray").grid(row=row, column=0, columnspan=2, sticky="w", pady=5, padx=10)
        row += 1

        ttk.Button(frame, text="Применить", command=self.apply_tools, bootstyle="primary").grid(row=row, column=0, columnspan=2, pady=20)

    def _create_system_tab(self, notebook):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Система")

        row = 0
        # Имя ассистента
        ttk.Label(frame, text="Имя ассистента:").grid(row=row, column=0, sticky="w", pady=5, padx=10)
        self.assistant_name_var = tk.StringVar(value=self.config.get('system', {}).get('assistant_name', 'Ассистент'))
        ttk.Entry(frame, textvariable=self.assistant_name_var, width=30).grid(row=row, column=1, sticky="w", pady=5)
        row += 1

        # Автоматическое включение микрофона
        self.auto_enable_mic_var = tk.BooleanVar(value=self.config.get('system', {}).get('auto_enable_mic', True))
        ttk.Checkbutton(frame, text="Автоматически включать микрофон при запуске", 
                        variable=self.auto_enable_mic_var, bootstyle="round-toggle")\
            .grid(row=row, column=0, columnspan=2, sticky="w", padx=10, pady=5)
        row += 1

        # Сворачивание в трей
        self.minimize_to_tray_var = tk.BooleanVar(value=self.config.get('system', {}).get('minimize_to_tray', False))
        ttk.Checkbutton(frame, text="Сворачивать в трей вместо закрытия", 
                        variable=self.minimize_to_tray_var, bootstyle="round-toggle")\
            .grid(row=row, column=0, columnspan=2, sticky="w", padx=10, pady=5)
        row += 1

        ttk.Button(frame, text="Применить", command=self.apply_system, bootstyle="primary")\
            .grid(row=row, column=0, columnspan=2, pady=20)

    def apply_rag(self):
        updates = {
            'top_k': int(self.top_k_var.get()),
            'embed_model': self.embed_model_var.get().strip()
        }
        self._update_config_and_assistant(updates, 'memory')
        messagebox.showinfo("Успех", "Настройки памяти сохранены!")

    def apply_tts(self):
        updates = {
            'engine': self.tts_engine_var.get(),
            'voice': self.edge_voice_var.get(),
            'voice_id': int(self.pyttsx3_voice_id_var.get()),
            'silero_language': self.silero_language_var.get(),
            'silero_speaker': self.silero_speaker_var.get(),
            'silero_device': self.silero_device_var.get(),
        }
        self._update_config_and_assistant(updates, 'tts')
        messagebox.showinfo("Успех", "Настройки TTS сохранены!")

    def apply_prompt(self):
        new_prompt = self.prompt_text.get("1.0", tk.END).strip()
        self._update_config_and_assistant({'system_prompt': new_prompt}, 'llm')
        messagebox.showinfo("Успех", "Системный промпт обновлён!")

    def apply_tools(self):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                current_config = yaml.safe_load(f)
        except:
            current_config = self.config.copy()

        if 'tools' not in current_config:
            current_config['tools'] = {}
        if 'tavily' not in current_config['tools']:
            current_config['tools']['tavily'] = {}
        if 'deepseek' not in current_config['tools']:
            current_config['tools']['deepseek'] = {}

        current_config['tools']['tavily']['enabled'] = self.tavily_enabled_var.get()
        current_config['tools']['tavily']['api_key'] = self.tavily_api_key_var.get().strip()
        current_config['tools']['deepseek']['enabled'] = self.deepseek_enabled_var.get()
        current_config['tools']['deepseek']['provider'] = self.deepseek_provider_var.get()
        current_config['tools']['deepseek']['api_key_official'] = self.deepseek_api_key_official_var.get().strip()
        current_config['tools']['deepseek']['api_key_aitunnel'] = self.deepseek_api_key_aitunnel_var.get().strip()

        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            yaml.dump(current_config, f, allow_unicode=True, sort_keys=False)

        self.config = current_config
        self.assistant.config = current_config
        self.assistant.__init__(self.config)
        messagebox.showinfo("Успех", "Настройки инструментов сохранены! Возможно, потребуется перезапустить программу для полного применения.")

    def apply_system(self):
        updates = {
            'assistant_name': self.assistant_name_var.get().strip(),
            'auto_enable_mic': self.auto_enable_mic_var.get(),
            'minimize_to_tray': self.minimize_to_tray_var.get()
        }
        self._update_config_and_assistant(updates, 'system')
        messagebox.showinfo("Успех", "Системные настройки сохранены!")

    def _update_config_and_assistant(self, updates, section=None):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                current = yaml.safe_load(f) or {}
        except:
            current = self.config.copy()
        if section:
            if section not in current:
                current[section] = {}
            current[section].update(updates)
        else:
            current.update(updates)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            yaml.dump(current, f, allow_unicode=True, sort_keys=False)
        self.config = current
        self.assistant.config = current
        # Перезагружаем ассистента с новым именем
        self.assistant.__init__(self.config)

# ------------------------------------------------------------
#  Окно памяти
# ------------------------------------------------------------
class MemoryWindow(ttk.Toplevel):
    def __init__(self, parent, assistant):
        super().__init__(parent)
        self.assistant = assistant
        self.title("Управление памятью Мэй")
        self.geometry("900x710")
        self.minsize(900, 710)
        self.transient(parent)
        self.grab_set()

        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="🔄 Обновить список", command=self.refresh_list, bootstyle="info").pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="🗑️ Очистить всё", command=self.clear_all, bootstyle="danger").pack(side=tk.LEFT, padx=5)

        self.count_var = tk.StringVar(value="Записей: ?")
        ttk.Label(main_frame, textvariable=self.count_var).pack(anchor="w", pady=5)

        search_frame = ttk.Frame(main_frame)
        search_frame.pack(fill=tk.X, pady=5)
        ttk.Label(search_frame, text="Поиск по тексту:").pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        ttk.Entry(search_frame, textvariable=self.search_var, width=50).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(search_frame, text="🔍 Найти", command=self.search, bootstyle="primary").pack(side=tk.LEFT, padx=5)

        filter_frame = ttk.Frame(main_frame)
        filter_frame.pack(fill=tk.X, pady=5)
        ttk.Label(filter_frame, text="Фильтр по роли:").pack(side=tk.LEFT)
        self.role_var = tk.StringVar(value="Все")
        ttk.Combobox(filter_frame, textvariable=self.role_var, values=["Все", "user", "assistant", "fact", "summary", "user_profile"], width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(filter_frame, text="Применить", command=self.apply_filter, bootstyle="secondary").pack(side=tk.LEFT, padx=5)

        columns = ("ID", "Роль", "Время", "Текст (первые 50 символов)")
        self.tree = ttk.Treeview(main_frame, columns=columns, show="headings", height=15)
        self.tree.heading("ID", text="ID")
        self.tree.heading("Роль", text="Роль")
        self.tree.heading("Время", text="Время")
        self.tree.heading("Текст (первые 50 символов)", text="Текст (первые 50 символов)")
        self.tree.column("ID", width=150)
        self.tree.column("Роль", width=80)
        self.tree.column("Время", width=150)
        self.tree.column("Текст (первые 50 символов)", width=300)
        self.tree.pack(fill=tk.BOTH, expand=True, pady=10)

        tree_scroll = ttk.Scrollbar(main_frame, orient="vertical", command=self.tree.yview)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=tree_scroll.set)

        action_frame = ttk.Frame(main_frame)
        action_frame.pack(fill=tk.X, pady=5)

        ttk.Button(action_frame, text="📋 Показать полный текст", command=self.show_full, bootstyle="info").pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="❌ Удалить выбранное", command=self.delete_selected, bootstyle="danger").pack(side=tk.LEFT, padx=5)

        self.status_var = tk.StringVar()
        ttk.Label(main_frame, textvariable=self.status_var, foreground="red").pack(anchor="w", pady=5)

        self.all_entries = []
        self.refresh_list()

    def refresh_list(self):
        try:
            results = self.assistant.memory.collection.get()
            self.all_entries = list(zip(results['ids'], results['metadatas'], results['documents']))
            self.count_var.set(f"Записей: {len(self.all_entries)}")
            print(f"🔄 Обновление памяти: найдено {len(self.all_entries)} записей")
            self.tree.delete(*self.tree.get_children())
            for doc_id, meta, doc in self.all_entries:
                role = meta.get('role', 'unknown')
                timestamp = meta.get('timestamp', 'unknown')[:19]
                preview = doc[:50] + "..." if len(doc) > 50 else doc
                self.tree.insert("", "end", values=(doc_id, role, timestamp, preview))
            self.status_var.set("")
        except Exception as e:
            self.status_var.set(f"Ошибка загрузки: {e}")
            print(f"❌ Ошибка в refresh_list: {e}")

    def search(self):
        query = self.search_var.get().strip()
        if not query:
            self.refresh_list()
            return
        similar = self.assistant.memory.search_similar(query, top_k=20)
        self.tree.delete(*self.tree.get_children())
        for item in similar:
            for eid, meta, doc in self.all_entries:
                if doc == item['text']:
                    role = meta.get('role', 'unknown')
                    ts = meta.get('timestamp', '—')[:19]
                    preview = doc[:50] + "..." if len(doc) > 50 else doc
                    self.tree.insert("", "end", values=(eid, role, ts, preview))
                    break

    def apply_filter(self):
        role = self.role_var.get()
        if role == "Все":
            self.refresh_list()
            return
        filtered = [(id_, m, d) for id_, m, d in self.all_entries if m.get('role') == role]
        self.tree.delete(*self.tree.get_children())
        for id_, meta, doc in filtered:
            ts = meta.get('timestamp', '—')[:19]
            preview = doc[:50] + "..." if len(doc) > 50 else doc
            self.tree.insert("", "end", values=(id_, role, ts, preview))

    def show_full(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showwarning("Выберите запись")
            return
        item = self.tree.item(sel[0])
        doc_id = item['values'][0]
        for eid, meta, doc in self.all_entries:
            if eid == doc_id:
                top = ttk.Toplevel(self)
                top.title("Полный текст")
                top.geometry("700x500")
                t = scrolledtext.ScrolledText(top, wrap=tk.WORD)
                t.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                t.insert("1.0", f"ID: {doc_id}\nРоль: {meta.get('role')}\nВремя: {meta.get('timestamp')}\n\n{doc}")
                t.config(state='disabled')
                return
        messagebox.showerror("Ошибка", "Текст не найден.")

    def delete_selected(self):
        sel = self.tree.selection()
        if not sel:
            return
        item = self.tree.item(sel[0])
        doc_id = item['values'][0]
        if messagebox.askyesno("Удалить", "Удалить эту запись?"):
            self.assistant.memory.collection.delete(ids=[doc_id])
            self.refresh_list()

    def clear_all(self):
        if messagebox.askyesno("Очистить", "Удалить всю память?"):
            # Удаляем коллекцию
            self.assistant.memory._client.delete_collection(self.assistant.memory.collection_name)
            # Сбрасываем кэш в объекте памяти, чтобы при следующем обращении создалась новая коллекция
            self.assistant.memory._collection = None
            # Создаём новую коллекцию (обращение к свойству collection создаст её)
            _ = self.assistant.memory.collection
            self.refresh_list()

# ------------------------------------------------------------
#  Запуск программы
# ------------------------------------------------------------
if __name__ == "__main__":
    os.chdir(Path(__file__).parent)

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assistant = Assistant(config)  # по умолчанию api_mode=False
    gui = AssistantGUI(assistant)

    threading.Thread(target=assistant.preload_embedder, daemon=True).start()

    try:
        gui.root.mainloop()
    finally:
        if gui.voice_listener:
            gui.voice_listener.stop()