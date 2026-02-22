import os
import tempfile
import threading
import re
from pathlib import Path
from typing import Optional
import numpy as np
import soundfile as sf
import torch
from loguru import logger

from .tts_interface import TTSInterface


class SileroTTS(TTSInterface):
    def __init__(self, **kwargs):
        self.language = kwargs.get("language", "ru")
        self.speaker = kwargs.get("speaker", "v5_ru")
        self.device = kwargs.get("device", "cpu")
        self.sample_rate = 48000
        self.max_chars = 800  # безопасный лимит (Silero не любит >1000)
        self.model = None
        self.ready = False

        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self):
        try:
            import silero
            if self.device == "cuda" and torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info("Silero TTS using CUDA")
            else:
                device = torch.device("cpu")
                if self.device == "cuda":
                    logger.warning("CUDA not available, falling back to CPU")

            self.model, _ = silero.silero_tts(
                language=self.language,
                speaker=self.speaker,
                device=device
            )
            self.ready = True
            logger.info(f"Silero TTS loaded: {self.language}/{self.speaker} on {device}")
        except Exception as e:
            logger.error(f"Failed to load Silero TTS: {e}")
            self.ready = False

    def _split_text(self, text: str) -> list[str]:
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
        # Если предложение длиннее max_chars (редко) – режем принудительно
        final_parts = []
        for p in parts:
            if len(p) <= self.max_chars:
                final_parts.append(p)
            else:
                for i in range(0, len(p), self.max_chars):
                    final_parts.append(p[i:i+self.max_chars])
        return final_parts

    def _synthesize_part(self, text: str) -> Optional[np.ndarray]:
        """Синтезирует одну часть текста и возвращает numpy массив."""
        try:
            audio = self.model.apply_tts(text)
            if isinstance(audio, torch.Tensor):
                audio_np = audio.cpu().numpy()
            elif isinstance(audio, list):
                if len(audio) == 1 and isinstance(audio[0], torch.Tensor):
                    audio_np = audio[0].cpu().numpy()
                else:
                    audio_np = np.array(audio)
            elif isinstance(audio, np.ndarray):
                audio_np = audio
            else:
                raise TypeError(f"Unexpected audio type: {type(audio)}")
            audio_np = np.squeeze(audio_np).astype(np.float32)
            return audio_np
        except Exception as e:
            logger.error(f"Silero synthesis failed for part: {e}")
            return None

    def generate_audio(self, text: str, voice: Optional[str] = None) -> Optional[str]:
        """
        Генерирует аудио из текста, при необходимости разбивая на части.
        Возвращает путь к временному WAV-файлу (объединённому из частей).
        Параметр voice игнорируется.
        """
        if not self.ready:
            logger.warning("Silero TTS not ready yet")
            return None

        parts = self._split_text(text)
        if len(parts) > 1:
            logger.info(f"Long text split into {len(parts)} parts")

        # Синтезируем все части
        audio_parts = []
        for part in parts:
            audio_np = self._synthesize_part(part)
            if audio_np is None:
                continue
            audio_parts.append(audio_np)

        if not audio_parts:
            return None

        # Объединяем все части в один массив
        combined = np.concatenate(audio_parts)

        # Сохраняем во временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            temp_filename = f.name
        sf.write(temp_filename, combined, self.sample_rate)
        return temp_filename