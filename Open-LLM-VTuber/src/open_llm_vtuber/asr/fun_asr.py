import os
import tempfile
import threading
import re
from typing import Optional
import numpy as np
import soundfile as sf
import librosa
from loguru import logger

from .asr_interface import ASRInterface

class VoiceRecognition(ASRInterface):
    def __init__(
        self,
        model_name: str = "iic/SenseVoiceSmall",  # будет использоваться как путь
        language: str = "auto",  # игнорируется
        vad_model: str = "fsmn-vad",  # игнорируется
        punc_model: str = "ct-punc",  # игнорируется
        ncpu: int = None,  # игнорируется
        hub: str = None,  # игнорируется
        device: str = "cpu",
        disable_update: bool = True,  # игнорируется
        sample_rate: int = 16000,  # игнорируется
        use_itn: bool = False,  # игнорируется
    ):
        self.model_path = model_name
        if self.model_path is None:
            raise ValueError("model_name must be provided as path to GigaAM model folder")
        self.device = device
        self.model = None
        self.ready = False

        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self):
        try:
            import onnx_asr
            self.model = onnx_asr.load_model(self.model_path)
            self.ready = True
            logger.info(f"GigaAM (onnx-asr) model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load GigaAM model via onnx-asr: {e}")
            self.ready = False

    @staticmethod
    def _clean_text(text: str) -> str:
        if not text:
            return ""
        cleaned = re.sub(r'<\|[^>]+\|>', '', text)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    async def transcribe(self, audio_data: bytes, sample_rate: int = 16000) -> Optional[str]:
        if not self.ready:
            logger.warning("ASR model not ready yet.")
            return None

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            temp_filename = f.name
        try:
            with open(temp_filename, "wb") as f_out:
                f_out.write(audio_data)

            import asyncio
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._transcribe_sync,
                temp_filename
            )
            return result
        except Exception as e:
            logger.error(f"ASR transcription failed: {e}")
            return None
        finally:
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)

    def transcribe_np(self, audio: np.ndarray, sample_rate: int = 16000) -> Optional[str]:
        if not self.ready:
            return None
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            temp_filename = f.name
        try:
            sf.write(temp_filename, audio, sample_rate)
            return self._transcribe_sync(temp_filename)
        except Exception as e:
            logger.error(f"ASR transcribe_np failed: {e}")
            return None
        finally:
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)

    def _transcribe_sync(self, audio_path: str) -> Optional[str]:
        try:
            # Ресемплинг в 16 кГц, моно
            y, sr = librosa.load(audio_path, sr=16000, mono=True)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                temp_fixed = f.name
            sf.write(temp_fixed, y, 16000, subtype='PCM_16')

            result = self.model.recognize(temp_fixed)
            if result:
                cleaned = self._clean_text(result)
                return cleaned if cleaned else None
            return None
        except Exception as e:
            logger.error(f"Error during synchronous transcription: {e}")
            return None
        finally:
            if os.path.exists(temp_fixed):
                os.unlink(temp_fixed)

    def transcribe_sync(self, audio_data: bytes, sample_rate: int = 16000) -> Optional[str]:
        if not self.ready:
            return None
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            temp_filename = f.name
        try:
            with open(temp_filename, "wb") as f_out:
                f_out.write(audio_data)
            y, sr = librosa.load(temp_filename, sr=16000, mono=True)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                temp_fixed = f.name
            sf.write(temp_fixed, y, 16000, subtype='PCM_16')

            result = self.model.recognize(temp_fixed)
            if result:
                cleaned = self._clean_text(result)
                return cleaned if cleaned else None
            return None
        except Exception as e:
            logger.error(f"ASR sync transcription failed: {e}")
            return None
        finally:
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
            if os.path.exists(temp_fixed):
                os.unlink(temp_fixed)