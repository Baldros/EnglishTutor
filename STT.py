import queue
import threading
import time

import numpy as np
import sounddevice as sd
import torch
from transformers import pipeline


class ContinuousSTT:
    def __init__(
        self,
        model_name: str = "distil-whisper/distil-large-v3",
        sample_rate: int = 16000,
        block_duration: float = 0.1,
        speech_threshold: float = 0.015,
        silence_duration: float = 0.8,
        min_utterance_duration: float = 0.5,
        max_utterance_duration: float = 20.0,
        input_device: int | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.blocksize = int(sample_rate * block_duration)
        self.speech_threshold = speech_threshold
        self.silence_duration = silence_duration
        self.min_utterance_duration = min_utterance_duration
        self.max_utterance_duration = max_utterance_duration
        self.input_device = input_device
        self.model_name = model_name

        self._device = 0 if torch.cuda.is_available() else -1
        self._torch_dtype = torch.float16 if self._device >= 0 else torch.float32
        self.model = self._build_model(self._device, self._torch_dtype)

        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._paused = threading.Event()
        self._recording = False
        self._frames: list[np.ndarray] = []
        self._speech_started_at = 0.0
        self._last_speech_at = 0.0
        self._stream: sd.InputStream | None = None

    def start(self) -> None:
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.blocksize,
            device=self.input_device,
            callback=self._audio_callback,
        )
        self._stream.start()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def pause(self) -> None:
        self._paused.set()
        self._reset_state()

    def resume(self) -> None:
        self._paused.clear()

    def get_next_audio(self, timeout: float = 0.1) -> np.ndarray | None:
        try:
            return self._audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def transcribe(self, audio: np.ndarray, language: str = "en") -> str:
        asr_input = {"raw": audio, "sampling_rate": self.sample_rate}
        generate_kwargs = {"task": "transcribe"}
        if language:
            generate_kwargs["language"] = language

        try:
            result = self.model(asr_input, generate_kwargs=generate_kwargs)
            return str(result.get("text", "")).strip()
        except Exception:
            # Some Whisper variants/configs reject explicit language.
            result = self.model(asr_input, generate_kwargs={"task": "transcribe"})
            return str(result.get("text", "")).strip()

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        del frames, time_info
        if status:
            print(f"[STT warning] {status}")

        if self._paused.is_set():
            self._reset_state()
            return

        chunk = indata[:, 0].copy()
        now = time.monotonic()
        rms = float(np.sqrt(np.mean(chunk**2)))

        if rms >= self.speech_threshold:
            if not self._recording:
                self._recording = True
                self._speech_started_at = now
            self._last_speech_at = now
            self._frames.append(chunk)
            return

        if not self._recording:
            return

        self._frames.append(chunk)
        utterance_duration = now - self._speech_started_at
        silence_elapsed = now - self._last_speech_at

        if utterance_duration >= self.max_utterance_duration:
            self._finalize_utterance()
            return

        if silence_elapsed >= self.silence_duration:
            if utterance_duration >= self.min_utterance_duration:
                self._finalize_utterance()
            else:
                self._reset_state()

    def _finalize_utterance(self) -> None:
        if not self._frames:
            self._reset_state()
            return
        audio = np.concatenate(self._frames, axis=0).astype(np.float32)
        self._audio_queue.put(audio)
        self._reset_state()

    def _reset_state(self) -> None:
        self._recording = False
        self._frames = []
        self._speech_started_at = 0.0
        self._last_speech_at = 0.0

    def _build_model(self, device: int, torch_dtype: torch.dtype):
        try:
            return pipeline(
                "automatic-speech-recognition",
                model=self.model_name,
                device=device,
                torch_dtype=torch_dtype,
            )
        except RuntimeError as exc:
            if device >= 0 and self._is_cuda_runtime_error(exc):
                print(
                    "[STT warning] CUDA indisponivel no init do STT. "
                    "Usando CPU."
                )
                self._device = -1
                self._torch_dtype = torch.float32
                return pipeline(
                    "automatic-speech-recognition",
                    model=self.model_name,
                    device=-1,
                    torch_dtype=torch.float32,
                )
            raise

    def _is_cuda_runtime_error(self, exc: Exception) -> bool:
        text = str(exc).lower()
        cuda_markers = (
            "cublas",
            "cudnn",
            "cuda",
            "libcuda",
            "cannot be loaded",
            "not found",
        )
        return any(marker in text for marker in cuda_markers)
