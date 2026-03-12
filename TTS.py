import numpy as np
import sounddevice as sd
import torch
from transformers import pipeline


class RealtimeTTS:
    def __init__(
        self,
        model_name: str = "facebook/mms-tts-eng",
    ) -> None:
        device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline("text-to-speech", model=model_name, device=device)

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        output = self.pipe(text)
        audio = np.asarray(output["audio"], dtype=np.float32).squeeze()
        if audio.ndim > 1:
            audio = audio[:, 0]
        sampling_rate = int(output["sampling_rate"])
        return audio, sampling_rate

    def speak(self, text: str) -> None:
        audio, sampling_rate = self.synthesize(text)
        self.play_audio(audio, sampling_rate)

    def play_audio(self, audio: np.ndarray, sampling_rate: int) -> None:
        sd.play(audio, sampling_rate)
        sd.wait()
