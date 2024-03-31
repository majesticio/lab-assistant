# app/tts_handler.py
import os
from datetime import datetime
from TTS.api import TTS
import torch

class TTSHandler:
    def __init__(self, model_name="tts_models/en/jenny/jenny", wav_dir="speaker/"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts_engine = TTS(model_name=model_name, progress_bar=True).to(self.device)
        self.wav_dir = wav_dir

        if not os.path.exists(self.wav_dir):
            os.makedirs(self.wav_dir)

    def text_to_speech(self, text):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        file_path = os.path.join(self.wav_dir, f"sentence_{timestamp}.wav")

        try:
            self.tts_engine.tts_to_file(text=text, file_path=file_path)
            return file_path
        except Exception as e:
            print(f"Error during speech synthesis: {e}")
            return None