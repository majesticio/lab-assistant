import os
import whisper
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

whisper_model = os.getenv('WHISPER_MODEL')

class TranscriptionHandler:
    def __init__(self, model_name=whisper_model, recordings_folder="recordings"):
        self.model = whisper.load_model(model_name)
        self.recordings_folder = recordings_folder

        # Ensure recordings directory exists
        os.makedirs(self.recordings_folder, exist_ok=True)

    def transcribe_file(self, file_path):
        if not os.path.isfile(file_path):
            print(f"Attempted file path: {file_path}")
            print("File does not exist.")
            return None

        try:
            print(f"Transcribing: {file_path}")
            result = self.model.transcribe(file_path, fp16=False)
            transcription = result['text']
            return transcription
        except Exception as e:
            print(f"Error during transcription: {e}")
            return None