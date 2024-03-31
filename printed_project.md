.
├── audio
├── audio_playback.py
├── audio_recorder.py
├── chat
├── chat_processing.py
├── main.py
├── recordings
├── requirements.txt
├── sentence_streamer.py
├── speaker
├── transcription_handler.py
├── tts
├── tts_handler.py
└── tts_processing.py

6 directories, 9 files

## ./chat_processing.py
```python
def process_transcription_to_chat(client, transcription_queue, streamer):
    """Process transcriptions and send them to the chat API."""
    history = []

    while True:
        transcription = transcription_queue.get()

        user_msg = {"role": "user", "content": transcription}
        history.append(user_msg)

        # Chat API call
        try:
            completion = client.chat.completions.create(
                model="mixtral",
                messages=history,
                temperature=0.7,
                stream=True
            )
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    # Directly feed chunks from chat API response to the SentenceStreamer
                    streamer.process_text_chunk(chunk.choices[0].delta.content)
        except Exception as e:
            print(f"Error during chat interaction: {e}")

        transcription_queue.task_done()
```

## ./tts_processing.py
```python
from tts_handler import TTSHandler

def process_tts(tts_queue, speaker_queue):
    """Process sentences from the TTS queue and send the generated audio file paths to the speaker queue."""
    tts_handler = TTSHandler(wav_dir="speaker")

    while True:
        sentence = tts_queue.get()
        audio_file_path = tts_handler.text_to_speech(sentence)
        if audio_file_path:
            print(f"Generated audio file: {audio_file_path}")
            speaker_queue.put(audio_file_path)
        tts_queue.task_done()
```

## ./audio_recorder.py
```python
import os
import sys
import threading
import sounddevice as sd
import numpy as np
import keyboard
from datetime import datetime
import itertools
import wave
import pyloudnorm as pyln

class AudioNormalizer:
    """Handles audio normalization in various modes."""
    def __init__(self, mode='lufs', samplerate=44100):
        self.mode = mode
        self.samplerate = samplerate

    def normalize(self, data):
        """Normalize audio data based on the specified mode."""
        if self.mode == 'peak':
            return self._normalize_peak(data)
        elif self.mode == 'rms':
            return self._normalize_rms(data)
        elif self.mode == 'lufs':
            return self._normalize_lufs(data)
        else:
            return data  # Bypasses normalization if mode is 'off' or invalid

    def _normalize_peak(self, data):
        peak = np.max(np.abs(data))
        return data / peak if peak != 0 else data

    def _normalize_rms(self, data, target_rms=0.1):
        current_rms = np.sqrt(np.mean(data**2))
        return data * (target_rms / current_rms) if current_rms != 0 else data

    def _normalize_lufs(self, data):
        meter = pyln.Meter(self.samplerate)  # create BS.1770 meter
        loudness = meter.integrated_loudness(data)
        target_lufs = -23.0
        return pyln.normalize.loudness(data, loudness, target_lufs)

class SpinningCursor:
    """Class for creating a spinning cursor in the command line."""
    def __enter__(self):
        self.stop_running = False
        self.cursor_thread = threading.Thread(target=self._animate)
        self.cursor_thread.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_running = True
        self.cursor_thread.join()

    def _animate(self):
        for cursor in itertools.cycle('|/-\\'):
            if self.stop_running:
                break
            sys.stdout.write('\r' + cursor)
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write('\r')
        sys.stdout.flush()

class AudioRecorder:
    def __init__(self, recordings_folder='recordings', samplerate=44100, channels=1, normalization_mode='lufs'):
        self.recordings_folder = recordings_folder
        self.samplerate = samplerate
        self.channels = channels
        self.normalizer = AudioNormalizer(normalization_mode, samplerate)
        self.frames = []
        self.ensure_folder_exists()

    def ensure_folder_exists(self):
        if not os.path.exists(self.recordings_folder):
            os.makedirs(self.recordings_folder)

    def record_audio(self):
        print("Press Ctrl+C anytime to stop the program.")
        print("\nPress and hold the space bar to start recording. Release to stop.")
        while True:
            try:
                keyboard.wait('space')
                print("\nRecording started...")
                self.frames = []
                with sd.InputStream(callback=self.callback, samplerate=self.samplerate, channels=self.channels, dtype='float32'):
                    while keyboard.is_pressed('space'):
                        sd.sleep(100)
                print("\nRecording stopped.")
                self.save_recording()
            except KeyboardInterrupt:
                print("\nExiting program...")
                break
            except Exception as e:
                print(f"\nAn error occurred: {e}")



    def callback(self, indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        scaled_indata = indata * 0.5
        self.frames.append(scaled_indata.copy())
        
    def save_recording(self):
        audio_data = np.concatenate(self.frames, axis=0)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}.wav"
        filepath = os.path.join(self.recordings_folder, filename)
        normalized_data = self.normalizer.normalize(audio_data)
        self.write_wav(filepath, normalized_data)
        print(f"Recording saved: {filepath}")
        # enqueue filename?
    def write_wav(self, filepath, data):
        scaled_data = np.int16(data / np.max(np.abs(data)) * 32767)
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.samplerate)
            wf.writeframes(scaled_data.tobytes())

if __name__ == "__main__":
    recorder = AudioRecorder()
    threading.Thread(target=recorder.record_audio(), daemon=True).start()
```

## ./sentence_streamer.py
```python
# app/sentence_streamer.py
class SentenceStreamer:
    def __init__(self):
        self.buffer = ""  # Initialize the buffer for holding streamed text

    def process_text_chunk(self, text_chunk):
        """Process a chunk of text, update the buffer, and add complete sentences for publishing."""
        self.buffer += text_chunk
        self._process_buffer()

    def _process_buffer(self):
        """Process the buffer to extract complete sentences and publish them."""
        i = 0
        while i < len(self.buffer):
            if self.buffer[i] in ".!?" and (i + 1 == len(self.buffer) or self.buffer[i + 1].isspace()):
                if i > 0 and (self.buffer[i - 1].isdigit() or (self.buffer[i - 1].isupper() and (i == 1 or self.buffer[i - 2].isspace()))):
                    i += 1
                    continue
                if i > 2 and self.buffer[i - 1] == '.' and self.buffer[i - 2].isupper() and self.buffer[i - 3] == ' ':
                    i += 1
                    continue
                sentence = self.buffer[:i + 1].strip()
                if sentence:
                    self.publish_function(sentence)  # Publish the sentence directly
                self.buffer = self.buffer[i + 1:].lstrip()
                i = 0
            else:
                i += 1

```

## ./audio_playback.py
```python
import os
import sounddevice as sd
import soundfile as sf

def playback_audio_files(speaker_queue):
    """Playback audio files from the speaker queue."""
    while True:
        audio_file_path = speaker_queue.get()
        print(f"Playing audio file: {audio_file_path}")
        try:
            data, fs = sf.read(audio_file_path, dtype='float32')
            sd.play(data, fs, blocking=True)
            print(f"Playback completed for: {audio_file_path}")
            os.remove(audio_file_path)
            print(f"Removed file: {audio_file_path}")
        except Exception as e:
            print(f"Failed to play back audio file {audio_file_path}: {e}")
        speaker_queue.task_done()
```

## ./transcription_handler.py
```python
import os
import whisper
from datetime import datetime

# Define the default model name here; you can adjust it based on your needs.
MODEL_NAME = "small"

class TranscriptionHandler:
    def __init__(self, model_name=MODEL_NAME, recordings_folder="recordings"):
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
```

## ./main.py
```python
import threading
import time
import queue
import os
from openai import OpenAI
from sentence_streamer import SentenceStreamer
from transcription_handler import TranscriptionHandler
from tts_handler import TTSHandler
from audio_playback import playback_audio_files
from chat_processing import process_transcription_to_chat
from tts_processing import process_tts

# Initialize OpenAI client
client = OpenAI(base_url="http://10.200.200.1:1234/v1", api_key="not-needed")

# Initialize queues
transcription_queue = queue.Queue()
tts_queue = queue.Queue()
speaker_queue = queue.Queue()

# Initialize SentenceStreamer to process chat API responses into sentences
streamer = SentenceStreamer(sentence_queue=tts_queue)

def main():
    transcription_handler = TranscriptionHandler()

    # Threads for processing chat responses to sentences, converting sentences to TTS, and playback
    threading.Thread(target=process_transcription_to_chat, args=(client, transcription_queue, streamer), daemon=True).start()
    threading.Thread(target=process_tts, args=(tts_queue, speaker_queue), daemon=True).start()
    threading.Thread(target=playback_audio_files, args=(speaker_queue,), daemon=True).start()

    while True:
        try:
            for file_name in os.listdir(transcription_handler.recordings_folder):
                if file_name.endswith(".wav"):
                    file_path = os.path.join(transcription_handler.recordings_folder, file_name)
                    print(f"Transcribing file: {file_path}")
                    transcription = transcription_handler.transcribe_file(file_path)
                    if transcription:
                        print(f"Transcription: {transcription}")
                        transcription_queue.put(transcription)
                        os.remove(file_path)
                        print(f"Removed file: {file_path}")
            time.sleep(1)
        except KeyboardInterrupt:
            print("Program interrupted by user. Exiting...")
            break

if __name__ == "__main__":
    main()
```

## ./tts_handler.py
```python
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
```

## ./requirements.txt
```python
fastapi
uvicorn
pika
pydantic
python-dotenv
TTS
openai

# old requirements
sounddevice
numpy
keyboard
python-dotenv
pyloudnorm
aiofiles
# whisper
openai-whisper @ git+https://github.com/openai/whisper.git@0a60fcaa9b86748389a656aa013c416030287d47
pydantic-settings
python-multipart
gradio

```
