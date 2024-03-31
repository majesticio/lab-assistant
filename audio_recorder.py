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


from dotenv import load_dotenv

load_dotenv()

record_button = os.getenv('RECORD_BUTTON')

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
        print(f"\nPress and hold the {record_button} to start recording. Release to stop.")
        while True:
            try:
                keyboard.wait(record_button)
                print("\nRecording started...")
                self.frames = []
                with sd.InputStream(callback=self.callback, samplerate=self.samplerate, channels=self.channels, dtype='float32'):
                    while keyboard.is_pressed(record_button):
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