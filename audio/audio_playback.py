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