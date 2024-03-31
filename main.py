from dotenv import load_dotenv
from openai import OpenAI

import threading
import time
import queue
import json
import os

from chat.sentence_streamer import SentenceStreamer
from chat.transcription_handler import TranscriptionHandler
from audio.audio_playback import playback_audio_files
from chat.chat_processing import process_transcription_to_chat
from tts.tts_processing import process_tts

# Load environment variables from .env file
load_dotenv()

# Access environment variables
base_url = os.getenv('BASE_URL')
api_key = os.getenv('API_KEY')
history_path = os.getenv('HISTORY_PATH')
system_msg = os.getenv("SYSTEM_MSG")

# Initialize OpenAI client
client = OpenAI(base_url=base_url, api_key=api_key)

# Load or create history
if os.path.exists(history_path):
    with open(history_path, 'r') as file:
        history = json.load(file)
else:
    history = [
        {"role": "system", "content": system_msg },
    ]

# Initialize queues
transcription_queue = queue.Queue()
tts_queue = queue.Queue()
speaker_queue = queue.Queue()

# Initialize SentenceStreamer to process chat API responses into sentences
streamer = SentenceStreamer(sentence_queue=tts_queue)

def main():
    transcription_handler = TranscriptionHandler()

    # Threads for processing chat responses to sentences, converting sentences to TTS, and playback
    threading.Thread(target=process_transcription_to_chat, args=(client, history, transcription_queue, streamer), daemon=True).start()
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
            time.sleep(0.5)
        except KeyboardInterrupt:
            print("Program interrupted by user. Exiting...")
            break


if __name__ == "__main__":
    main()