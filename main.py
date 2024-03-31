import threading
import time
import queue
import os

from chat.sentence_streamer import SentenceStreamer
from chat.transcription_handler import TranscriptionHandler
from audio.audio_playback import playback_audio_files
from chat.chat_processing import process_transcription_to_chat
from tts.tts_processing import process_tts

# Initialize queues
transcription_queue = queue.Queue()
tts_queue = queue.Queue()
speaker_queue = queue.Queue()

# Initialize SentenceStreamer to process chat API responses into sentences
streamer = SentenceStreamer(sentence_queue=tts_queue)

def main():
    transcription_handler = TranscriptionHandler()

    # Threads for processing chat responses to sentences, converting sentences to TTS, and playback
    threading.Thread(target=process_transcription_to_chat, args=(transcription_queue, streamer), daemon=True).start()
    threading.Thread(target=process_tts, args=(tts_queue, speaker_queue), daemon=True).start()
    threading.Thread(target=playback_audio_files, args=(speaker_queue,), daemon=True).start()

    while True:
        try:
            for file_name in os.listdir(transcription_handler.recordings_folder):
                if file_name.endswith(".wav"):
                    file_path = os.path.join(transcription_handler.recordings_folder, file_name)
                    # print(f"Transcribing file: {file_path}")
                    transcription = transcription_handler.transcribe_file(file_path)
                    if transcription:
                        print(f"Transcription: {transcription}")
                        transcription_queue.put(transcription)
                        os.remove(file_path)
                        # print(f"Removed file: {file_path}")
            time.sleep(0.5)
        except KeyboardInterrupt:
            print("Program interrupted by user. Exiting...")
            break


if __name__ == "__main__":
    main()