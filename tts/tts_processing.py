from .tts_handler import TTSHandler

def process_tts(tts_queue, speaker_queue):
    """Process sentences from the TTS queue and send the generated audio file paths to the speaker queue."""
    tts_handler = TTSHandler(wav_dir="speaker")

    while True:
        sentence = tts_queue.get()
        audio_file_path = tts_handler.text_to_speech(sentence)
        if audio_file_path:
            # print(f"Generated audio file: {audio_file_path}")
            speaker_queue.put(audio_file_path)
        tts_queue.task_done()