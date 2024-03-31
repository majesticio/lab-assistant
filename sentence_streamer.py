# app/sentence_streamer.py
class SentenceStreamer:
    def __init__(self, sentence_queue=None):
        self.buffer = ""  # Initialize the buffer for holding streamed text
        self.sentence_queue = sentence_queue

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
                if sentence and self.sentence_queue:
                    self.sentence_queue.put(sentence)  # Enqueue the sentence for TTS processing
                self.buffer = self.buffer[i + 1:].lstrip()
                i = 0
            else:
                i += 1