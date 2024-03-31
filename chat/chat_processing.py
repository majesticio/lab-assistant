
from dotenv import load_dotenv
import json
import os

from openai import OpenAI

load_dotenv()

# Access environment variables
base_url = os.getenv('BASE_URL')
api_key = os.getenv('API_KEY')
history_path = os.getenv('HISTORY_PATH')
system_msg = os.getenv("SYSTEM_MSG")
model_name = os.getenv('MODEL_NAME')

# Initialize OpenAI client
client = OpenAI(base_url=base_url, api_key=api_key)

# Load or create history
if os.path.exists(history_path):
    with open(history_path, 'r') as file:
        history = json.load(file)
else:
    # Create the directory if it doesn't exist
    history_dir = os.path.dirname(history_path)
    if not os.path.exists(history_dir):
        os.makedirs(history_dir)
    
    # Create the .chat_history file with the default history
    history = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": "Hello Mage, follow my instructions carefully and think through each problem step by step."},
    ]
    with open(history_path, 'w') as file:
        json.dump(history, file)

def process_transcription_to_chat(transcription_queue, streamer):
    """Process transcriptions and send them to the chat API."""

    while True:
        transcription = transcription_queue.get()

        user_msg = {"role": "user", "content": transcription}
        history.append(user_msg)

        # Chat API call
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=history,
                temperature=0.7,
                stream=True
            )
            
            response = {"role": "assistant", "content": ""}
    
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    text_chunk = chunk.choices[0].delta.content
                    # Directly feed chunks from chat API response to the SentenceStreamer
                    response["content"] += text_chunk
                    streamer.process_text_chunk(text_chunk)
            
            history.append(response)
        except Exception as e:
            print(f"Error during chat interaction: {e}")
        finally:
        # Write the history to the history_path before exiting
            with open(history_path, 'w') as file:
                json.dump(history, file)

        transcription_queue.task_done()