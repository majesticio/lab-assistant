
from dotenv import load_dotenv
import json
import os

load_dotenv()
history_path = os.getenv('HISTORY_PATH')
model_name = os.getenv('MODEL_NAME')

def process_transcription_to_chat(client, history, transcription_queue, streamer):
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