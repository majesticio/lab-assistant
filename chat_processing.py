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