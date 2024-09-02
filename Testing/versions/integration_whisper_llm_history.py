import asyncio
from collections import deque
from Models.faster_whisper_stt_tiny import capture_audio, transcribe_audio  # Importing from faster_whisper_stt_tiny.py
from Models.llm_response import generate  # Importing from llm_response.py
import time

# Initialize a deque to keep track of the last 10 transcriptions
conversation_memory = deque(maxlen=10)

# Main interaction loop
async def main_interaction_loop():
    while True:
        # Step 1: Capture and transcribe speech
        audio_file = await capture_audio()
        if not audio_file:
            print("Please try speaking again.")
            continue

        transcribed_text = await transcribe_audio(audio_file)
        if 'watching' in transcribed_text or "Let's go" in transcribed_text :
            print("Unclear transcription, please try again.")
            continue
        print(f"You said: {transcribed_text}\n") 

        # Add the transcribed text to the conversation memory
        conversation_memory.append(transcribed_text)

        # Create a string of the last 10 messages for contextual awareness
        memory_context = " ".join(conversation_memory)

        # Step 2: Wait for user input to continue or break the loop
        if 'stop' in transcribed_text or 'Stop' in transcribed_text:
            print("Goodbye!")
            break

        # Step 3: Generate a response from LLM using memory context
        prompt = f"With context '{memory_context}', respond to the latest input: '{transcribed_text}' in at most 20 words."
        response = generate(
            prompt,
            system_prompt="Be concise, helpful, and friendly. Respond briefly with at most 20 words. Always end your response with a positive or friendly note, like a smiley emote.",
            model="mistralai/Mistral-7B-Instruct-v0.3",
            temperature=0.7,
            chat_template="mistral",
            verbose=False
        )

        print(f"Assistant: {response}\n")
        
# Run the interaction loop in an asyncio event loop
asyncio.run(main_interaction_loop())
