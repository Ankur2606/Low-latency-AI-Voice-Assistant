import asyncio
from Models.faster_whisper_stt_tiny import capture_audio, transcribe_audio  # Importing from faster_whisper_stt_tiny.py
from Models.llm_response import generate  # Importing from llm_response.py
from ...utils.tts_conversion import convert_text_to_speech  # Importing from tts.py
import edge_tts  # For TTS conversion using Edge-TTS
from playsound import playsound  # To play the audio file
import os  # For file deletion
import time


# Main interaction loop with history tracking
async def main_interaction_loop():
    conversation_history = []  # To store the history of user inputs and assistant responses
    
    while True:
        # Step 1: Capture and transcribe speech
        audio_file = await capture_audio()
        if not audio_file:
            print("Please try speaking again.")
            continue

        transcribed_text = await transcribe_audio(audio_file, language="en")
        if 'watching' in transcribed_text or "Let's go" in transcribed_text:
            print("Unclear transcription, please try again.")
            continue
        print(f"You said: {transcribed_text}\n") 

        if 'stop' in transcribed_text.lower():
            print("Goodbye!")
            break

        # Append user input to history
        conversation_history.append({"User": transcribed_text})
        
        # Step 2: Generate a response from LLM with history context
        history_context = ' '.join([f"{key}: {value}" for entry in conversation_history for key, value in entry.items()])
        response = generate(
            f"Respond to '{transcribed_text}' based on the following context: {history_context}. Don't exceed more than 20 words.",
            system_prompt="Be concise, helpful, and friendly. Respond briefly with at most 20 words. Always end your response with a friendly note. Only Include Text in the entire response.",
            model="mistralai/Mistral-7B-Instruct-v0.3",
            temperature=0.7,
            chat_template="mistral",
            verbose=False
        )
        print(f"Assistant: {response}\n")

        # Append assistant response to history
        conversation_history.append({"Assistant": response})

        # Step 3: Convert the response text to speech
        audio_file = await convert_text_to_speech(response)

        # Step 4: Play the audio
        playsound(audio_file)
        print("Audio playback finished.\n")


# Run the interaction loop in an asyncio event loop
asyncio.run(main_interaction_loop())
