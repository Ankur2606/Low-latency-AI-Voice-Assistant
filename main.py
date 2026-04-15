import asyncio
import argparse
from utils.audio_processing import capture_and_transcribe_audio
from utils.llm_interaction import generate_llm_response
from utils.tts_conversion import (
    convert_text_to_speech, play_audio,
    stream_kokoro_tts
)
import os

async def main_interaction_loop(use_kokoro=False):
    """Main loop for capturing speech, generating responses, and playing audio."""
    if use_kokoro:
        print("🎤 TTS Engine: Kokoro-82M (local)")
    else:
        print("🎤 TTS Engine: Edge-TTS (cloud)")

    conversation_history = []  # Store history of user inputs and assistant responses

    while True:
        # Step 1: Capture and transcribe speech
        transcribed_text = await capture_and_transcribe_audio()
        if not transcribed_text:
            print("Please try speaking again.")
            continue

        print(f"You said: {transcribed_text}\n") 

        if 'stop' in transcribed_text.lower():
            print("Goodbye!")
            break

        # Append user input to history
        conversation_history.append({"User": transcribed_text})
        
        # Step 2: Generate a response from LLM with history context
        history_context = ' '.join(
            [f"{key}: {value}" for entry in conversation_history for key, value in entry.items()]
        )
        response = generate_llm_response(transcribed_text, history_context)
        print(f"Assistant: {response}\n")

        # Append assistant response to history
        conversation_history.append({"Assistant": response})

        # Step 3: Convert the response text to speech and play it
        if use_kokoro:
            # Kokoro streams audio directly to speakers — no file I/O
            stream_kokoro_tts(response)
        else:
            audio_file = await convert_text_to_speech(response, rate="+0%", pitch="+0Hz")
            play_audio(audio_file)

        # Optional: Delete the audio file after playing (uncomment if needed)
        # os.remove(audio_file)
        print("Audio playback finished.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Low-Latency AI Voice Assistant")
    parser.add_argument(
        "-k", "--kokoro",
        action="store_true",
        help="Use Kokoro-82M local TTS instead of Edge-TTS (cloud)"
    )
    args = parser.parse_args()

    asyncio.run(main_interaction_loop(use_kokoro=args.kokoro))
