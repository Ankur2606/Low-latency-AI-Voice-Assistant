import numpy as np
import soundfile as sf
import asyncio
import time
import os
import speech_recognition as sr
from faster_whisper import WhisperModel

# VAD Parameters
VAD_THRESHOLD = 0.5  # Adjust based on your environment

# Function to capture audio using SpeechRecognition
async def capture_audio(duration=5, filename="stt_transcribe.flac"):
    recognizer = sr.Recognizer()
    print("Listening...\n")

    with sr.Microphone() as source:
        # Adjust the recognizer sensitivity to ambient noise
        recognizer.adjust_for_ambient_noise(source)
        audio_data = recognizer.record(source, duration=duration)

    # Convert audio data to numpy array
    audio = np.frombuffer(audio_data.get_raw_data(), np.float32)
    
    # Define the path where the audio file will be saved
    output_dir = "Testing/audio files"  # Use forward slashes for compatibility
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
    
    # Full path to the output file
    full_filename = os.path.join(output_dir, filename)

    # Apply VAD filtering
    sf.write(full_filename, audio, int(audio_data.sample_rate), format='FLAC')
    if vad_filter(audio, threshold=VAD_THRESHOLD):
        # Save the audio to a FLAC file
        return full_filename
    else:
        print("No significant audio detected. \n")
        return None

# VAD filtering function
def vad_filter(audio, threshold=VAD_THRESHOLD):
    # Simple VAD logic: checks if the max amplitude exceeds the threshold
    return np.max(np.abs(audio)) > threshold

# Function to transcribe audio using Faster-Whisper
async def transcribe_audio(filename, language="en"):
    # Initialize Faster-Whisper model with low-memory usage
    model = WhisperModel("tiny", device="cpu", compute_type="int8")  # int8 for CPU efficiency

    segments, info = model.transcribe(filename, language=language)
    transcribed_text = ""
    for segment in segments:
        transcribed_text += segment.text + " "

    # print("Transcribed Text:", transcribed_text)
    return transcribed_text

# Main function to handle audio capture and processing
async def main():
    # Capture audio with VAD
    audio_file = await capture_audio()

    if audio_file:
        # Transcribe audio asynchronously using Faster-Whisper
        start = time.time()
        await transcribe_audio(audio_file)
        end = time.time()
        print("Time taken: ", end-start)

# Run the main function in an asyncio event loop
if __name__ == "__main__":
    asyncio.run(main())
