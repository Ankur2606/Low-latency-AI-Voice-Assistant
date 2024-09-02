import sounddevice as sd
import numpy as np
import soundfile as sf
import asyncio
from faster_whisper import WhisperModel
import time
import os
# VAD Parameters
VAD_THRESHOLD = 0.01  # Adjust based on your environment

# Asynchronously capture and save audio with VAD filtering
async def capture_audio(duration=5, filename="stt_transcribe.flac"):
    fs = 16000  # Sampling rate
    print("Listening...\n")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.float32)
    sd.wait()  # Wait until recording is finished

    # Define the path where the audio file will be saved
    output_dir = "Testing/audio files"  # Use forward slashes for compatibility
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
    
    # Full path to the output file
    full_filename = os.path.join(output_dir, filename)

    # Apply VAD filtering
    if vad_filter(audio, threshold=VAD_THRESHOLD):
        # Save the audio to a FLAC file
        sf.write(full_filename, audio, fs, format='FLAC')
        return full_filename
    else:
        print("No significant audio detected. \n")
        return None

# VAD filtering function
def vad_filter(audio, threshold=VAD_THRESHOLD):
    # Simple VAD logic: checks if the max amplitude exceeds the threshold
    return np.max(np.abs(audio)) > threshold

# Function to transcribe audio using Faster-Whisper
async def transcribe_audio(filename, language = "en"):
    # Initialize Faster-Whisper model with low-memory usage
    model = WhisperModel("tiny", device="cpu", compute_type="int8")

    segments, info = model.transcribe(filename, language = language)
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