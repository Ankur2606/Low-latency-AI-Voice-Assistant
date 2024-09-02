import sounddevice as sd
import numpy as np
import soundfile as sf
import asyncio
from faster_whisper import WhisperModel

# VAD Parameters
VAD_THRESHOLD = 0.01  # Adjust based on your environment

# Asynchronously capture and save audio with VAD filtering
async def capture_audio(duration=5, filename="audio.flac"):
    fs = 16000  # Sampling rate
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.float32)
    sd.wait()  # Wait until recording is finished
    print("Recording complete.")

    # Apply VAD filtering
    if vad_filter(audio, threshold=VAD_THRESHOLD):
        # Save the audio to a FLAC file
        sf.write(filename, audio, fs, format='FLAC')
        return filename
    else:
        print("No significant audio detected.")
        return None

# VAD filtering function
def vad_filter(audio, threshold=VAD_THRESHOLD):
    # Simple VAD logic: checks if the max amplitude exceeds the threshold
    return np.max(np.abs(audio)) > threshold

# Function to transcribe audio using Faster-Whisper
async def transcribe_audio(filename):
    # Initialize Faster-Whisper model with medium size
    model = WhisperModel("medium", device="cpu", compute_type="int8")  # Using "int8" for CPU efficiency

    segments, info = model.transcribe(filename)
    transcribed_text = ""
    for segment in segments:
        transcribed_text += segment.text + " "

    print("Transcribed Text:", transcribed_text)
    return transcribed_text

# Main function to handle audio capture and processing
async def main():
    # Capture audio with VAD
    audio_file = await capture_audio()

    if audio_file:
        # Transcribe audio asynchronously using Faster-Whisper
        await transcribe_audio(audio_file)

# Run the main function in an asyncio event loop
asyncio.run(main())
