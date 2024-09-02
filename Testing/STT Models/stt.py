import sounddevice as sd
import numpy as np
import soundfile as sf
import aiohttp
import asyncio
import time
import os
from dotenv import load_dotenv

load_dotenv()
# VAD Parameters
VAD_THRESHOLD = 0.5  # Adjust this threshold based on your environment

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

# Asynchronously send the audio file in chunks
async def send_audio_in_chunks(filename):
    HF_API_TOKEN = os.getenv("HF_API_KEY")
    # HF_API_URL = "https://api-inference.huggingface.co/models/openai/whisper-small"
    HF_API_URL = "https://api-inference.huggingface.co/models/openai/whisper-tiny"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

    async with aiohttp.ClientSession() as session:
        with open(filename, "rb") as f:
            # Send the file in chunks
            async with session.post(HF_API_URL, headers=headers, data=f) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'text' in data:
                        print("Transcribed Text:", data['text'])
                    else:
                        print("Error or different response format:", data)
                else:
                    print(f"Failed with status code: {response.status}")
                    print(await response.text())

# Main function to handle audio capture and processing
async def main():
    # Capture audio
    audio_file = await capture_audio()
    if audio_file:
        start  = time.time()
        # Send audio asynchronously
        await send_audio_in_chunks(audio_file)
        end = time.time()
        print("Time taken: ", end-start)
    else:
        print("No valid audio to process.")

# Run the main function in an asyncio event loop
asyncio.run(main())
