import asyncio
from Models.faster_whisper_stt_tiny import capture_audio, transcribe_audio

async def capture_and_transcribe_audio():
    """Capture audio and return transcribed text."""
    audio_file = await capture_audio()
    if not audio_file:
        return None

    transcribed_text = await transcribe_audio(audio_file, language="en")
    if 'watching' in transcribed_text or "Let's go" in transcribed_text:
        return None
    
    return transcribed_text
