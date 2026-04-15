"""
Kokoro-82M STREAMING TTS for the voice assistant pipeline.
Plays audio chunks in real-time as they're generated — no file I/O delay.
GPU (RTX 5070 Ti): ~200x real-time | CPU: ~2.5x real-time
First chunk latency: ~200ms on GPU
"""

import torch
import numpy as np
import sys
import time

# Lazy-loaded pipeline
_pipeline = None


def _get_pipeline(lang_code='a'):
    """Lazy-initialize the Kokoro pipeline (loads model on first call)."""
    global _pipeline
    if _pipeline is None:
        try:
            from kokoro import KPipeline
        except ImportError:
            print("❌ Kokoro not installed. Install with:")
            print("   pip install kokoro>=0.9.2 sounddevice")
            sys.exit(1)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🎤 Kokoro TTS initializing on: {device}")
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem / 1e9
            print(f"   GPU: {gpu} ({vram:.1f}GB VRAM)")

        _pipeline = KPipeline(lang_code=lang_code)
        print("✅ Kokoro TTS ready (streaming mode)")
    return _pipeline


def stream_speak(text, voice='af_heart', speed=1.0, lang_code='a'):
    """
    Stream-synthesize text and play each chunk immediately via sounddevice.
    Each sentence is played as soon as it's generated — true real-time TTS.

    Parameters:
        text (str): Text to speak.
        voice (str): Voice preset (e.g. 'af_heart', 'af_nicole', 'am_adam').
        speed (float): Speech speed multiplier (1.0 = normal).
        lang_code (str): 'a' = US English, 'b' = UK English.
    """
    import sounddevice as sd

    pipeline = _get_pipeline(lang_code)
    start_time = time.time()

    generator = pipeline(
        text,
        voice=voice,
        speed=speed,
        split_pattern=r'[.!?]+'  # Sentence-level chunks for natural streaming
    )

    first_chunk = True
    for i, (gs, ps, audio_chunk) in enumerate(generator):
        if first_chunk:
            latency = time.time() - start_time
            print(f"   ⚡ First chunk latency: {latency*1000:.0f}ms")
            first_chunk = False

        # Play chunk IMMEDIATELY — no file save
        sd.play(audio_chunk, samplerate=24000)
        sd.wait()  # Block until this chunk finishes playing

    total = time.time() - start_time
    print(f"   🎵 Stream complete: {total:.2f}s")


def synthesize(text, output_file=None, voice='af_heart', speed=1.0, lang_code='a'):
    """
    Generate speech and save to file (non-streaming fallback).
    Used when file output is needed instead of direct playback.

    Parameters:
        text (str): Text to synthesize.
        output_file (str): Path to save the WAV file.
        voice (str): Voice preset.
        speed (float): Speech speed multiplier.
        lang_code (str): Language code.

    Returns:
        str: Path to the generated WAV audio file.
    """
    import soundfile as sf
    import os

    pipeline = _get_pipeline(lang_code)

    if output_file is None:
        output_dir = os.path.join("Testing", "audio files")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "kokoro_tts_response.wav")

    generator = pipeline(
        text,
        voice=voice,
        speed=speed,
        split_pattern=r'[.!?]+'
    )

    audio_chunks = []
    for i, (gs, ps, audio_chunk) in enumerate(generator):
        audio_chunks.append(audio_chunk)

    if not audio_chunks:
        print("⚠️ Kokoro generated no audio")
        return output_file

    audio = np.concatenate(audio_chunks)
    sf.write(output_file, audio, 24000)
    return output_file


if __name__ == "__main__":
    text = "Hello! This is Kokoro streaming TTS. Each sentence plays immediately. No waiting for the full response."
    start = time.time()
    stream_speak(text, voice='af_heart')
    print(f"\n✅ Total time: {time.time() - start:.2f}s")
