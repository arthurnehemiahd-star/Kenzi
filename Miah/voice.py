"""Voice in, voice out, and speaker recognition.

Three separate things live here, deliberately kept apart:

  1. Speech-to-text       — Whisper, listens to whoever is talking
  2. Text-to-speech       — XTTS, always speaks in the OWNER's cloned voice
  3. Speaker embeddings   — Resemblyzer, recognizes HER voice for login

The original file conflated (1) and (3) inside the transcribe endpoint,
which meant every single transcription paid the cost of a Resemblyzer
pass even during setup — and any Resemblyzer error was swallowed by a
bare `except: pass`. Here they're separate functions so the endpoint can
decide what to run and when.
"""

import os
import subprocess
import tempfile

import numpy as np

from config import REFERENCE_CLIP_PATH

# ----------------------------------------------------------------------
# Whisper (speech to text)
# ----------------------------------------------------------------------
# Loading the model is expensive (seconds, on CPU), so we load it once
# per process and reuse it. The original loaded it on every request.
_whisper_model = None


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        _whisper_model = whisper.load_model("base")
    return _whisper_model


def transcribe_file(path):
    """Transcribe an audio file (any ffmpeg-readable format) to text.
    Returns a plain string, possibly empty."""
    model = get_whisper_model()
    result = model.transcribe(path)
    return (result.get("text") or "").strip()


# ----------------------------------------------------------------------
# XTTS (text to speech, cloned voice)
# ----------------------------------------------------------------------
# Same reasoning as Whisper — XTTS is multi-GB and slow to construct.
# We build it once and reuse it. The original constructed a fresh TTS()
# object on every /api/speak call, which meant a multi-second model load
# per reply. That's the single biggest performance fix in this refactor.
_tts_model = None


def get_tts_model():
    global _tts_model
    if _tts_model is None:
        from TTS.api import TTS
        _tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    return _tts_model


def synthesize_speech(text, language="en"):
    """Render `text` in the owner's cloned voice. Returns the path to a
    temp WAV file. Caller is responsible for deleting it after sending."""
    if not os.path.exists(REFERENCE_CLIP_PATH):
        raise FileNotFoundError(
            "No voice enrolled yet — enroll a reference clip first."
        )

    tts = get_tts_model()
    fd, output_path = tempfile.mkstemp(suffix=".wav", prefix="miah_reply_")
    os.close(fd)

    tts.tts_to_file(
        text=text,
        speaker_wav=REFERENCE_CLIP_PATH,
        language=language,
        file_path=output_path,
    )
    return output_path


# ----------------------------------------------------------------------
# Speaker embeddings (Resemblyzer)
# ----------------------------------------------------------------------
_voice_encoder = None


def get_voice_encoder():
    global _voice_encoder
    if _voice_encoder is None:
        from resemblyzer import VoiceEncoder
        _voice_encoder = VoiceEncoder()
    return _voice_encoder


def compute_embedding_from_file(path):
    """Convert any audio file to a clean mono 16k WAV, then return a
    speaker embedding as a plain list (JSON-safe).

    The intermediate WAV is written next to the input so ffmpeg can't
    collide with a same-named file in /tmp; it's deleted in `finally`.
    """
    from resemblyzer import preprocess_wav

    wav_path = path + ".resemblyzer.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", path, "-ar", "16000", "-ac", "1", wav_path],
        check=True,
        capture_output=True,
    )
    try:
        wav = preprocess_wav(wav_path)
        embedding = get_voice_encoder().embed_utterance(wav)
        return embedding.tolist()
    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)


def cosine_similarity(a, b):
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def convert_to_reference_wav(input_path, output_path):
    """Convert a browser-recorded clip (webm/mp4/ogg, varies by device)
    into the clean mono 22.05k WAV that XTTS expects as a reference."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-ar", "22050", "-ac", "1", output_path],
        check=True,
        capture_output=True,
    )