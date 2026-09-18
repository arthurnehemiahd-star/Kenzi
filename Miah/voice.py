
"""
MIAH — voice input and voice output.

Three separate capabilities:

    1. Speech-to-text
       Whisper listens to the user.

    2. Text-to-speech
       XTTS speaks using MIAH's enrolled owner voice.

    3. Speaker embeddings
       Resemblyzer support retained for compatibility with the
       existing voice-profile code.

Heavy AI libraries are loaded only when the related feature
is actually used. This keeps Flask/Gunicorn startup fast.
"""

import os
import subprocess
import tempfile


# =============================================================================
# CONFIG
# =============================================================================

from config import REFERENCE_CLIP_PATH


# =============================================================================
# WHISPER — SPEECH TO TEXT
# =============================================================================

_whisper_model = None


def get_whisper_model():
    """
    Load Whisper only when speech-to-text is actually requested.
    """

    global _whisper_model

    if _whisper_model is None:

        print("[voice] Loading Whisper model...")

        import whisper

        _whisper_model = whisper.load_model("base")

        print("[voice] Whisper model loaded.")

    return _whisper_model


def transcribe_file(path):
    """
    Transcribe an audio file into plain text.
    """

    model = get_whisper_model()

    result = model.transcribe(path)

    return (
        result.get("text")
        or ""
    ).strip()


# =============================================================================
# XTTS — MIAH'S SPEAKING VOICE
# =============================================================================

_tts_model = None


def get_tts_model():
    """
    Load XTTS only when MIAH actually needs to speak.
    """

    global _tts_model

    if _tts_model is None:

        print("[voice] Loading XTTS model...")

        from TTS.api import TTS

        _tts_model = TTS(
            "tts_models/multilingual/"
            "multi-dataset/xtts_v2"
        )

        print("[voice] XTTS model loaded.")

    return _tts_model


def ensure_reference_voice():
    """
    Make sure MIAH's owner voice exists locally.

    Render's filesystem is temporary, so the saved voice is restored
    from Supabase when the local WAV is missing.
    """

    from supabase_store import (
        restore_owner_voice_if_needed,
    )

    return restore_owner_voice_if_needed(
        REFERENCE_CLIP_PATH
    )


def synthesize_speech(
    text,
    language="en",
):
    """
    Speak text using MIAH's enrolled owner voice.

    Returns the path to a temporary WAV file.
    """

    if not ensure_reference_voice():

        raise FileNotFoundError(
            "No MIAH voice is enrolled yet."
        )

    tts = get_tts_model()

    fd, output_path = tempfile.mkstemp(
        suffix=".wav",
        prefix="miah_reply_",
    )

    os.close(fd)

    try:

        tts.tts_to_file(
            text=text,
            speaker_wav=REFERENCE_CLIP_PATH,
            language=language,
            file_path=output_path,
        )

    except Exception:

        if os.path.exists(output_path):
            os.unlink(output_path)

        raise

    return output_path


# =============================================================================
# SPEAKER EMBEDDINGS
# =============================================================================

_voice_encoder = None


def get_voice_encoder():
    """
    Load Resemblyzer only when speaker recognition is needed.
    """

    global _voice_encoder

    if _voice_encoder is None:

        print("[voice] Loading Resemblyzer...")

        from resemblyzer import VoiceEncoder

        _voice_encoder = VoiceEncoder()

        print("[voice] Resemblyzer loaded.")

    return _voice_encoder


def compute_embedding_from_file(path):
    """
    Convert browser audio to clean mono 16 kHz WAV and compute
    a speaker embedding.
    """

    from resemblyzer import preprocess_wav

    wav_path = (
        path
        + ".resemblyzer.wav"
    )

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            path,
            "-ar",
            "16000",
            "-ac",
            "1",
            wav_path,
        ],
        check=True,
        capture_output=True,
    )

    try:

        wav = preprocess_wav(
            wav_path
        )

        embedding = (
            get_voice_encoder()
            .embed_utterance(wav)
        )

        return embedding.tolist()

    finally:

        if os.path.exists(wav_path):
            os.unlink(wav_path)


def cosine_similarity(
    a,
    b,
):
    """
    Calculate cosine similarity between two speaker embeddings.
    """

    import numpy as np

    a = np.asarray(
        a,
        dtype=np.float32,
    )

    b = np.asarray(
        b,
        dtype=np.float32,
    )

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    ) + 1e-8

    return float(
        np.dot(a, b)
        / denominator
    )


# =============================================================================
# AUDIO CONVERSION
# =============================================================================

def convert_to_reference_wav(
    input_path,
    output_path,
):
    """
    Convert browser-recorded audio into the WAV format
    expected by XTTS.
    """

    parent = os.path.dirname(
        output_path
    )

    if parent:
        os.makedirs(
            parent,
            exist_ok=True,
        )

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-ar",
            "22050",
            "-ac",
            "1",
            output_path,
        ],
        check=True,
        capture_output=True,
    )

