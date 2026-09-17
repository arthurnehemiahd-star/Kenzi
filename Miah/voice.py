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

The MIAH owner voice is automatically restored from Supabase when
the Render filesystem no longer has the local cached WAV.
"""

import os
import subprocess
import tempfile

import numpy as np

from config import REFERENCE_CLIP_PATH

from supabase_store import (
    restore_owner_voice_if_needed,
)


# =============================================================================
# WHISPER — SPEECH TO TEXT
# =============================================================================

_whisper_model = None


def get_whisper_model():
    global _whisper_model

    if _whisper_model is None:

        import whisper

        _whisper_model = (
            whisper.load_model("base")
        )

    return _whisper_model


def transcribe_file(path):
    """
    Transcribe an audio file into plain text.
    """

    model = get_whisper_model()

    result = model.transcribe(
        path
    )

    return (
        result.get("text")
        or ""
    ).strip()


# =============================================================================
# XTTS — MIAH'S SPEAKING VOICE
# =============================================================================

_tts_model = None


def get_tts_model():
    global _tts_model

    if _tts_model is None:

        from TTS.api import TTS

        _tts_model = TTS(
            "tts_models/multilingual/"
            "multi-dataset/xtts_v2"
        )

    return _tts_model


def ensure_reference_voice():
    """
    Make sure MIAH's owner voice exists locally.

    On Render Free, the local filesystem can disappear after a restart.
    When that happens, restore the WAV from Supabase.
    """

    return restore_owner_voice_if_needed(
        REFERENCE_CLIP_PATH
    )


def synthesize_speech(
    text,
    language="en",
):
    """
    Speak `text` using MIAH's enrolled voice.

    Returns the path to a temporary WAV file.
    """

    if not ensure_reference_voice():

        raise FileNotFoundError(
            "No MIAH voice is enrolled yet."
        )

    tts = get_tts_model()

    fd, output_path = (
        tempfile.mkstemp(
            suffix=".wav",
            prefix="miah_reply_",
        )
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

        if os.path.exists(
            output_path
        ):
            os.unlink(
                output_path
            )

        raise

    return output_path


# =============================================================================
# SPEAKER EMBEDDINGS
# =============================================================================

_voice_encoder = None


def get_voice_encoder():
    global _voice_encoder

    if _voice_encoder is None:

        from resemblyzer import VoiceEncoder

        _voice_encoder = (
            VoiceEncoder()
        )

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

        if os.path.exists(
            wav_path
        ):
            os.unlink(
                wav_path
            )


def cosine_similarity(
    a,
    b,
):
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
    Convert browser-recorded audio into the WAV format XTTS expects.
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
