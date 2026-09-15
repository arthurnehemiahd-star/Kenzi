"""Central config for MIAH.

Every environment variable MIAH reads lives here, in one place.

Storage:
    Local development:
        Miah/miah_data/

    Render production:
        Set:
            MIAH_DATA_DIR=/var/data/miah_data

        and mount a Render Persistent Disk at:
            /var/data

MIAH's enrolled voice is its SPEAKING VOICE.
It is not used for login.

The user's microphone is used only for speech-to-text.
"""

import os
import secrets


# ----------------------------------------------------------------------
# Base / persistent storage
# ----------------------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# On Render, this should point inside the Persistent Disk.
#
# Render environment variable:
#
#     MIAH_DATA_DIR=/var/data/miah_data
#
# When the variable is not set, local development uses:
#
#     Miah/miah_data/
#
DATA_DIR = os.environ.get(
    "MIAH_DATA_DIR",
    os.path.join(
        BASE_DIR,
        "miah_data",
    ),
)


# ----------------------------------------------------------------------
# MIAH data files
# ----------------------------------------------------------------------

# Saved reference recording for MIAH's speaking voice.
REFERENCE_CLIP_PATH = os.path.join(
    DATA_DIR,
    "owner_voice.wav",
)

# Conversation history, password hash, memory, etc.
DB_PATH = os.path.join(
    DATA_DIR,
    "miah_db.json",
)

# Local music files.
MUSIC_DIR = os.path.join(
    DATA_DIR,
    "music",
)

# Spotify OAuth token cache.
SPOTIFY_TOKEN_CACHE = os.path.join(
    DATA_DIR,
    "spotify_tokens.json",
)


# ----------------------------------------------------------------------
# Audio
# ----------------------------------------------------------------------

ALLOWED_AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".m4a",
    ".ogg",
    ".flac",
}


# ----------------------------------------------------------------------
# LLM
# ----------------------------------------------------------------------
#
# Hugging Face Inference Providers
# OpenAI-compatible chat completions endpoint
#

HF_API_URL = (
    "https://router.huggingface.co/v1/chat/completions"
)

HF_MODEL = os.environ.get(
    "HF_MODEL",
    "meta-llama/Llama-3.3-70B-Instruct",
)

HF_TOKEN = os.environ.get(
    "HF_TOKEN",
)

LLM_TIMEOUT_SECONDS = 60

LLM_MAX_TOKENS = 400


# Maximum number of tool-call round trips for one
# user request. This prevents accidental infinite loops.
MAX_TOOL_ITERATIONS = 6


# ----------------------------------------------------------------------
# Flask session
# ----------------------------------------------------------------------
#
# SECRET_KEY MUST be set in Render for stable sessions.
#
# Example Render variable:
#
#     SECRET_KEY=<a long random secret>
#
# The random fallback is useful for local development, but a deployment
# should always have SECRET_KEY configured.
#

SECRET_KEY = (
    os.environ.get(
        "SECRET_KEY"
    )
    or secrets.token_hex(32)
)

SESSION_LIFETIME_DAYS = 30


# ----------------------------------------------------------------------
# Voice authentication settings
# ----------------------------------------------------------------------
#
# These values are retained for compatibility with the existing
# authentication module.
#
# IMPORTANT:
#
# The normal MIAH login flow is PASSWORD ONLY.
#
# MIAH's enrolled reference voice is its speaking voice.
#
# The microphone recordings made while talking to MIAH are NOT
# automatically added to the voice profile.
#

VOICE_LOGIN_MIN_DAYS = float(
    os.environ.get(
        "VOICE_LOGIN_MIN_DAYS",
        "4",
    )
)

VOICE_LOGIN_MIN_SAMPLES = 8

VOICE_PROFILE_MAX_SAMPLES = 60

VOICE_MATCH_THRESHOLD = 0.75


# ----------------------------------------------------------------------
# Conversation memory
# ----------------------------------------------------------------------

CONVERSATION_KEY = "main"

SUMMARY_TRIGGER_COUNT = 40

KEEP_RECENT_MESSAGES = 20


# ----------------------------------------------------------------------
# Spotify
# ----------------------------------------------------------------------

SPOTIFY_AUTH_URL = (
    "https://accounts.spotify.com/authorize"
)

SPOTIFY_TOKEN_URL = (
    "https://accounts.spotify.com/api/token"
)

SPOTIFY_API_BASE = (
    "https://api.spotify.com/v1"
)

SPOTIFY_SCOPES = (
    "user-library-modify "
    "user-library-read"
)

SPOTIFY_CLIENT_ID = os.environ.get(
    "SPOTIFY_CLIENT_ID",
    "",
)

SPOTIFY_CLIENT_SECRET = os.environ.get(
    "SPOTIFY_CLIENT_SECRET",
    "",
)

SPOTIFY_REDIRECT_URI = os.environ.get(
    "SPOTIFY_REDIRECT_URI",
    "",
)


# ----------------------------------------------------------------------
# Directory initialization
# ----------------------------------------------------------------------

def ensure_dirs():
    """Create all MIAH data directories if they do not exist."""

    os.makedirs(
        DATA_DIR,
        exist_ok=True,
    )

    os.makedirs(
        MUSIC_DIR,
        exist_ok=True,
    )
