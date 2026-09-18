"""Central configuration for MIAH.

Storage:
    Local development:
        Miah/miah_data/

    Production:
        MIAH's persistent state is stored in Supabase.
        The local miah_data directory is only an ephemeral cache.

Voice:
    The enrolled voice is MIAH's SPEAKING VOICE.
    It is not used for login.

    The user's microphone is used only for speech-to-text.
"""

import os
import secrets


# ----------------------------------------------------------------------
# Base / local cache storage
# ----------------------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# This is only a local cache.
# Do NOT set this to /var/data on Render Free.
DATA_DIR = os.environ.get(
    "MIAH_DATA_DIR",
    os.path.join(
        BASE_DIR,
        "miah_data",
    ),
)


# ----------------------------------------------------------------------
# Supabase
# ----------------------------------------------------------------------

SUPABASE_URL = os.environ.get(
    "SUPABASE_URL",
    "",
).strip()

SUPABASE_SECRET_KEY = os.environ.get(
    "SUPABASE_SECRET_KEY",
    "",
).strip()


# ----------------------------------------------------------------------
# MIAH data files
# ----------------------------------------------------------------------

# Local cached copy of MIAH's speaking voice.
REFERENCE_CLIP_PATH = os.path.join(
    DATA_DIR,
    "owner_voice.wav",
)

# Local cache of database state.
# The real persistent copy is in Supabase.
DB_PATH = os.path.join(
    DATA_DIR,
    "miah_db.json",
)

# Local music files.
MUSIC_DIR = os.path.join(
    DATA_DIR,
    "music",
)

# Local Spotify OAuth token cache.
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

HF_API_URL = (
    "https://router.huggingface.co/v1/chat/completions"
)

HF_MODEL = os.environ.get(
    "HF_MODEL",
    "meta-llama/Llama-3.3-70B-Instruct",
)

HF_TOKEN = os.environ.get(
    "HF_TOKEN",
    "",
).strip()

LLM_TIMEOUT_SECONDS = 60

LLM_MAX_TOKENS = 400

# Maximum number of tool-call round trips for one request.
MAX_TOOL_ITERATIONS = 6


# ----------------------------------------------------------------------
# Flask / API authentication
# ----------------------------------------------------------------------

SECRET_KEY = (
    os.environ.get(
        "SECRET_KEY",
        "",
    ).strip()
    or secrets.token_hex(32)
)

SESSION_LIFETIME_DAYS = 30


# ----------------------------------------------------------------------
# Voice authentication compatibility settings
# ----------------------------------------------------------------------

# Normal MIAH login is PASSWORD ONLY.

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
).strip()

SPOTIFY_CLIENT_SECRET = os.environ.get(
    "SPOTIFY_CLIENT_SECRET",
    "",
).strip()

SPOTIFY_REDIRECT_URI = os.environ.get(
    "SPOTIFY_REDIRECT_URI",
    "",
).strip()


# ----------------------------------------------------------------------
# Directory initialization + remote voice restore
# ----------------------------------------------------------------------

def ensure_dirs():
    """Create local cache directories and restore MIAH's voice."""

    os.makedirs(
        DATA_DIR,
        exist_ok=True,
    )

    os.makedirs(
        MUSIC_DIR,
        exist_ok=True,
    )

    # MIAH's voice is persistent in Supabase.
    #
    # After every Render restart, the local cache may disappear.
    # Restore the saved WAV so XTTS and /api/status can use it.
    try:
        from supabase_store import restore_owner_voice_if_needed

        restored = restore_owner_voice_if_needed(
            REFERENCE_CLIP_PATH
        )

        if restored:
            print(
                "[miah] MIAH's speaking voice restored from Supabase."
            )
        else:
            print(
                "[miah] No saved MIAH voice was restored."
            )

    except Exception as exc:
        print(
            f"[miah] Voice restore skipped: {exc}"
        )
