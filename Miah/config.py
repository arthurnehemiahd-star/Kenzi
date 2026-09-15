"""
MIAH — Configuration
--------------------

All configuration for the MIAH assistant lives here.

For Render Free:
    Data is stored inside the application's miah_data/ directory.

For a Render paid plan with a Persistent Disk:
    This file can later be changed to use /var/data/miah_data.
"""

import os
import secrets


# =============================================================================
# BASE DIRECTORIES
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Render Free does not provide Persistent Disks.
# Therefore we keep MIAH's data inside the application directory for now.
DATA_DIR = os.path.join(BASE_DIR, "miah_data")


# =============================================================================
# DATA FILES / DIRECTORIES
# =============================================================================

# MIAH's enrolled speaking voice.
# This is the voice MIAH uses when generating speech.
REFERENCE_CLIP_PATH = os.path.join(
    DATA_DIR,
    "owner_voice.wav",
)

# Main MIAH JSON database.
DB_PATH = os.path.join(
    DATA_DIR,
    "miah_db.json",
)

# Local music storage.
MUSIC_DIR = os.path.join(
    DATA_DIR,
    "music",
)

# Spotify token cache.
SPOTIFY_TOKEN_CACHE = os.path.join(
    DATA_DIR,
    "spotify_tokens.json",
)


# =============================================================================
# SECURITY
# =============================================================================

# IMPORTANT:
# Set SECRET_KEY in Render Environment Variables for production.
#
# Example:
# SECRET_KEY=<long-random-secret>
#
# The fallback is useful for local development, but changing the fallback
# between restarts would invalidate signed authentication tokens.
SECRET_KEY = os.environ.get(
    "SECRET_KEY",
    "miah-development-secret-change-this",
)


# =============================================================================
# SESSION / AUTHENTICATION
# =============================================================================

# How long signed login tokens remain valid.
SESSION_LIFETIME_DAYS = int(
    os.environ.get(
        "SESSION_LIFETIME_DAYS",
        "7",
    )
)


# =============================================================================
# AI MODEL
# =============================================================================

HF_MODEL = os.environ.get(
    "HF_MODEL",
    "meta-llama/Llama-3.3-70B-Instruct",
)


# Maximum number of tool-use loops MIAH can perform for one request.
MAX_TOOL_ITERATIONS = int(
    os.environ.get(
        "MAX_TOOL_ITERATIONS",
        "5",
    )
)


# =============================================================================
# OPTIONAL API CONFIGURATION
# =============================================================================

# Hugging Face token.
# Keep the actual token in Render Environment Variables.
HF_TOKEN = os.environ.get("HF_TOKEN", "")


# Anthropic/API-related variables may be used by other modules.
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# OpenAI-compatible API key, if used by llm.py.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Optional API base URL.
OPENAI_BASE_URL = os.environ.get(
    "OPENAI_BASE_URL",
    "",
)


# =============================================================================
# FRONTEND
# =============================================================================

FRONTEND_URL = os.environ.get(
    "FRONTEND_URL",
    "https://kenzilynn.vercel.app",
)


# =============================================================================
# MIAH DATA DIRECTORY SETUP
# =============================================================================

def ensure_dirs():
    """
    Create all directories MIAH needs.

    This version intentionally creates directories only inside BASE_DIR,
    because Render Free does not provide /var/data.
    """

    os.makedirs(
        DATA_DIR,
        exist_ok=True,
    )

    os.makedirs(
        MUSIC_DIR,
        exist_ok=True,
    )


# =============================================================================
# INITIAL SETUP
# =============================================================================

# Make sure the required directories exist whenever config.py is imported.
ensure_dirs()
