"""Central config. Every env var MIAH reads lives here, in one place."""

import os
import secrets

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "miah_data")

# Files inside DATA_DIR
REFERENCE_CLIP_PATH = os.path.join(DATA_DIR, "owner_voice.wav")
DB_PATH = os.path.join(DATA_DIR, "miah_db.json")
MUSIC_DIR = os.path.join(DATA_DIR, "music")
SPOTIFY_TOKEN_CACHE = os.path.join(DATA_DIR, "spotify_tokens.json")

ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}

# --- LLM (Hugging Face Inference Providers, OpenAI-compatible) ---
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL = os.environ.get("HF_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN")
LLM_TIMEOUT_SECONDS = 60
LLM_MAX_TOKENS = 400

# Hard cap on tool-call round-trips per user turn. Prevents a misbehaving
# model from looping forever inside /api/chat. Was missing in the original.
MAX_TOOL_ITERATIONS = 6

# --- Flask session ---
# NOTE: original defaulted to a per-process random key, which silently
# logs everyone out on every restart. We keep that fallback so first-run
# works, but warn loudly in README to set SECRET_KEY for anything real.
SECRET_KEY = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
SESSION_LIFETIME_DAYS = 30

# --- Voice recognition (passive voice-login learning) ---
VOICE_LOGIN_MIN_DAYS = float(os.environ.get("VOICE_LOGIN_MIN_DAYS", "4"))
VOICE_LOGIN_MIN_SAMPLES = 8
VOICE_PROFILE_MAX_SAMPLES = 60
VOICE_MATCH_THRESHOLD = 0.75

# --- Conversation memory ---
CONVERSATION_KEY = "main"
SUMMARY_TRIGGER_COUNT = 40
KEEP_RECENT_MESSAGES = 20

# --- Spotify ---
SPOTIFY_AUTH_URL = "https://accounts.spotify.com/authorize"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API_BASE = "https://api.spotify.com/v1"
SPOTIFY_SCOPES = "user-library-modify user-library-read"
SPOTIFY_CLIENT_ID = os.environ.get("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET", "")
SPOTIFY_REDIRECT_URI = os.environ.get("SPOTIFY_REDIRECT_URI", "")


def ensure_dirs():
    """Create the data/music folders if they don't exist yet."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MUSIC_DIR, exist_ok=True)