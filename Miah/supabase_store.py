"""
MIAH — Supabase persistence layer.

Stores:
    - MIAH JSON state in the Supabase `miah_state` table
    - MIAH owner voice in the private `miah-private` Storage bucket

The Supabase secret key is used only on the backend.
It must never be exposed to the frontend.
"""

import os
import tempfile
from pathlib import Path


SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_SECRET_KEY = os.environ.get("SUPABASE_SECRET_KEY", "").strip()

STATE_TABLE = "miah_state"
STATE_ROW_ID = "miah"

VOICE_BUCKET = "miah-private"
VOICE_OBJECT = "owner_voice.wav"


_client = None


def supabase_configured():
    return bool(SUPABASE_URL and SUPABASE_SECRET_KEY)


def get_client():
    """
    Create the Supabase client lazily.

    We deliberately do not create it during module import so that
    the application can still start locally when Supabase variables
    are not present.
    """
    global _client

    if not supabase_configured():
        return None

    if _client is None:
        from supabase import create_client

        _client = create_client(
            SUPABASE_URL,
            SUPABASE_SECRET_KEY,
        )

    return _client


# =============================================================================
# DATABASE
# =============================================================================

def load_remote_state():
    """
    Load MIAH's single state row from Supabase.

    Returns:
        dict | None

    None means either no row exists yet or Supabase is not configured.
    Exceptions are allowed to propagate so callers can distinguish a
    real database problem from an empty database.
    """

    client = get_client()

    if client is None:
        return None

    response = (
        client
        .table(STATE_TABLE)
        .select("data")
        .eq("id", STATE_ROW_ID)
        .maybe_single()
        .execute()
    )

    if not response.data:
        return None

    data = response.data.get("data")

    if not isinstance(data, dict):
        return None

    return data


def save_remote_state(data):
    """
    Save MIAH's complete JSON state to Supabase.
    """

    client = get_client()

    if client is None:
        return False

    from datetime import datetime, timezone

    payload = {
        "id": STATE_ROW_ID,
        "data": data,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    (
        client
        .table(STATE_TABLE)
        .upsert(payload)
        .execute()
    )

    return True


# =============================================================================
# OWNER VOICE STORAGE
# =============================================================================

def upload_owner_voice(local_path):
    """
    Upload MIAH's enrolled owner voice to the private Supabase bucket.

    The fixed object name lets us replace the old voice when recording
    a new one.
    """

    client = get_client()

    if client is None:
        return False

    local_path = Path(local_path)

    if not local_path.exists():
        raise FileNotFoundError(
            f"Voice file does not exist: {local_path}"
        )

    with open(local_path, "rb") as voice_file:
        (
            client
            .storage
            .from_(VOICE_BUCKET)
            .upload(
                file=voice_file,
                path=VOICE_OBJECT,
                file_options={
                    "content-type": "audio/wav",
                    "upsert": "true",
                },
            )
        )

    return True


def download_owner_voice(local_path):
    """
    Download MIAH's enrolled owner voice from private Supabase Storage.

    Writes through a temporary file first so a failed download cannot
    leave a partially written WAV file behind.
    """

    client = get_client()

    if client is None:
        return False

    destination = Path(local_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    audio_bytes = (
        client
        .storage
        .from_(VOICE_BUCKET)
        .download(VOICE_OBJECT)
    )

    if not audio_bytes:
        return False

    fd, temp_name = tempfile.mkstemp(
        suffix=".wav",
        prefix="miah_voice_",
        dir=str(destination.parent),
    )

    try:
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(audio_bytes)

        os.replace(
            temp_name,
            destination,
        )

        return True

    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass

        raise


def restore_owner_voice_if_needed(local_path):
    """
    Restore the persistent MIAH voice from Supabase if the local
    Render filesystem no longer contains it.

    Returns True when a usable local voice exists afterward.
    """

    destination = Path(local_path)

    if destination.exists() and destination.stat().st_size > 0:
        return True

    try:
        return download_owner_voice(destination)
    except Exception as exc:
        print(f"[supabase] Could not restore owner voice: {exc}")
        return False
