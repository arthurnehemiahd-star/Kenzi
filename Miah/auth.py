"""Setup (voice enrollment + password creation) and login.

Two changes from the original, both security-relevant:

  - Passwords are hashed with PBKDF2-HMAC-SHA256 + a per-user random
    salt, not a bare SHA-256. A bare SHA-256 of a short password is
    trivially reversible with a rainbow table; PBKDF2 with 200k
    iterations is the standard cheap fix and needs no extra dependency
    (hashlib ships it).
  - Voice-login compares against the profile mean, and the threshold
    logic is unchanged — but it now refuses to even try if the profile
    is empty, rather than crashing on np.mean of an empty list.
"""

import hashlib
import hmac
import os
import secrets
import tempfile

from werkzeug.utils import secure_filename

from config import VOICE_MATCH_THRESHOLD
from db import add_voice_sample, get_voice_login_readiness, load_db, save_db
from voice import compute_embedding_from_file, convert_to_reference_wav


PBKDF2_ITERATIONS = 200_000


# ----------------------------------------------------------------------
# Password hashing
# ----------------------------------------------------------------------
def hash_password(password, salt=None):
    """Return (salt_hex, hash_hex). If salt is None, generate a new one."""
    if salt is None:
        salt = secrets.token_bytes(16)
    elif isinstance(salt, str):
        salt = bytes.fromhex(salt)

    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    )
    return salt.hex(), dk.hex()


def verify_password(password, salt_hex, expected_hash_hex):
    _, computed = hash_password(password, salt_hex)
    return hmac.compare_digest(computed, expected_hash_hex)


# ----------------------------------------------------------------------
# First-run setup
# ----------------------------------------------------------------------
def set_owner_voice(file_storage, reference_path):
    """Save the uploaded voice clip, convert it to XTTS reference WAV,
    and write it to reference_path. Raises on ffmpeg failure."""
    original_name = secure_filename(file_storage.filename or "voice.webm")
    extension = os.path.splitext(original_name)[1] or ".webm"

    with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as tmp:
        file_storage.save(tmp.name)
        tmp_path = tmp.name

    try:
        convert_to_reference_wav(tmp_path, reference_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def set_password(password):
    """Set the owner's password. Only works if none is set yet.
    Returns (ok, error_message)."""
    if len(password) < 4:
        return False, "Password must be at least 4 characters."

    db = load_db()
    if db.get("password_hash"):
        return False, "A password is already set."

    salt_hex, hash_hex = hash_password(password)
    db["password_salt"] = salt_hex
    db["password_hash"] = hash_hex
    save_db(db)
    return True, None


def check_password(password):
    db = load_db()
    salt_hex = db.get("password_salt")
    expected = db.get("password_hash")
    if not salt_hex or not expected:
        return False
    return verify_password(password, salt_hex, expected)


# ----------------------------------------------------------------------
# Voice login
# ----------------------------------------------------------------------
def try_voice_login(file_storage):
    """Attempt voice login. Returns (ok, error_message, ready).

    `ready` is True when MIAH has heard the owner enough over enough days
    to make a decision (whether or not the decision was a match) — the
    frontend uses this to decide whether to show the "just say something"
    button at all.
    """
    db = load_db()
    ready, days_remaining = get_voice_login_readiness(db)
    if not ready:
        if days_remaining is not None:
            return False, f"Still learning your voice — about {days_remaining:.1f} more day(s).", False
        return False, "MIAH hasn't learned your voice yet.", False

    original_name = secure_filename(file_storage.filename or "clip.webm")
    extension = os.path.splitext(original_name)[1] or ".webm"

    with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as tmp:
        file_storage.save(tmp.name)
        tmp_path = tmp.name

    try:
        live_embedding = compute_embedding_from_file(tmp_path)
    except Exception:
        return False, "Couldn't process that recording.", True
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    embeddings = [e["vector"] for e in db["her_voice_profile"]["embeddings"]]
    if not embeddings:
        return False, "Voice profile is empty.", True

    # Mean of all stored embeddings = her average voice fingerprint.
    profile_vector = [
        float(sum(col)) / len(col)
        for col in zip(*embeddings)
    ]

    from voice import cosine_similarity
    score = cosine_similarity(profile_vector, live_embedding)

    if score >= VOICE_MATCH_THRESHOLD:
        return True, None, True
    return False, "Didn't recognize your voice — try your password.", True


def record_voice_sample(audio_path):
    """Best-effort: add one embedding to her voice profile. Used after
    a successful login during voice mode, so the profile grows over
    time. Never raises — a failure here should never break a turn."""
    try:
        db = load_db()
        if not db.get("password_hash"):
            return  # don't learn before setup is done
        embedding = compute_embedding_from_file(audio_path)
        add_voice_sample(db, embedding)
        save_db(db)
    except Exception:
        pass