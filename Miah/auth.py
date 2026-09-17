"""
MIAH — authentication and first-run setup.

Handles:

    - Password creation
    - Password verification
    - MIAH speaking-voice enrollment
    - Legacy speaker-profile functions kept for compatibility

The password and voice are persisted through Supabase.
"""

import hashlib
import hmac
import os
import secrets
import tempfile

from werkzeug.utils import secure_filename

from config import VOICE_MATCH_THRESHOLD

from db import (
    add_voice_sample,
    get_voice_login_readiness,
    load_db,
    save_db,
)

from voice import (
    compute_embedding_from_file,
    convert_to_reference_wav,
)

from supabase_store import (
    upload_owner_voice,
)


PBKDF2_ITERATIONS = 200_000


# =============================================================================
# PASSWORD HASHING
# =============================================================================

def hash_password(
    password,
    salt=None,
):
    """
    Return:

        (salt_hex, hash_hex)
    """

    if salt is None:
        salt = secrets.token_bytes(16)

    elif isinstance(salt, str):
        salt = bytes.fromhex(salt)

    derived_key = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    )

    return (
        salt.hex(),
        derived_key.hex(),
    )


def verify_password(
    password,
    salt_hex,
    expected_hash_hex,
):
    """
    Safely compare a password hash.
    """

    _, computed = hash_password(
        password,
        salt_hex,
    )

    return hmac.compare_digest(
        computed,
        expected_hash_hex,
    )


# =============================================================================
# FIRST-RUN VOICE SETUP
# =============================================================================

def set_owner_voice(
    file_storage,
    reference_path,
):
    """
    Receive the recording from the browser.

    Process:

        browser recording
            ↓
        temporary file
            ↓
        clean WAV for XTTS
            ↓
        local cache
            ↓
        Supabase private Storage

    The local file is intentionally retained as a cache so the current
    request can use it immediately.
    """

    original_name = secure_filename(
        file_storage.filename
        or "voice.webm"
    )

    extension = (
        os.path.splitext(original_name)[1]
        or ".webm"
    )

    with tempfile.NamedTemporaryFile(
        suffix=extension,
        delete=False,
    ) as tmp:

        file_storage.save(
            tmp.name
        )

        tmp_path = tmp.name

    try:

        # Make sure the parent folder exists.
        parent = os.path.dirname(
            reference_path
        )

        if parent:
            os.makedirs(
                parent,
                exist_ok=True,
            )

        # Convert browser audio to XTTS reference WAV.
        convert_to_reference_wav(
            tmp_path,
            reference_path,
        )

        # Persist the finished voice.
        upload_owner_voice(
            reference_path
        )

    finally:

        if os.path.exists(
            tmp_path
        ):
            os.unlink(
                tmp_path
            )


# =============================================================================
# PASSWORD SETUP
# =============================================================================

def set_password(password):
    """
    Set the owner's password.

    Only works before a password has already been created.

    Returns:

        (True, None)

    or:

        (False, error_message)
    """

    if len(password) < 4:
        return (
            False,
            "Password must be at least 4 characters.",
        )

    db = load_db()

    if db.get("password_hash"):
        return (
            False,
            "A password is already set.",
        )

    salt_hex, hash_hex = hash_password(
        password
    )

    db["password_salt"] = salt_hex
    db["password_hash"] = hash_hex

    save_db(db)

    return True, None


def check_password(password):
    """
    Check the supplied password.
    """

    db = load_db()

    salt_hex = db.get(
        "password_salt"
    )

    expected_hash = db.get(
        "password_hash"
    )

    if not salt_hex or not expected_hash:
        return False

    return verify_password(
        password,
        salt_hex,
        expected_hash,
    )


# =============================================================================
# LEGACY VOICE LOGIN SUPPORT
# =============================================================================

def try_voice_login(
    file_storage,
):
    """
    Legacy speaker-verification login.

    The current frontend uses password-only login, but these functions
    remain so older backend references do not break.
    """

    db = load_db()

    ready, days_remaining = (
        get_voice_login_readiness(db)
    )

    if not ready:

        if days_remaining is not None:

            return (
                False,
                (
                    "Still learning your voice — "
                    f"about {days_remaining:.1f} more day(s)."
                ),
                False,
            )

        return (
            False,
            "MIAH hasn't learned your voice yet.",
            False,
        )

    original_name = secure_filename(
        file_storage.filename
        or "clip.webm"
    )

    extension = (
        os.path.splitext(original_name)[1]
        or ".webm"
    )

    with tempfile.NamedTemporaryFile(
        suffix=extension,
        delete=False,
    ) as tmp:

        file_storage.save(
            tmp.name
        )

        tmp_path = tmp.name

    try:

        live_embedding = (
            compute_embedding_from_file(
                tmp_path
            )
        )

    except Exception:

        return (
            False,
            "Couldn't process that recording.",
            True,
        )

    finally:

        if os.path.exists(
            tmp_path
        ):
            os.unlink(
                tmp_path
            )

    embeddings = [
        entry["vector"]
        for entry in db[
            "her_voice_profile"
        ]["embeddings"]
    ]

    if not embeddings:
        return (
            False,
            "Voice profile is empty.",
            True,
        )

    profile_vector = [
        float(
            sum(column)
        ) / len(column)
        for column in zip(
            *embeddings
        )
    ]

    from voice import cosine_similarity

    score = cosine_similarity(
        profile_vector,
        live_embedding,
    )

    if score >= VOICE_MATCH_THRESHOLD:

        return (
            True,
            None,
            True,
        )

    return (
        False,
        "Didn't recognize your voice — try your password.",
        True,
    )


def record_voice_sample(
    audio_path,
):
    """
    Best-effort legacy voice-profile learning.

    A failure here must never break the user's normal interaction.
    """

    try:

        db = load_db()

        if not db.get(
            "password_hash"
        ):
            return

        embedding = (
            compute_embedding_from_file(
                audio_path
            )
        )

        add_voice_sample(
            db,
            embedding,
        )

        save_db(db)

    except Exception:
        pass
