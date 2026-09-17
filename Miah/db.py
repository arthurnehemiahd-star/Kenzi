"""
MIAH — persistent JSON state store.

Primary storage:
    Supabase PostgreSQL table `miah_state`

Local storage:
    Used as a cache/fallback only.

This keeps the rest of MIAH using the same load_db() / save_db()
interface while moving persistent state away from Render's temporary
filesystem.
"""

import json
import os
import threading
import time

from config import (
    CONVERSATION_KEY,
    DB_PATH,
    KEEP_RECENT_MESSAGES,
    SUMMARY_TRIGGER_COUNT,
    ensure_dirs,
)

from supabase_store import (
    load_remote_state,
    save_remote_state,
    supabase_configured,
)


_db_lock = threading.RLock()


def _empty_db():
    return {
        "conversations": {
            CONVERSATION_KEY: []
        },
        "notes": [],
        "memory_summary": "",
        "password_hash": None,
        "password_salt": None,
        "her_voice_profile": {
            "embeddings": [],
            "first_sample_at": None,
        },
        "spotify": {},
    }


def _merge_with_defaults(data):
    """
    Backfill keys introduced by newer versions.
    """

    base = _empty_db()

    if isinstance(data, dict):
        base.update(data)

    if not isinstance(base.get("conversations"), dict):
        base["conversations"] = {}

    base["conversations"].setdefault(
        CONVERSATION_KEY,
        [],
    )

    base.setdefault(
        "notes",
        [],
    )

    base.setdefault(
        "memory_summary",
        "",
    )

    base.setdefault(
        "password_hash",
        None,
    )

    base.setdefault(
        "password_salt",
        None,
    )

    base.setdefault(
        "her_voice_profile",
        {
            "embeddings": [],
            "first_sample_at": None,
        },
    )

    base.setdefault(
        "spotify",
        {},
    )

    return base


def _load_local_db():
    """
    Load the local cache.
    """

    if not os.path.exists(DB_PATH):
        return None

    try:
        with open(
            DB_PATH,
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        return _merge_with_defaults(data)

    except (json.JSONDecodeError, OSError):
        try:
            os.rename(
                DB_PATH,
                DB_PATH + f".corrupt.{int(time.time())}",
            )
        except OSError:
            pass

        return None


def _save_local_db(db):
    """
    Save the local cache atomically.
    """

    ensure_dirs()

    tmp = DB_PATH + ".tmp"

    with open(
        tmp,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            db,
            f,
            indent=2,
            ensure_ascii=False,
        )

    os.replace(
        tmp,
        DB_PATH,
    )


def load_db():
    """
    Load MIAH state.

    Order:
        1. Supabase
        2. Local cache
        3. Fresh empty DB

    If local data exists but Supabase is empty, the local data is
    automatically migrated to Supabase.
    """

    with _db_lock:

        # --------------------------------------------------------------
        # 1. Try Supabase
        # --------------------------------------------------------------

        if supabase_configured():

            try:
                remote = load_remote_state()

                if remote is not None:
                    db = _merge_with_defaults(remote)

                    # Refresh local cache too.
                    try:
                        _save_local_db(db)
                    except Exception:
                        pass

                    return db

            except Exception as exc:
                print(
                    f"[supabase] Database read failed: {exc}"
                )

        # --------------------------------------------------------------
        # 2. Local fallback / migration
        # --------------------------------------------------------------

        local = _load_local_db()

        if local is not None:

            if supabase_configured():

                try:
                    save_remote_state(local)

                    print(
                        "[supabase] Migrated local MIAH data "
                        "to Supabase."
                    )

                except Exception as exc:
                    print(
                        "[supabase] Local-to-remote migration "
                        f"failed: {exc}"
                    )

            return local

        # --------------------------------------------------------------
        # 3. Brand-new database
        # --------------------------------------------------------------

        return _empty_db()


def save_db(db):
    """
    Save MIAH state locally and remotely.

    Supabase is the persistent source of truth.
    """

    with _db_lock:

        db = _merge_with_defaults(db)

        # Always keep a local cache.
        try:
            _save_local_db(db)
        except Exception as exc:
            print(
                f"[miah] Local DB save failed: {exc}"
            )

        # Persist remotely.
        if supabase_configured():

            save_remote_state(db)


# ----------------------------------------------------------------------
# Long-term memory summarization
# ----------------------------------------------------------------------

def _flatten_message_for_summary(msg):
    content = msg.get("content")

    if isinstance(content, str):
        text = content

    elif isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if (
                isinstance(block, dict)
                and block.get("type") == "text"
            )
        ]

        text = " ".join(
            part
            for part in parts
            if part
        )

    else:
        text = ""

    if not text:
        return None

    return f"{msg.get('role', 'unknown')}: {text}"


def maybe_summarize_history(
    db,
    call_llm_fn,
):
    """
    Collapse older conversation turns into an evolving memory summary.
    """

    history = db["conversations"].setdefault(
        CONVERSATION_KEY,
        [],
    )

    if len(history) <= SUMMARY_TRIGGER_COUNT:
        return

    to_summarize = history[
        :-KEEP_RECENT_MESSAGES
    ]

    remaining = history[
        -KEEP_RECENT_MESSAGES:
    ]

    lines = [
        _flatten_message_for_summary(message)
        for message in to_summarize
    ]

    conversation_text = "\n".join(
        line
        for line in lines
        if line
    )

    if not conversation_text.strip():

        db["conversations"][CONVERSATION_KEY] = remaining

        return

    existing_summary = db.get(
        "memory_summary",
        "",
    )

    prompt = (
        "Here is what you already knew about this user, from earlier "
        f"conversations:\n{existing_summary or '(nothing yet)'}\n\n"
        f"Here is a new stretch of conversation with her:\n"
        f"{conversation_text}\n\n"
        "Update your understanding of her: her personality, interests, "
        "recurring topics, how she seems to feel about things, and how "
        "she likes to be talked to. Write a concise, updated summary "
        "(a few short paragraphs) that merges this with what you already "
        "knew, rather than just appending to it. Write it as private notes "
        "for yourself to use later — not as a message to her."
    )

    try:

        result = call_llm_fn(
            [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_tokens=600,
        )

        new_summary = (
            result["choices"][0]["message"]
            .get("content")
            or ""
        )

    except Exception:
        # Keep raw history intact if summarization fails.
        return

    db["memory_summary"] = new_summary

    db["conversations"][
        CONVERSATION_KEY
    ] = remaining


# ----------------------------------------------------------------------
# Voice profile
# ----------------------------------------------------------------------

def add_voice_sample(
    db,
    embedding,
):
    profile = db.setdefault(
        "her_voice_profile",
        {
            "embeddings": [],
            "first_sample_at": None,
        },
    )

    if profile["first_sample_at"] is None:
        profile["first_sample_at"] = time.time()

    profile["embeddings"].append(
        {
            "vector": embedding,
            "timestamp": time.time(),
        }
    )

    # Keep at most 60 samples.
    if len(profile["embeddings"]) > 60:

        profile["embeddings"] = (
            profile["embeddings"][-60:]
        )


def get_voice_login_readiness(db):
    """
    Returns:

        (ready, days_remaining)

    This is retained for compatibility with the existing MIAH backend.
    """

    from config import (
        VOICE_LOGIN_MIN_DAYS,
        VOICE_LOGIN_MIN_SAMPLES,
    )

    profile = db.get(
        "her_voice_profile",
        {},
    )

    embeddings = profile.get(
        "embeddings",
        [],
    )

    first_sample_at = profile.get(
        "first_sample_at"
    )

    if (
        not embeddings
        or not first_sample_at
        or len(embeddings)
        < VOICE_LOGIN_MIN_SAMPLES
    ):
        return False, None

    days_collected = (
        time.time() - first_sample_at
    ) / 86400

    if days_collected >= VOICE_LOGIN_MIN_DAYS:
        return True, None

    return (
        False,
        max(
            0.0,
            VOICE_LOGIN_MIN_DAYS
            - days_collected,
        ),
    )
