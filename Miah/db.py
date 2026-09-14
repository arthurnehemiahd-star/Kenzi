"""JSON-file-backed store. One file, one user, no concurrency guarantees
beyond a single-process Flask app. Fine for a personal gift; not built
for multiple simultaneous writers."""

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

# One process-wide lock so two requests can't interleave a read-modify-write
# on the JSON file and lose each other's changes. The original had no lock
# at all — rare in practice with one user, but cheap to add and removes a
# whole class of "where did my note go" bugs.
_db_lock = threading.RLock()


def _empty_db():
    return {
        "conversations": {CONVERSATION_KEY: []},
        "notes": [],
        "memory_summary": "",
        "password_hash": None,
        "password_salt": None,
        "her_voice_profile": {"embeddings": [], "first_sample_at": None},
        "spotify": {},
    }


def load_db():
    """Load the DB. Returns a fresh empty DB if the file doesn't exist
    or is corrupt (rather than crashing on startup)."""
    with _db_lock:
        if not os.path.exists(DB_PATH):
            return _empty_db()
        try:
            with open(DB_PATH, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            # Corrupt file — don't lose the user's data silently. Rename it.
            try:
                os.rename(DB_PATH, DB_PATH + f".corrupt.{int(time.time())}")
            except OSError:
                pass
            return _empty_db()

        # Backfill any keys a previous version didn't have.
        base = _empty_db()
        base.update(data)
        base["conversations"].setdefault(CONVERSATION_KEY, [])
        base.setdefault("her_voice_profile", {"embeddings": [], "first_sample_at": None})
        return base


def save_db(db):
    """Write the DB atomically — write to a temp file, then rename, so a
    crash mid-write can't leave a half-written JSON file behind."""
    with _db_lock:
        ensure_dirs()
        tmp = DB_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(db, f, indent=2)
        os.replace(tmp, DB_PATH)


# ----------------------------------------------------------------------
# Long-term memory summarization
# ----------------------------------------------------------------------
def _flatten_message_for_summary(msg):
    content = msg.get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts = [
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        text = " ".join(p for p in parts if p)
    else:
        text = ""
    return f"{msg['role']}: {text}" if text else None


def maybe_summarize_history(db, call_llm_fn):
    """If the conversation has grown past SUMMARY_TRIGGER_COUNT, collapse
    the older turns into an evolving summary and keep only the most recent
    KEEP_RECENT_MESSAGES verbatim.

    Takes `call_llm_fn` as an argument instead of importing llm.py directly
    so db.py stays free of LLM dependencies (and so tests can pass a fake).
    """
    history = db["conversations"].setdefault(CONVERSATION_KEY, [])
    if len(history) <= SUMMARY_TRIGGER_COUNT:
        return

    to_summarize = history[:-KEEP_RECENT_MESSAGES]
    remaining = history[-KEEP_RECENT_MESSAGES:]

    lines = [_flatten_message_for_summary(m) for m in to_summarize]
    conversation_text = "\n".join(l for l in lines if l)
    if not conversation_text.strip():
        db["conversations"][CONVERSATION_KEY] = remaining
        return

    existing_summary = db.get("memory_summary", "")
    prompt = (
        "Here is what you already knew about this user, from earlier "
        f"conversations:\n{existing_summary or '(nothing yet)'}\n\n"
        f"Here is a new stretch of conversation with her:\n{conversation_text}\n\n"
        "Update your understanding of her: her personality, interests, "
        "recurring topics, how she seems to feel about things, and how she "
        "likes to be talked to. Write a concise, updated summary (a few "
        "short paragraphs) that merges this with what you already knew, "
        "rather than just appending to it. Write it as private notes for "
        "yourself to use later — not as a message to her."
    )

    try:
        result = call_llm_fn(
            [{"role": "user", "content": prompt}],
            max_tokens=600,
        )
        new_summary = result["choices"][0]["message"].get("content") or ""
    except Exception:
        # If summarization fails (rate limit, network), keep the raw history
        # intact rather than dropping it. Next turn will try again.
        return

    db["memory_summary"] = new_summary
    db["conversations"][CONVERSATION_KEY] = remaining


# ----------------------------------------------------------------------
# Voice profile (passive learning for voice login)
# ----------------------------------------------------------------------
def add_voice_sample(db, embedding):
    profile = db.setdefault(
        "her_voice_profile", {"embeddings": [], "first_sample_at": None}
    )
    if profile["first_sample_at"] is None:
        profile["first_sample_at"] = time.time()
    profile["embeddings"].append({"vector": embedding, "timestamp": time.time()})
    if len(profile["embeddings"]) > 60:  # VOICE_PROFILE_MAX_SAMPLES
        profile["embeddings"] = profile["embeddings"][-60:]


def get_voice_login_readiness(db):
    """Returns (ready: bool, days_remaining: float|None)."""
    from config import VOICE_LOGIN_MIN_DAYS, VOICE_LOGIN_MIN_SAMPLES

    profile = db.get("her_voice_profile", {})
    embeddings = profile.get("embeddings", [])
    first_sample_at = profile.get("first_sample_at")

    if not embeddings or not first_sample_at or len(embeddings) < VOICE_LOGIN_MIN_SAMPLES:
        return False, None

    days_collected = (time.time() - first_sample_at) / 86400
    if days_collected >= VOICE_LOGIN_MIN_DAYS:
        return True, None
    return False, max(0.0, VOICE_LOGIN_MIN_DAYS - days_collected)