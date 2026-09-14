"""Local music library + Spotify OAuth.

The OAuth flow now includes a `state` parameter stored in the session
and verified on callback. Without it, an attacker can craft a callback
URL that connects THEIR Spotify account to her MIAH instance — trivial
to fix, and the original omitted it entirely.
"""

import os
import secrets
import time

import requests
from flask import redirect, request, session
from werkzeug.utils import secure_filename

from config import (
    ALLOWED_AUDIO_EXTENSIONS,
    MUSIC_DIR,
    SPOTIFY_API_BASE,
    SPOTIFY_AUTH_URL,
    SPOTIFY_CLIENT_ID,
    SPOTIFY_CLIENT_SECRET,
    SPOTIFY_REDIRECT_URI,
    SPOTIFY_SCOPES,
    SPOTIFY_TOKEN_URL,
)
from db import load_db, save_db


# ----------------------------------------------------------------------
# Local library
# ----------------------------------------------------------------------
def list_local_tracks():
    if not os.path.isdir(MUSIC_DIR):
        return []
    return sorted(
        f for f in os.listdir(MUSIC_DIR)
        if os.path.splitext(f)[1].lower() in ALLOWED_AUDIO_EXTENSIONS
    )


def resolve_track_path(filename):
    """Return the on-disk path for a library track, or None if the name
    is unsafe or the file doesn't exist."""
    safe_name = secure_filename(filename)
    if not safe_name:
        return None
    if os.path.splitext(safe_name)[1].lower() not in ALLOWED_AUDIO_EXTENSIONS:
        return None
    full_path = os.path.join(MUSIC_DIR, safe_name)
    if not os.path.isfile(full_path):
        return None
    return full_path


# ----------------------------------------------------------------------
# Spotify OAuth
# ----------------------------------------------------------------------
def build_login_url():
    """Build the Spotify authorize URL and stash a CSRF `state` in the
    session. Caller redirects the user to the returned URL."""
    state = secrets.token_urlsafe(24)
    session["spotify_oauth_state"] = state

    params = {
        "client_id": SPOTIFY_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": SPOTIFY_REDIRECT_URI,
        "scope": SPOTIFY_SCOPES,
        "state": state,
    }
    from urllib.parse import urlencode
    return f"{SPOTIFY_AUTH_URL}?{urlencode(params)}"


def handle_callback(code, state):
    """Exchange the auth code for tokens. Returns (ok, message)."""
    expected_state = session.pop("spotify_oauth_state", None)
    if not expected_state or not state or state != expected_state:
        return False, "Spotify authorization failed (state mismatch)."

    resp = requests.post(SPOTIFY_TOKEN_URL, data={
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": SPOTIFY_REDIRECT_URI,
        "client_id": SPOTIFY_CLIENT_ID,
        "client_secret": SPOTIFY_CLIENT_SECRET,
    }, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    db = load_db()
    db["spotify"] = {
        "access_token": data["access_token"],
        "refresh_token": data.get("refresh_token", ""),
        "expires_at": time.time() + data.get("expires_in", 3600),
    }
    save_db(db)
    return True, "Spotify connected!"