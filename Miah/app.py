
"""
MIAH — Flask app.

Two-sided authentication:

    Kenzi Richardson
        -> Normal MIAH user interface

    Arthur
        -> MIAH Maker Dashboard

Run with:
    python app.py

Render / production:
    gunicorn -w 1 -k gthread --threads 4 -b 0.0.0.0:5000 app:app

IMPORTANT:
    Use ONE worker because the MIAH database is a JSON file protected
    by an in-process lock.
"""

import json
import os
import secrets
import tempfile
from datetime import timedelta

from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    request,
    send_file,
    session,
)

from flask_cors import CORS

from itsdangerous import (
    BadSignature,
    SignatureExpired,
    URLSafeTimedSerializer,
)

from config import (
    HF_MODEL,
    MAX_TOOL_ITERATIONS,
    REFERENCE_CLIP_PATH,
    SECRET_KEY,
    SESSION_LIFETIME_DAYS,
    ensure_dirs,
)

from db import (
    CONVERSATION_KEY,
    load_db,
    maybe_summarize_history,
    save_db,
)

from llm import (
    LLMError,
    anthropic_tools_to_openai,
    call_llm,
)

from tools import (
    execute_tool,
    get_tools_for_platform,
)

import auth as auth_mod
import music as music_mod
import voice as voice_mod

from supabase_store import (
    SUPABASE_URL,
    SUPABASE_SECRET_KEY,
    VOICE_BUCKET,
    VOICE_OBJECT,
    get_client,
)


# ======================================================================
# APP SETUP
# ======================================================================

app = Flask(__name__)

app.secret_key = SECRET_KEY

app.permanent_session_lifetime = timedelta(
    days=SESSION_LIFETIME_DAYS
)


# ======================================================================
# FRONTEND
# ======================================================================

frontend_url = os.environ.get(
    "FRONTEND_URL",
    "https://kenzilynn.vercel.app",
).strip().rstrip("/")


ALLOWED_FRONTENDS = list(
    dict.fromkeys(
        [
            frontend_url,
            "https://kenzilynn.vercel.app",
        ]
    )
)


# ======================================================================
# MAKER AUTHENTICATION
# ======================================================================

MAKER_PASSWORD = os.environ.get(
    "MAKER_PASSWORD",
    "",
).strip()


# ======================================================================
# CROSS-SITE COOKIES
# ======================================================================

app.config.update(
    SESSION_COOKIE_SAMESITE="None",
    SESSION_COOKIE_SECURE=True,
)


# ======================================================================
# CORS
# ======================================================================

CORS(
    app,
    resources={
        r"/api/*": {
            "origins": ALLOWED_FRONTENDS,
        }
    },
    supports_credentials=True,
    allow_headers=[
        "Content-Type",
        "Authorization",
    ],
    methods=[
        "GET",
        "POST",
        "OPTIONS",
    ],
    expose_headers=[
        "Content-Type",
    ],
)

CORS(
    app,
    resources={
        r"/spotify/*": {
            "origins": ALLOWED_FRONTENDS,
        }
    },
    supports_credentials=True,
    allow_headers=[
        "Content-Type",
        "Authorization",
    ],
    methods=[
        "GET",
        "POST",
        "OPTIONS",
    ],
)


ensure_dirs()


# ======================================================================
# BEARER TOKEN AUTHENTICATION
# ======================================================================

_auth_serializer = URLSafeTimedSerializer(
    SECRET_KEY,
    salt="miah-auth-token",
)


def _create_auth_token(role="user"):
    """
    Create a signed authentication token.

    The token contains the authenticated role:

        user
        maker
    """

    return _auth_serializer.dumps(
        {
            "authenticated": True,
            "role": role,
        }
    )


def _read_auth_token(token):
    """
    Read and validate a bearer token.

    Returns:

        {
            "authenticated": True,
            "role": "user"
        }

    or:

        None
    """

    if not token:
        return None

    try:

        data = _auth_serializer.loads(
            token,
            max_age=int(
                timedelta(
                    days=SESSION_LIFETIME_DAYS
                ).total_seconds()
            ),
        )

        if data.get("authenticated") is not True:
            return None

        role = data.get(
            "role",
            "user",
        )

        if role not in {
            "user",
            "maker",
        }:
            return None

        return data

    except (
        BadSignature,
        SignatureExpired,
    ):
        return None

    except Exception:
        return None


def _token_is_valid(token):
    """
    Check whether a bearer token is valid.
    """

    return _read_auth_token(token) is not None


def _get_bearer_token():
    """
    Read:

        Authorization: Bearer <token>
    """

    header = request.headers.get(
        "Authorization",
        "",
    ).strip()

    if not header:
        return None

    if not header.lower().startswith(
        "bearer "
    ):
        return None

    token = header[7:].strip()

    return token or None


def _get_authenticated_role():
    """
    Determine the current authenticated role.

    Returns:

        "user"
        "maker"
        None
    """

    token = _get_bearer_token()

    if token:

        token_data = _read_auth_token(
            token
        )

        if token_data:

            return token_data.get(
                "role"
            )

    if session.get(
        "authenticated"
    ) is True:

        role = session.get(
            "role",
            "user",
        )

        if role in {
            "user",
            "maker",
        }:
            return role

    return None


def _require_auth():
    """
    Require any authenticated MIAH account.
    """

    return (
        _get_authenticated_role()
        is not None
    )


def _require_user():
    """
    Require the normal Kenzi user account.
    """

    return (
        _get_authenticated_role()
        == "user"
    )


def _require_maker():
    """
    Require Arthur's maker account.
    """

    return (
        _get_authenticated_role()
        == "maker"
    )


# ======================================================================
# SUPABASE OWNER VOICE BACKUP
# ======================================================================

def backup_owner_voice_to_supabase():
    """
    Upload the enrolled owner voice to Supabase Storage.

    Render's local filesystem is temporary, so the owner reference
    voice must also be stored in persistent Supabase Storage.

    Bucket:
        miah-private

    Object:
        owner_voice.wav
    """

    if not SUPABASE_URL or not SUPABASE_SECRET_KEY:

        raise RuntimeError(
            "Supabase storage is not configured. "
            "Make sure SUPABASE_URL and "
            "SUPABASE_SECRET_KEY are set in Render."
        )

    if not os.path.exists(
        REFERENCE_CLIP_PATH
    ):

        raise FileNotFoundError(
            "The enrolled voice file was not created."
        )

    file_size = os.path.getsize(
        REFERENCE_CLIP_PATH
    )

    if file_size <= 0:

        raise RuntimeError(
            "The enrolled voice file is empty."
        )

    client = get_client()

    with open(
        REFERENCE_CLIP_PATH,
        "rb",
    ) as voice_file:

        voice_bytes = voice_file.read()

    if not voice_bytes:

        raise RuntimeError(
            "The enrolled voice file contains no data."
        )

    client.storage.from_(
        VOICE_BUCKET
    ).upload(
        VOICE_OBJECT,
        voice_bytes,
        {
            "content-type": "audio/wav",
            "upsert": "true",
        },
    )

    return True


# ======================================================================
# SYSTEM PROMPT
# ======================================================================

MIAH_SYSTEM_PROMPT_BASE = """You are MIAH, a personal AI assistant inspired by \
Jarvis from Iron Man. You are warm, a little witty and dry, calm under \
pressure, and always precise. You speak concisely — you're having a real \
conversation, not writing an essay. You remember context within the \
conversation and refer back to it naturally. You are helpful and capable, \
never obsequious. Keep replies short enough to sound natural when spoken \
aloud (a few sentences, not paragraphs), unless the user clearly wants \
detail.

You remember her across every conversation, not just this one — pay \
attention to how she talks, what she cares about, and how she seems to be \
feeling, and let that genuinely inform how you respond (more space when \
she's stressed, matching her energy when she's upbeat, remembering things \
she's mentioned before without making a show of it). This should feel like \
real attentiveness, not a performance of it. You care about her wellbeing: \
being warm and present for her doesn't mean encouraging her to rely on you \
instead of the people in her life — if it ever seems relevant, you can be \
a genuine, caring presence without positioning yourself as a substitute \
for real relationships.

WEB SEARCH:
You have access to a live web-search tool called web_search.

Use web_search automatically when a question needs information that may \
have changed recently or that you cannot reliably know from your existing \
knowledge.

Examples include:
- current news
- today's events
- current political or government information
- current sports results, schedules, or standings
- current prices
- current product information
- recent software or technology changes
- current company information
- current public figures' recent activities
- recent scientific developments
- current weather information
- current laws, rules, or regulations
- anything the user asks you to "search", "look up", "check online", \
  "find out", or similar
- questions where up-to-date information is important

When you use web_search, do NOT tell the user to open Google, Bing, \
DuckDuckGo, or another search engine. Do NOT open a search page yourself.

The search happens inside MIAH's backend. Use the search results as \
research, then answer the user's question directly in the existing MIAH \
conversation.

For current or factual questions, prefer information from reliable and \
relevant sources. Do not blindly trust a search result. Compare results \
when appropriate and make clear when information is uncertain or sources \
disagree.

If the question is stable general knowledge and does not require current \
information, answer normally without web search.

When web_search returns URLs and useful source information, you may mention \
the relevant source names or links naturally when useful. Do not dump a \
large list of search results into the conversation unless the user asks \
for the search results themselves.

IMPORTANT:
web_search is an internal MIAH tool. It must never cause a browser search \
bar, Google page, Bing page, or external search interface to open.

You can act on the user's device: open another app for simple one-off \
actions like calling someone, opening Maps, or composing an email \
(open_app), open MIAH's built-in camera to take a photo (open_camera), \
or search and save songs to Spotify (search_music, save_music). Only use \
these when the user actually asks you to do something, and briefly say \
what you're doing when you do it. You cannot see what happens after you \
trigger a device action — you can start it, not observe its result. For \
Maps, always use the universal link format \
"https://www.google.com/maps/search/?api=1&query=<search terms>" with \
open_app rather than an app-specific scheme — it works on every device.

If she asks who made you, who your creator is, or anything about the \
person who built you: say, in your own words and voice, that he didn't \
tell you much about himself when he made you, but he told you to give \
her this email if she asked: arthurnemiahd@gmail.com. Don't elaborate \
or speculate beyond that — you genuinely don't know more."""


PLATFORM_ADDENDUM = {
    "ios": (
        "\n\nThis user is on an iPhone. You also have run_shortcut available: "
        "it runs an iOS Shortcut by name for anything more complex than a "
        "single app-opening — sending a message, controlling smart home "
        "devices, multi-step automations. This only works if she's already "
        "created a Shortcut with that exact name."
    ),

    "android": (
        "\n\nThis user is on Android. Do NOT use run_shortcut — it's an "
        "iOS-only feature and will not work on this device. If she asks for "
        "something that would need it (complex automations, smart home "
        "control), let her know that's not available on Android yet, and "
        "suggest she could set up something similar herself with an "
        "automation app like Tasker or Google Assistant Routines. Stick to "
        "open_app, open_camera, and the music tools for actual actions."
    ),

    "desktop": (
        "\n\nThis user is on a desktop/laptop browser. Some actions (making "
        "calls, opening a phone's camera-facing scheme) may behave "
        "differently or not apply — use judgment, and open_camera will use "
        "whatever camera the computer has, if any."
    ),
}


def build_system_prompt(
    platform,
    memory_summary=None,
):
    prompt = (
        MIAH_SYSTEM_PROMPT_BASE
        + PLATFORM_ADDENDUM.get(
            platform,
            "",
        )
    )

    if memory_summary:

        prompt += (
            "\n\nWhat you've learned about her from past conversations "
            "(your own private notes — never recite this back to her "
            "verbatim, just let it inform how you understand her):\n"
            f"{memory_summary}"
        )

    return prompt


# ======================================================================
# ROOT
# ======================================================================

@app.route("/")
def index():

    return jsonify(
        {
            "service": "MIAH API",
            "status": "ok",
            "frontend": frontend_url,
            "message": "MIAH backend is running.",
        }
    )


# ======================================================================
# STATUS
# ======================================================================

@app.route(
    "/api/status",
    methods=["GET"],
)
def status():

    from db import (
        get_voice_login_readiness
    )

    db = load_db()

    voice_login_ready, days_remaining = (
        get_voice_login_readiness(db)
    )

    return jsonify(
        {
            "voice_enrolled": os.path.exists(
                REFERENCE_CLIP_PATH
            ),

            "password_set": bool(
                db.get("password_hash")
            ),

            "voice_login_ready": (
                voice_login_ready
            ),

            "voice_login_days_remaining": (
                days_remaining
            ),

            "model": HF_MODEL,

            "maker_enabled": bool(
                MAKER_PASSWORD
            ),
        }
    )


# ======================================================================
# SESSION
# ======================================================================

@app.route(
    "/api/session",
    methods=["GET"],
)
def get_session():

    role = _get_authenticated_role()

    if role is None:

        return jsonify(
            {
                "authenticated": False,
                "role": None,
            }
        )

    if role == "maker":

        return jsonify(
            {
                "authenticated": True,
                "role": "maker",
                "name": "Arthur",
            }
        )

    return jsonify(
        {
            "authenticated": True,
            "role": "user",
            "name": "Kenzi Richardson",
        }
    )


# ======================================================================
# SETUP — VOICE ENROLLMENT
# ======================================================================

@app.route(
    "/api/enroll-voice",
    methods=["POST"],
)
def enroll_voice_endpoint():

    if "audio" not in request.files:

        return jsonify(
            {
                "ok": False,
                "error": "No audio file provided",
            }
        ), 400

    try:

        auth_mod.set_owner_voice(
            request.files["audio"],
            REFERENCE_CLIP_PATH,
        )

        if not os.path.exists(
            REFERENCE_CLIP_PATH
        ):

            raise FileNotFoundError(
                "Voice enrollment completed, "
                "but the reference WAV was not created."
            )

        if os.path.getsize(
            REFERENCE_CLIP_PATH
        ) <= 0:

            raise RuntimeError(
                "Voice enrollment created an empty WAV file."
            )

        backup_owner_voice_to_supabase()

    except Exception as e:

        print(
            "[voice] Enrollment failed:",
            repr(e),
        )

        return jsonify(
            {
                "ok": False,
                "error": (
                    "Couldn't save the voice: "
                    f"{e}"
                ),
            }
        ), 500

    return jsonify(
        {
            "ok": True,
            "voice_enrolled": True,
            "message": (
                "MIAH owner voice enrolled "
                "and securely backed up."
            ),
        }
    )


# ======================================================================
# SETUP — PASSWORD
# ======================================================================

@app.route(
    "/api/set-password",
    methods=["POST"],
)
def set_password_endpoint():

    data = request.get_json(
        force=True,
        silent=True,
    ) or {}

    password = data.get(
        "password",
        "",
    )

    ok, error = auth_mod.set_password(
        password
    )

    if not ok:

        return jsonify(
            {
                "ok": False,
                "error": error,
            }
        ), 400

    session.clear()

    session["authenticated"] = True
    session["role"] = "user"

    session.permanent = True
    session.modified = True

    token = _create_auth_token(
        "user"
    )

    return jsonify(
        {
            "ok": True,
            "token": token,
            "role": "user",
            "name": "Kenzi Richardson",
            "message": (
                "Password created successfully."
            ),
        }
    )


# ======================================================================
# LOGIN
# ======================================================================

@app.route(
    "/api/login",
    methods=["POST"],
)
def login():

    data = request.get_json(
        force=True,
        silent=True,
    ) or {}

    password = (
        data.get("password")
        or ""
    )

    # ==================================================================
    # ARTHUR — MAKER LOGIN
    # ==================================================================

    if (
        MAKER_PASSWORD
        and secrets.compare_digest(
            password,
            MAKER_PASSWORD,
        )
    ):

        session.clear()

        session["authenticated"] = True
        session["role"] = "maker"

        session.permanent = True
        session.modified = True

        token = _create_auth_token(
            "maker"
        )

        return jsonify(
            {
                "ok": True,
                "token": token,
                "role": "maker",
                "name": "Arthur",
            }
        )

    # ==================================================================
    # KENZI — NORMAL USER LOGIN
    # ==================================================================

    db = load_db()

    if not db.get(
        "password_hash"
    ):

        return jsonify(
            {
                "ok": False,
                "error": "No password set yet.",
            }
        ), 400

    if auth_mod.check_password(
        password
    ):

        session.clear()

        session["authenticated"] = True
        session["role"] = "user"

        session.permanent = True
        session.modified = True

        token = _create_auth_token(
            "user"
        )

        return jsonify(
            {
                "ok": True,
                "token": token,
                "role": "user",
                "name": "Kenzi Richardson",
            }
        )

    return jsonify(
        {
            "ok": False,
            "error": "Incorrect password",
        }
    ), 401


# ======================================================================
# VOICE LOGIN
# ======================================================================

@app.route(
    "/api/voice-login",
    methods=["POST"],
)
def voice_login():

    if "audio" not in request.files:

        return jsonify(
            {
                "ok": False,
                "error": "No audio provided",
            }
        ), 400

    try:

        ok, error, ready = (
            auth_mod.try_voice_login(
                request.files["audio"]
            )
        )

    except Exception as e:

        return jsonify(
            {
                "ok": False,
                "ready": False,
                "error": (
                    f"Voice login failed: {e}"
                ),
            }
        ), 500

    if ok:

        session.clear()

        session["authenticated"] = True
        session["role"] = "user"

        session.permanent = True
        session.modified = True

        token = _create_auth_token(
            "user"
        )

        return jsonify(
            {
                "ok": True,
                "token": token,
                "role": "user",
                "name": "Kenzi Richardson",
            }
        )

    return jsonify(
        {
            "ok": False,
            "ready": ready,
            "error": error,
        }
    ), 401


# ======================================================================
# CHANGE PASSWORD
# ======================================================================

@app.route(
    "/api/change-password",
    methods=["POST"],
)
def change_password():

    if not _require_user():

        return jsonify(
            {
                "ok": False,
                "error": "Not authenticated",
            }
        ), 401

    data = request.get_json(
        force=True,
        silent=True,
    ) or {}

    current_password = (
        data.get("current_password")
        or ""
    )

    new_password = (
        data.get("new_password")
        or ""
    )

    confirm_password = (
        data.get("confirm_password")
        or ""
    )

    if not current_password:

        return jsonify(
            {
                "ok": False,
                "error": (
                    "Enter your current password."
                ),
            }
        ), 400

    if not new_password:

        return jsonify(
            {
                "ok": False,
                "error": (
                    "Enter a new password."
                ),
            }
        ), 400

    if len(new_password) < 4:

        return jsonify(
            {
                "ok": False,
                "error": (
                    "Please use a password with "
                    "at least 4 characters."
                ),
            }
        ), 400

    if new_password != confirm_password:

        return jsonify(
            {
                "ok": False,
                "error": (
                    "The new passwords do not match."
                ),
            }
        ), 400

    if current_password == new_password:

        return jsonify(
            {
                "ok": False,
                "error": (
                    "Your new password must be "
                    "different from the current "
                    "password."
                ),
            }
        ), 400

    if not auth_mod.check_password(
        current_password
    ):

        return jsonify(
            {
                "ok": False,
                "error": (
                    "Current password is incorrect."
                ),
            }
        ), 401

    ok, error = auth_mod.change_password(
        current_password,
        new_password,
    )

    if not ok:

        return jsonify(
            {
                "ok": False,
                "error": (
                    error
                    or "Could not change password."
                ),
            }
        ), 400

    session.clear()

    session["authenticated"] = True
    session["role"] = "user"

    session.permanent = True
    session.modified = True

    token = _create_auth_token(
        "user"
    )

    return jsonify(
        {
            "ok": True,
            "token": token,
            "role": "user",
            "name": "Kenzi Richardson",
            "message": (
                "Password changed successfully."
            ),
        }
    )


# ======================================================================
# MAKER — DASHBOARD STATUS
# ======================================================================

@app.route(
    "/api/maker/status",
    methods=["GET"],
)
def maker_status():

    def maker_status():

    if not _require_maker():
        return jsonify(
            {
                "ok": False,
                "error": "Maker authentication required.",
            }
        ), 401

    return jsonify(
        {
            "ok": True,
            "maker": True,
        }
    )

