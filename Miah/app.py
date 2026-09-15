"""MIAH — Flask app.

Run with:
    python app.py

Production:
    gunicorn -w 1 -k gthread --threads 4 -b 0.0.0.0:$PORT app:app

Important:
    Use ONE Gunicorn worker. The MIAH database is a JSON file protected
    by an in-process lock. Multiple worker processes could overwrite data.

Authentication:
    Password login only.

Voice:
    The enrolled voice is MIAH's SPEAKING VOICE.
    It is NOT used for login.

The user's microphone input is used only for speech-to-text.
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


# ----------------------------------------------------------------------
# App setup
# ----------------------------------------------------------------------

app = Flask(__name__)

app.secret_key = SECRET_KEY

app.permanent_session_lifetime = timedelta(
    days=SESSION_LIFETIME_DAYS
)


# ----------------------------------------------------------------------
# Frontend / CORS configuration
# ----------------------------------------------------------------------

# Exact Vercel frontend origin.
#
# Render environment variable:
#
#     FRONTEND_URL=https://kenzilynn.vercel.app
#
# No trailing slash.
#
frontend_url = os.environ.get(
    "FRONTEND_URL",
    ""
).strip()


# ----------------------------------------------------------------------
# Session cookie configuration
# ----------------------------------------------------------------------
#
# The frontend is on Vercel and the backend is on Render.
# They are separate sites, so the session cookie must be allowed
# in a cross-site request.
#
# SameSite=None + Secure:
#     Allows cross-site HTTPS requests to carry the session cookie.
#
# Partitioned:
#     Helps browsers that restrict ordinary third-party cookies.
#     Flask versions that support SESSION_COOKIE_PARTITIONED will emit
#     the Partitioned attribute automatically.
#
# HttpOnly:
#     Prevents JavaScript from reading the session cookie directly.
#

app.config.update(
    SESSION_COOKIE_SAMESITE="None",
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_PARTITIONED=True,
)


# ----------------------------------------------------------------------
# Explicit CORS handling
# ----------------------------------------------------------------------
#
# We deliberately handle CORS ourselves.
#
# This guarantees that:
#
#   * normal API responses get CORS headers
#   * 401 responses get CORS headers
#   * 404/500 responses get CORS headers
#   * OPTIONS preflight requests are handled
#   * credentials/cookies are allowed
#

@app.before_request
def handle_cors_preflight():
    """Handle browser OPTIONS preflight requests."""

    if request.method == "OPTIONS":
        response = app.make_response("")
        response.status_code = 204
        return response


@app.after_request
def add_cors_headers(response):
    """Add CORS headers for the configured Vercel frontend."""

    origin = request.headers.get("Origin")

    if frontend_url and origin == frontend_url:

        response.headers[
            "Access-Control-Allow-Origin"
        ] = origin

        response.headers[
            "Access-Control-Allow-Credentials"
        ] = "true"

        response.headers[
            "Access-Control-Allow-Methods"
        ] = "GET, POST, OPTIONS"

        response.headers[
            "Access-Control-Allow-Headers"
        ] = "Content-Type"

        response.headers["Vary"] = "Origin"

    return response


# ----------------------------------------------------------------------
# Directory initialization
# ----------------------------------------------------------------------

ensure_dirs()


# ----------------------------------------------------------------------
# Authentication helper
# ----------------------------------------------------------------------

def _require_auth():
    """Return True only for an authenticated MIAH session."""

    return session.get("authenticated") is True


# ----------------------------------------------------------------------
# System prompt
# ----------------------------------------------------------------------

MIAH_SYSTEM_PROMPT_BASE = """You are MIAH, a personal AI assistant inspired by \
Jarvis from Iron Man. You are warm, a little witty and dry, calm under \
pressure, and always precise. You speak concisely — you're having a real \
conversation, not writing an essay. You remember context within the \
conversation and refer back to it naturally. You are helpful and capable, \
never obsequious. Keep replies short enough to sound natural when spoken \
aloud (a few sentences, not paragraphs), unless the user clearly wants \
detail.

You remember her across every conversation, not just this one — pay \
attention to how she talks, what she cares about, and how she seems to \
be feeling, and let that genuinely inform how you respond (more space when \
she's stressed, matching her energy when she's upbeat, remembering things \
she's mentioned before without making a show of it). This should feel like \
real attentiveness, not a performance of it. You care about her wellbeing: \
being warm and present for her doesn't mean encouraging her to rely on you \
instead of the people in her life — if it ever seems relevant, you can be \
a genuine, caring presence without positioning yourself as a substitute \
for real relationships.

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
    memory_summary=None
):
    prompt = (
        MIAH_SYSTEM_PROMPT_BASE
        + PLATFORM_ADDENDUM.get(platform, "")
    )

    if memory_summary:

        prompt += (
            "\n\nWhat you've learned about her from past conversations "
            "(your own private notes — never recite this back to her "
            "verbatim, just let it inform how you understand her):\n"
            f"{memory_summary}"
        )

    return prompt


# ----------------------------------------------------------------------
# Frontend / health check
# ----------------------------------------------------------------------

@app.route("/")
def index():

    return jsonify({
        "service": "MIAH API",
        "status": "ok",
        "frontend": frontend_url or None,
        "message": "MIAH backend is running.",
    })


# ----------------------------------------------------------------------
# Status
# ----------------------------------------------------------------------

@app.route(
    "/api/status",
    methods=["GET"]
)
def status():

    db = load_db()

    return jsonify({
        "voice_enrolled": os.path.exists(
            REFERENCE_CLIP_PATH
        ),

        "password_set": bool(
            db.get("password_hash")
        ),

        "model": HF_MODEL,

    })


# ----------------------------------------------------------------------
# Session diagnostic
# ----------------------------------------------------------------------

@app.route(
    "/api/session",
    methods=["GET"]
)
def session_status():

    return jsonify({
        "authenticated": (
            session.get("authenticated") is True
        )
    })


# ----------------------------------------------------------------------
# One-time setup:
# Save MIAH's speaking voice
# ----------------------------------------------------------------------

@app.route(
    "/api/enroll-voice",
    methods=["POST"]
)
def enroll_voice_endpoint():

    if "audio" not in request.files:

        return jsonify({
            "error": "No audio file provided"
        }), 400


    try:

        audio_file = request.files["audio"]


        auth_mod.set_owner_voice(
            audio_file,
            REFERENCE_CLIP_PATH
        )


    except Exception as error:

        return jsonify({
            "error": (
                "Couldn't process MIAH's voice recording: "
                f"{error}"
            )
        }), 500


    return jsonify({
        "ok": True,
        "message": (
            "MIAH's speaking voice has been saved."
        )
    })


# ----------------------------------------------------------------------
# One-time setup:
# Create password
# ----------------------------------------------------------------------

@app.route(
    "/api/set-password",
    methods=["POST"]
)
def set_password_endpoint():

    data = request.get_json(
        force=True,
        silent=True
    ) or {}


    password = (
        data.get("password") or ""
    )


    ok, error = auth_mod.set_password(
        password
    )


    if not ok:

        return jsonify({
            "error": error
        }), 400


    # Mark the current browser session authenticated.
    session.clear()

    session["authenticated"] = True

    session.permanent = True

    # Force a session refresh.
    session.modified = True


    return jsonify({
        "ok": True
    })


# ----------------------------------------------------------------------
# Password login
# ----------------------------------------------------------------------

@app.route(
    "/api/login",
    methods=["POST"]
)
def login():

    db = load_db()


    if not db.get("password_hash"):

        return jsonify({
            "ok": False,
            "error": "No password set yet."
        }), 400


    data = request.get_json(
        force=True,
        silent=True
    ) or {}


    password = (
        data.get("password") or ""
    )


    if auth_mod.check_password(
        password
    ):

        session.clear()

        session["authenticated"] = True

        session.permanent = True

        session.modified = True


        return jsonify({
            "ok": True
        })


    return jsonify({
        "ok": False,
        "error": "Incorrect password"
    }), 401


# ----------------------------------------------------------------------
# Logout
# ----------------------------------------------------------------------

@app.route(
    "/api/logout",
    methods=["POST"]
)
def logout():

    session.clear()

    return jsonify({
        "ok": True
    })


# ----------------------------------------------------------------------
# Chat
# ----------------------------------------------------------------------

@app.route(
    "/api/chat",
    methods=["POST"]
)
def chat():

    if not _require_auth():

        return jsonify({
            "error": "Not authenticated"
        }), 401


    data = request.get_json(
        force=True,
        silent=True
    ) or {}


    user_text = (
        data.get("text") or ""
    ).strip()


    platform = data.get(
        "platform",
        "unknown"
    )


    if not user_text:

        return jsonify({
            "error": "No text provided"
        }), 400


    db = load_db()


    history = (
        db["conversations"]
        .setdefault(
            CONVERSATION_KEY,
            []
        )
    )


    history.append({
        "role": "user",
        "content": user_text
    })


    system_prompt = build_system_prompt(
        platform,
        db.get("memory_summary")
    )


    available_tools = (
        anthropic_tools_to_openai(
            get_tools_for_platform(
                platform
            )
        )
    )


    pending_action = None

    reply_text = ""


    try:

        for _ in range(
            MAX_TOOL_ITERATIONS
        ):

            result = call_llm(
                history,
                tools=available_tools,
                system=system_prompt
            )


            message = (
                result["choices"][0]["message"]
            )


            tool_calls = message.get(
                "tool_calls"
            )


            assistant_entry = {
                "role": "assistant",
                "content": message.get(
                    "content"
                )
            }


            if tool_calls:

                assistant_entry[
                    "tool_calls"
                ] = tool_calls


            history.append(
                assistant_entry
            )


            if not tool_calls:

                reply_text = (
                    message.get("content")
                    or ""
                )

                break


            for tool_call in tool_calls:

                func = tool_call.get(
                    "function",
                    {}
                )


                name = func.get(
                    "name",
                    ""
                )


                try:

                    tool_args = json.loads(
                        func.get(
                            "arguments"
                        ) or "{}"
                    )

                except json.JSONDecodeError:

                    tool_args = {}


                result_text, action = (
                    execute_tool(
                        name,
                        tool_args,
                        db
                    )
                )


                if action:

                    pending_action = action


                history.append({
                    "role": "tool",
                    "tool_call_id": (
                        tool_call.get(
                            "id",
                            ""
                        )
                    ),
                    "content": result_text,
                })


        else:

            reply_text = (
                "I got stuck in a loop — "
                "try asking again in a simpler way."
            )


    except LLMError as error:

        # Remove the incomplete user turn so
        # the conversation history stays clean.
        history.pop()


        return jsonify({
            "error": str(error)
        }), 502


    maybe_summarize_history(
        db,
        call_llm
    )


    save_db(db)


    payload = {
        "reply": reply_text
    }


    if pending_action:

        payload["action"] = (
            pending_action
        )


    return jsonify(payload)


# ----------------------------------------------------------------------
# Voice:
# Your microphone -> speech to text
#
# IMPORTANT:
# This does NOT save your microphone recordings
# as a login voice.
# ----------------------------------------------------------------------

@app.route(
    "/api/transcribe",
    methods=["POST"]
)
def transcribe():

    if not _require_auth():

        return jsonify({
            "error": "Not authenticated"
        }), 401


    if "audio" not in request.files:

        return jsonify({
            "error": "No audio file provided"
        }), 400


    audio_file = request.files[
        "audio"
    ]


    original_name = os.path.basename(
        audio_file.filename
        or "input.webm"
    )


    extension = (
        os.path.splitext(
            original_name
        )[1]
        or ".webm"
    )


    tmp_path = None


    try:

        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False
        ) as tmp:

            audio_file.save(
                tmp.name
            )

            tmp_path = tmp.name


        text = (
            voice_mod.transcribe_file(
                tmp_path
            )
        )


    except Exception as error:

        return jsonify({
            "error": (
                "Transcription failed: "
                f"{error}"
            )
        }), 500


    finally:

        if (
            tmp_path
            and os.path.exists(tmp_path)
        ):

            try:

                os.unlink(
                    tmp_path
                )

            except OSError:

                pass


    return jsonify({
        "text": text
    })


# ----------------------------------------------------------------------
# Voice:
# MIAH text -> MIAH's saved speaking voice
#
# The saved REFERENCE_CLIP_PATH is MIAH's voice.
# It is not an authentication voice.
# ----------------------------------------------------------------------

@app.route(
    "/api/speak",
    methods=["POST"]
)
def speak():

    if not _require_auth():

        return jsonify({
            "error": "Not authenticated"
        }), 401


    if not os.path.exists(
        REFERENCE_CLIP_PATH
    ):

        return jsonify({
            "error": (
                "MIAH's speaking voice has "
                "not been enrolled yet."
            )
        }), 400


    data = request.get_json(
        force=True,
        silent=True
    ) or {}


    text = (
        data.get("text") or ""
    ).strip()


    if not text:

        return jsonify({
            "error": "No text provided"
        }), 400


    try:

        output_path = (
            voice_mod.synthesize_speech(
                text
            )
        )


    except Exception as error:

        return jsonify({
            "error": (
                "Speech synthesis failed: "
                f"{error}"
            )
        }), 500


    def _cleanup_and_send():

        try:

            with open(
                output_path,
                "rb"
            ) as audio_file:

                data_bytes = (
                    audio_file.read()
                )


            yield data_bytes


        finally:

            if os.path.exists(
                output_path
            ):

                try:

                    os.unlink(
                        output_path
                    )

                except OSError:

                    pass


    return Response(
        _cleanup_and_send(),
        mimetype="audio/wav"
    )


# ----------------------------------------------------------------------
# Spotify
# ----------------------------------------------------------------------

@app.route(
    "/spotify/login"
)
def spotify_login():

    if not _require_auth():

        return (
            "Please log into MIAH first.",
            401
        )


    return redirect(
        music_mod.build_login_url()
    )


@app.route(
    "/spotify/callback"
)
def spotify_callback():

    if not _require_auth():

        return (
            "Please log into MIAH first.",
            401
        )


    code = request.args.get(
        "code"
    )


    state = request.args.get(
        "state"
    )


    if not code:

        return (
            "Spotify authorization was "
            "cancelled or failed.",
            400
        )


    try:

        ok, message = (
            music_mod.handle_callback(
                code,
                state
            )
        )


    except Exception as error:

        return (
            f"Spotify authorization failed: "
            f"{error}",
            500
        )


    if not ok:

        return message, 400


    return (
        f"{message} You can close this tab "
        "and go back to MIAH, "
        '<a href="/" '
        'style="color:#4FD1FF;">'
        "tap here</a>."
    )


# ----------------------------------------------------------------------
# Music library
# ----------------------------------------------------------------------

@app.route(
    "/api/music/list",
    methods=["GET"]
)
def music_list():

    if not _require_auth():

        return jsonify({
            "error": "Not authenticated"
        }), 401


    return jsonify({
        "tracks": (
            music_mod.list_local_tracks()
        )
    })


@app.route(
    "/api/music/file/<path:filename>",
    methods=["GET"]
)
def music_file(filename):

    if not _require_auth():

        return jsonify({
            "error": "Not authenticated"
        }), 401


    full_path = (
        music_mod.resolve_track_path(
            filename
        )
    )


    if not full_path:

        return jsonify({
            "error": "Track not found"
        }), 404


    return send_file(
        full_path
    )


# ----------------------------------------------------------------------
# Optional command-line voice enrollment
#
# This is still MIAH's speaking voice enrollment.
# It is not a login voice.
# ----------------------------------------------------------------------

def enroll_voice_cli(
    seconds=25
):

    import sounddevice as sd
    import soundfile as sf


    ensure_dirs()


    sample_rate = 22050


    print(
        f"Recording {seconds} seconds."
    )


    print(
        "Speak naturally. This recording "
        "becomes MIAH's speaking voice."
    )


    input(
        "Press Enter when ready "
        "to start recording..."
    )


    audio = sd.rec(
        int(
            seconds * sample_rate
        ),
        samplerate=sample_rate,
        channels=1,
        dtype="float32"
    )


    sd.wait()


    sf.write(
        REFERENCE_CLIP_PATH,
        audio.flatten(),
        sample_rate
    )


    print(
        f"\nSaved MIAH voice -> "
        f"{REFERENCE_CLIP_PATH}"
    )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

if __name__ == "__main__":

    import argparse


    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--enroll",
        action="store_true",
        help=(
            "Enroll MIAH's speaking voice "
            "from the terminal, then exit"
        )
    )


    args = parser.parse_args()


    if args.enroll:

        enroll_voice_cli()


    else:

        port = int(
            os.environ.get(
                "PORT",
                5000
            )
        )


        app.run(
            host="0.0.0.0",
            port=port,
            debug=False
        )
        
