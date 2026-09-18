"""Tool definitions and their execution.

Every tool returns (result_text_for_model, action_or_None).

`action` is a small dict the frontend carries out — the backend can't
reach her phone directly, so it hands the trigger back to the browser.

Action types:
    {"type": "open_url",    "url": "..."}
    {"type": "open_camera"}
"""

import os

from urllib.parse import quote

import requests

from config import (
    ALLOWED_AUDIO_EXTENSIONS,
    MUSIC_DIR,
    SPOTIFY_API_BASE,
    SPOTIFY_CLIENT_ID,
    SPOTIFY_CLIENT_SECRET,
    SPOTIFY_TOKEN_URL,
)


# ----------------------------------------------------------------------
# Tool definitions
# ----------------------------------------------------------------------
TOOLS = [
    {
        "name": "remember_note",
        "description": (
            "Save a short note or fact for later, so it can be recalled in a "
            "future conversation. Use this when the user shares something "
            "worth remembering (a preference, a plan, a reminder)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "note": {
                    "type": "string",
                    "description": "The note to remember, in plain words.",
                }
            },
            "required": ["note"],
        },
    },

    {
        "name": "recall_notes",
        "description": (
            "Retrieve previously saved notes. Use this when the user asks "
            "what you remember, or when past notes would help answer their "
            "question."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },

    {
        "name": "web_search",
        "description": (
            "Search the live web for information when the user's question "
            "needs current, recent, changing, or hard-to-know information. "
            "Examples include today's news, current events, current prices, "
            "recent software changes, current sports information, newly "
            "released products, current weather information, or facts that "
            "may have changed since the model's knowledge. Use this tool "
            "instead of telling the user to open Google or another search "
            "engine. The search happens privately in the backend and the "
            "results are returned to MIAH so she can answer directly in the "
            "conversation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A clear web search query containing the information "
                        "you need to find."
                    ),
                }
            },
            "required": ["query"],
        },
    },

    {
        "name": "list_music",
        "description": (
            "List the songs available in MIAH's music library. Use this when "
            "the user asks what music is available, or asks you to play "
            "something."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },

    {
        "name": "run_shortcut",
        "description": (
            "Run an iOS Shortcut by name to perform an action on the device — "
            "e.g. sending a message, controlling smart home devices, setting "
            "a timer or alarm, or any multi-step automation. Only works for a "
            "Shortcut the user has already created in Apple's Shortcuts app "
            "under this exact name. Use this when the user asks you to do "
            "something that requires acting on their phone in a way beyond "
            "simple app-opening."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "shortcut_name": {
                    "type": "string",
                    "description": (
                        "The exact name of the Shortcut as it appears in "
                        "the Shortcuts app."
                    ),
                }
            },
            "required": ["shortcut_name"],
        },
    },

    {
        "name": "open_app",
        "description": (
            "Open another app on the device for a simple, single action — "
            "calling someone, opening Maps with a search, composing an email, "
            "or opening a website. Use this for quick one-off actions; use "
            "run_shortcut instead for anything more complex."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": (
                        "The URL or URL scheme to open, e.g. "
                        "'tel:+15551234567', "
                        "'https://www.google.com/maps/search/?api=1&query=coffee+near+me', "
                        "'mailto:someone@example.com', or a regular https:// link."
                    ),
                }
            },
            "required": ["url"],
        },
    },

    {
        "name": "open_camera",
        "description": (
            "Open MIAH's built-in camera so the user can take a photo. Use "
            "this when the user asks you to open the camera or take a picture."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },

    {
        "name": "search_music",
        "description": (
            "Search Spotify for songs matching a query and list the top "
            "matches, without saving anything. Use this when the user asks "
            "what's available, or wants to hear options before choosing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Song, artist, or search terms.",
                }
            },
            "required": ["query"],
        },
    },

    {
        "name": "save_music",
        "description": (
            "Search Spotify for a song and save the top match to the user's "
            "Spotify library (Liked Songs). Use this when the user asks you "
            "to save, like, or add a song."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Song and/or artist to search for and save."
                    ),
                }
            },
            "required": ["query"],
        },
    },
]


def get_tools_for_platform(platform):
    """Return tools appropriate for the user's platform.

    run_shortcut only works on iOS — omit it entirely for other platforms.
    Web search is available on every platform because the search happens
    on the MIAH backend.
    """
    if platform == "ios":
        return TOOLS

    return [
        tool
        for tool in TOOLS
        if tool["name"] != "run_shortcut"
    ]


# ----------------------------------------------------------------------
# Web search
# ----------------------------------------------------------------------
def web_search(query, limit=5):
    """Search the live web and return concise results for the AI.

    The search is performed by the backend. Nothing is opened in the
    user's browser.

    Returns:
        A text summary containing titles, URLs and snippets.
    """

    query = (query or "").strip()

    if not query:
        return "No web search query was provided."

    try:
        # ddgs is imported here so the rest of MIAH still starts normally
        # if the package is temporarily unavailable.
        from ddgs import DDGS

        results = DDGS().text(
            query,
            max_results=limit,
        )

    except Exception as e:
        print("[web_search] Search failed:", repr(e))
        return (
            "The live web search could not be completed right now. "
            f"Search error: {e}"
        )

    if not results:
        return f"No web results were found for: {query}"

    lines = [
        f"Web search results for: {query}",
        "",
    ]

    for index, result in enumerate(results[:limit], start=1):
        title = (
            result.get("title")
            or "Untitled result"
        ).strip()

        url = (
            result.get("href")
            or result.get("url")
            or ""
        ).strip()

        snippet = (
            result.get("body")
            or result.get("snippet")
            or ""
        ).strip()

        lines.append(f"{index}. {title}")

        if url:
            lines.append(f"URL: {url}")

        if snippet:
            lines.append(f"Summary: {snippet}")

        lines.append("")

    return "\n".join(lines)


# ----------------------------------------------------------------------
# Spotify helpers
#
# These take the caller's already-loaded `db` dict and mutate it in place
# rather than doing their own load/save round trip. chat() loads db once,
# may run several tools against it, then saves once at the end.
# ----------------------------------------------------------------------
def get_spotify_access_token(db):
    """Return a valid access token, refreshing if needed.

    Mutates db['spotify'] in place on refresh.

    Returns None if Spotify has never been connected.
    """
    import time

    tokens = db.get("spotify", {})

    if not tokens.get("refresh_token"):
        return None

    if (
        tokens.get("access_token")
        and tokens.get("expires_at", 0) > time.time() + 30
    ):
        return tokens["access_token"]

    resp = requests.post(
        SPOTIFY_TOKEN_URL,
        data={
            "grant_type": "refresh_token",
            "refresh_token": tokens["refresh_token"],
            "client_id": SPOTIFY_CLIENT_ID,
            "client_secret": SPOTIFY_CLIENT_SECRET,
        },
    )

    resp.raise_for_status()

    data = resp.json()

    tokens["access_token"] = data["access_token"]

    tokens["expires_at"] = (
        time.time()
        + data.get("expires_in", 3600)
    )

    if "refresh_token" in data:
        tokens["refresh_token"] = data["refresh_token"]

    db["spotify"] = tokens

    return tokens["access_token"]


def spotify_search_tracks(db, query, limit=5):
    """Return a list of track dicts, or None if Spotify isn't connected."""
    token = get_spotify_access_token(db)

    if not token:
        return None

    resp = requests.get(
        f"{SPOTIFY_API_BASE}/search",
        params={
            "q": query,
            "type": "track",
            "limit": limit,
        },
        headers={
            "Authorization": f"Bearer {token}"
        },
        timeout=15,
    )

    resp.raise_for_status()

    return resp.json().get(
        "tracks",
        {},
    ).get(
        "items",
        [],
    )


# ----------------------------------------------------------------------
# Tool execution
# ----------------------------------------------------------------------
def execute_tool(tool_name, tool_input, db):
    """Run one tool call.

    Returns:
        (result_text_for_model, action_or_None)

    Mutates `db` in place; the caller is responsible for saving it.
    """

    # ------------------------------------------------------------------
    # MEMORY
    # ------------------------------------------------------------------
    if tool_name == "remember_note":
        note = (
            tool_input.get("note") or ""
        ).strip()

        if not note:
            return "Nothing to save.", None

        db.setdefault(
            "notes",
            []
        ).append(note)

        return "Saved.", None

    # ------------------------------------------------------------------
    # RECALL MEMORY
    # ------------------------------------------------------------------
    if tool_name == "recall_notes":
        notes = db.get(
            "notes",
            []
        )

        if not notes:
            return "No notes saved yet.", None

        return "\n".join(
            f"- {note}"
            for note in notes
        ), None

    # ------------------------------------------------------------------
    # WEB SEARCH
    # ------------------------------------------------------------------
    if tool_name == "web_search":
        query = (
            tool_input.get("query") or ""
        ).strip()

        if not query:
            return (
                "No web search query was provided.",
                None,
            )

        result = web_search(
            query,
            limit=5,
        )

        # Important:
        # No frontend action is returned.
        # The browser does NOT open Google/Bing/etc.
        return result, None

    # ------------------------------------------------------------------
    # LOCAL MUSIC LIBRARY
    # ------------------------------------------------------------------
    if tool_name == "list_music":
        if not os.path.isdir(MUSIC_DIR):
            return (
                "No music library folder found yet.",
                None,
            )

        tracks = [
            filename
            for filename in os.listdir(MUSIC_DIR)
            if os.path.splitext(filename)[1].lower()
            in ALLOWED_AUDIO_EXTENSIONS
        ]

        if not tracks:
            return (
                "The music library is empty.",
                None,
            )

        return "\n".join(
            f"- {track}"
            for track in sorted(tracks)
        ), None

    # ------------------------------------------------------------------
    # IOS SHORTCUTS
    # ------------------------------------------------------------------
    if tool_name == "run_shortcut":
        name = (
            tool_input.get("shortcut_name")
            or ""
        ).strip()

        if not name:
            return (
                "No shortcut name given.",
                None,
            )

        scheme_url = (
            "shortcuts://run-shortcut?name="
            f"{quote(name)}"
        )

        return (
            f"Running the '{name}' shortcut now.",
            {
                "type": "open_url",
                "url": scheme_url,
            },
        )

    # ------------------------------------------------------------------
    # OPEN APP / URL
    # ------------------------------------------------------------------
    if tool_name == "open_app":
        url = (
            tool_input.get("url")
            or ""
        ).strip()

        if not url:
            return (
                "No URL given.",
                None,
            )

        return (
            "Opening that now.",
            {
                "type": "open_url",
                "url": url,
            },
        )

    # ------------------------------------------------------------------
    # CAMERA
    # ------------------------------------------------------------------
    if tool_name == "open_camera":
        return (
            "Opening the camera now.",
            {
                "type": "open_camera",
            },
        )

    # ------------------------------------------------------------------
    # SPOTIFY SEARCH
    # ------------------------------------------------------------------
    if tool_name == "search_music":
        query = (
            tool_input.get("query")
            or ""
        ).strip()

        if not query:
            return (
                "No music search query was provided.",
                None,
            )

        try:
            items = spotify_search_tracks(
                db,
                query,
                limit=5,
            )

        except requests.RequestException as e:
            return (
                f"Spotify search failed: {e}",
                None,
            )

        if items is None:
            return (
                "Spotify isn't connected yet — she needs "
                "to connect it once.",
                {
                    "type": "open_url",
                    "url": "/spotify/login",
                },
            )

        if not items:
            return (
                f"No results found for '{query}'.",
                None,
            )

        lines = [
            (
                f"- {track['name']} by "
                f"{', '.join(artist['name'] for artist in track['artists'])}"
            )
            for track in items
        ]

        return "\n".join(lines), None

    # ------------------------------------------------------------------
    # SPOTIFY SAVE
    # ------------------------------------------------------------------
    if tool_name == "save_music":
        query = (
            tool_input.get("query")
            or ""
        ).strip()

        if not query:
            return (
                "No music search query was provided.",
                None,
            )

        try:
            items = spotify_search_tracks(
                db,
                query,
                limit=1,
            )

        except requests.RequestException as e:
            return (
                f"Spotify search failed: {e}",
                None,
            )

        if items is None:
            return (
                "Spotify isn't connected yet — she needs "
                "to connect it once.",
                {
                    "type": "open_url",
                    "url": "/spotify/login",
                },
            )

        if not items:
            return (
                f"Couldn't find '{query}' on Spotify.",
                None,
            )

        track = items[0]

        token = get_spotify_access_token(db)

        try:
            resp = requests.put(
                f"{SPOTIFY_API_BASE}/me/tracks",
                params={
                    "ids": track["id"]
                },
                headers={
                    "Authorization": f"Bearer {token}"
                },
                timeout=15,
            )

            resp.raise_for_status()

        except requests.RequestException as e:
            return (
                f"Couldn't save to Spotify: {e}",
                None,
            )

        artist_names = ", ".join(
            artist["name"]
            for artist in track["artists"]
        )

        return (
            f"Saved '{track['name']}' by "
            f"{artist_names} to your Spotify library.",
            None,
        )

    # ------------------------------------------------------------------
    # UNKNOWN TOOL
    # ------------------------------------------------------------------
    return (
        f"Unknown tool: {tool_name}",
        None,
    )
