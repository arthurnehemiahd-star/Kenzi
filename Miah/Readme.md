# MIAH

A personal voice assistant — Flask backend, browser frontend, cloned voice,
long-term memory, device actions, and optional Spotify control. Built to run
on your own machine for $0, behind a free tunnel.

---

## What it actually is

One Flask app, one JSON file for state, one browser tab for the interface.
MIAH listens (Whisper), thinks (Hugging Face Inference Providers), replies
in a cloned voice (XTTS), remembers across conversations (a single growing
thread plus an auto-summarized profile), and can trigger a small set of
device actions — open another app, open a camera, search/save Spotify.

It is **not** a native app. It is a web page that behaves like one when
added to the home screen.

---

## Honest costs and trade-offs

- **$0 to run.** No paid LLM API. Whisper, XTTS, and Resemblyzer run
  locally on your own hardware; Hugging Face's free tier handles the LLM;
  Cloudflare Tunnel or ngrok handles the URL.
- **The trade-offs are real:**
  - Hugging Face's free tier is rate-limited (a few hundred requests/hour
    depending on the model). Fine for one person's casual use, not for
    heavy back-and-forth.
  - Open models are weaker than Claude at nuance, emotional tone, and
    reliable tool-calling. If device actions stop triggering after you
    change `HF_MODEL`, that's usually why.
  - XTTS on CPU takes **10–30 seconds per reply** to synthesize. On a GPU
    it's a second or two. If that's too slow, you'll want a GPU box or a
    lighter TTS.
  - Uptime is only as good as the machine running it. Leave a laptop or a
    Raspberry Pi on if you want it always reachable.

---

## Requirements

- **Python 3.10 or 3.11.** XTTS and Whisper both break on some 3.12+ setups;
  3.11 is the safe choice.
- **ffmpeg** on your PATH.
- **portaudio** (only needed for the optional `--enroll` CLI path).
- **~6 GB of disk** for PyTorch + the XTTS model + Whisper weights
  (Whisper downloads on first transcription, XTTS on first synthesis).

### Install by OS

**macOS**
```bash
brew install ffmpeg portaudio
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Linux (Debian/Ubuntu)**
```bash
sudo apt update
sudo apt install -y ffmpeg libportaudio2 python3.11 python3.11-venv
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**
```powershell
winget install ffmpeg
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Environment variables

Set these before running. On macOS/Linux, `export` them in your shell or
put them in a `.env` and `source` it. On Windows, use `set` or a `.env`
loaded by your shell.

### Required

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
export SECRET_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')"
```

- `HF_TOKEN` — from https://huggingface.co/settings/tokens, with the
  **"Make calls to Inference Providers"** permission.
- `SECRET_KEY` — any long random string. **If you don't set this, sessions
  reset on every restart** and everyone gets logged out. For a gift, set it
  once and keep it.

### Optional

```bash
export HF_MODEL="meta-llama/Llama-3.3-70B-Instruct"   # default; change if you want
export VOICE_LOGIN_MIN_DAYS="4"                        # 3–5 recommended
export PORT="5000"

# Spotify — only if you want search_music / save_music
export SPOTIFY_CLIENT_ID="..."
export SPOTIFY_CLIENT_SECRET="..."
export SPOTIFY_REDIRECT_URI="https://<your-url>/spotify/callback"
```

---

## First run

```bash
cd miah
python app.py
```

Open http://localhost:5000.

The app walks through setup on its own:

1. **Voice enrollment screen** — appears for whoever opens it first,
   *before* any password exists. Records ~10–20 seconds in the browser,
   converts it via ffmpeg into the WAV MIAH clones from, and saves it
   permanently. This should be **you**, first.
2. **Password screen** — appears next, once voice is enrolled. She sets her
   own password here. Only happens once.
3. **Login screen** — from then on, opening the URL just shows a password
   field.

**Practically:** enroll your voice yourself, then send her the link and let
her land on the password screen.

### Optional: enroll from the command line instead

If you'd rather do the voice enrollment from the terminal (mic on the
machine running the server), you can:

```bash
python app.py --enroll
```

This uses `sounddevice` to record 25 seconds into `miah_data/owner_voice.wav`
and exits. The in-app flow is the primary path; this is just an alternative.

---

## Free hosting

MIAH is only reachable while the script is running. There are two free ways
to give it a public URL.

### Option A — Cloudflare Tunnel (most reliable, no time limit)

1. Install `cloudflared`:
   https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/
2. Run MIAH locally:
   ```bash
   python app.py
   ```
3. In another terminal:
   ```bash
   cloudflared tunnel --url http://localhost:5000
   ```
4. It prints a URL like `https://random-words.trycloudflare.com`. Send her
   that. Stays live as long as both keep running.

### Option B — ngrok (simplest to start)

1. Sign up free at https://ngrok.com and install it.
2. Run MIAH locally:
   ```bash
   python app.py
   ```
3. In another terminal:
   ```bash
   ngrok http 5000
   ```
4. Gives a URL like `https://abcd1234.ngrok-free.app`. Free-tier URLs change
   on restart and sessions can time out after a few hours.

---

## Spotify setup (optional)

Only needed if you want MIAH to search or save songs.

1. Create a free app at https://developer.spotify.com/dashboard
2. Add a Redirect URI: `https://<your-url>/spotify/callback`
3. Set `SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET`, `SPOTIFY_REDIRECT_URI`
4. Once MIAH is running and logged in, visit `<your-url>/spotify/login` once
   to connect the account.

A free Spotify account is enough — search and save-to-library don't need
Premium.

---

## Adding music to the local library

Drop audio files into `miah_data/music/`. Supported extensions: `.mp3`,
`.wav`, `.m4a`, `.ogg`, `.flac`. They'll show up in the **Music** tab.

---

## What works, and what can't

**Works:**
- Hands-free voice conversation (tap the mic once, then just talk)
- Text chat
- Camera (opens inside MIAH; saves via share sheet or download)
- Spotify search and save-to-library
- Open another app for calling, Maps, email
- Run an iOS Shortcut by name **(iOS only)**
- Remember notes and conversation context across sessions
- Voice login after a few days of use

**Can't (by design, not by bug):**
- Control another app's UI on her behalf (e.g. automating TikTok)
- Fully hands-free phone calls — the final tap has to come from her
- Observe the result of a device action it triggers (it can start it, not
  see what happened)
- Run on Android with iOS Shortcuts — that tool is excluded on non-iOS

---

## Known limitations

- **Single-user.** The DB is a JSON file with one password. Fine for a
  personal gift; not built for multiple people.
- **One worker only.** Run with `gunicorn -w 1 --threads 4`. Multiple
  worker *processes* would each hold their own file lock and clobber each
  other.
- **XTTS is slow on CPU.** Expect 10–30 seconds per reply. GPU or a lighter
  TTS is the fix if that matters.
- **Rate limits.** Hugging Face's free tier caps requests per hour. If you
  hit it, MIAH will return an LLM error — wait a bit, or point `HF_MODEL` at
  a smaller model.
- **No streaming.** Replies are generated whole, then synthesized whole.
  First-token latency is what it is.

---

## Project layout

```
miah/
├── app.py              Flask app, routes, system prompt
├── llm.py              Hugging Face call + tool-loop helper
├── tools.py            tool definitions + execute_tool + Spotify helpers
├── voice.py            Whisper, XTTS, Resemblyzer wrappers
├── db.py               JSON store + memory summarization
├── auth.py             password hashing, voice login, setup
├── music.py            local library + Spotify OAuth
├── config.py           every env var and path, in one place
├── requirements.txt
├── README.md
└── static/
    └── index.html      the whole frontend
```

`miah_data/` is created at first run and holds the DB, your voice reference,
and the music library. Back it up if you care about the conversation history.

---

## How to hand it to her

1. Run MIAH somewhere it'll stay up (your laptop, or a Pi).
2. Start the tunnel, get the URL.
3. Enroll your voice yourself, first.
4. Send her the URL. She lands on the password screen and sets her own.
5. Tell her to **Add to Home Screen** (iOS: Share menu → Add to Home Screen;
   Android: Chrome → menu → Install app). It'll open fullscreen like a real
   app.

---

## Packaging it yourself (since I can't hand you a zip)

From the folder containing `miah/`:

**macOS / Linux:**
```bash
zip -r miah.zip miah
```

**Windows (PowerShell):**
```powershell
Compress-Archive -Path miah -DestinationPath miah.zip
```

That produces one file you can move, email, or upload. To run it somewhere
else, unzip and follow the install steps above.