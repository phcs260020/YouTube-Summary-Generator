# YouTube Summary Generator

Streamlit application that takes a **YouTube video link** and generates a **short summary** using **Generative AI**.

## Features

- Paste any common YouTube URL (`youtube.com/watch`, `youtu.be`, `/shorts/`, `/embed/`, …)
- **Transcript extraction** with `youtube-transcript-api` (no YouTube API key required)
- **Generative AI summarization** via:
  - **Groq** (free tier, very fast — recommended)
  - **Google Gemini** (free tier)
- Multiple summary styles: Concise, Detailed, ELI5, Key takeaways
- Long-video support (chunk + merge)
- Embedded player, word counts, compression ratio, download `.txt`
- Works with **Streamlit Cloud secrets** (no need to type the key every time)

## Quick Start (local)

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a **public GitHub repository** (only `app.py`, `requirements.txt`, `README.md`).
2. Go to [https://share.streamlit.io](https://share.streamlit.io) (or share.streamlit.io) and sign in with GitHub.
3. Click **New app** → select your repo, branch, and set **Main file path** to `app.py`.
4. Before or after deploy, open **⚙️ App settings → Secrets** and add:

```toml
GROQ_API_KEY = "gsk_your_new_key_here"
```

   (or for Gemini)

```toml
GOOGLE_API_KEY = "AIza_your_key_here"
```

5. Save secrets and wait for the app to reboot.

**Important**
- **Never** commit the API key into GitHub.
- Rotate any key you previously pasted in chat.
- YouTube sometimes blocks transcript requests from cloud server IPs. If you see an “IpBlocked” / transcript error on Cloud, try a different video or run the app locally.

## API Keys (free)

| Provider | Free key |
|----------|----------|
| **Groq** (recommended) | https://console.groq.com |
| **Google Gemini** | https://aistudio.google.com/apikey |

## Project Structure

```
youtube_summary_generator/
├── app.py
├── requirements.txt
└── README.md
```

## License

MIT
