"""
YouTube Summary Generator
=========================
Streamlit app that takes a YouTube video URL, extracts its transcript,
and generates a short summary using Generative AI (Groq or Google Gemini).
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit command)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="YouTube Summary Generator",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Helpers – URL / video ID
# ---------------------------------------------------------------------------
def extract_video_id(url: str) -> Optional[str]:
    """Extract the 11-character YouTube video ID from common URL formats."""
    url = url.strip()
    if not url:
        return None

    # youtu.be/VIDEO_ID
    short = re.search(r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    if short:
        return short.group(1)

    # youtube.com/watch?v=VIDEO_ID  |  /embed/  |  /shorts/  |  /v/
    patterns = [
        r"(?:youtube\.com/watch\?v=)([a-zA-Z0-9_-]{11})",
        r"(?:youtube\.com/embed/)([a-zA-Z0-9_-]{11})",
        r"(?:youtube\.com/shorts/)([a-zA-Z0-9_-]{11})",
        r"(?:youtube\.com/v/)([a-zA-Z0-9_-]{11})",
        r"(?:youtube\.com/live/)([a-zA-Z0-9_-]{11})",
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            return m.group(1)

    # Fallback: query string
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    if "v" in qs and len(qs["v"][0]) == 11:
        return qs["v"][0]

    # Bare 11-char ID
    if re.fullmatch(r"[a-zA-Z0-9_-]{11}", url):
        return url

    return None


# ---------------------------------------------------------------------------
# Transcript extraction
# ---------------------------------------------------------------------------
def fetch_transcript(video_id: str, languages: Optional[List[str]] = None) -> Tuple[str, str]:
    """
    Fetch transcript text for a video (youtube-transcript-api v1.2+).
    Returns (full_text, language_code).
    Raises a clear Exception on failure.
    """
    from youtube_transcript_api import YouTubeTranscriptApi

    languages = languages or ["en", "en-US", "en-GB", "en-IN"]

    try:
        ytt = YouTubeTranscriptApi()

        # Preferred path: direct fetch for preferred languages
        try:
            fetched = ytt.fetch(video_id, languages=languages)
        except Exception:
            # Fallback: list available transcripts, pick first usable (any language)
            transcript_list = ytt.list(video_id)
            transcript = None
            try:
                transcript = transcript_list.find_transcript(languages)
            except Exception:
                try:
                    transcript = transcript_list.find_generated_transcript(languages)
                except Exception:
                    # Take the first available transcript of any language
                    for t in transcript_list:
                        transcript = t
                        break
            if transcript is None:
                raise RuntimeError("No transcript found for this video.")
            fetched = transcript.fetch()

        # Normalize FetchedTranscript / list / raw data into plain text
        texts: List[str] = []
        lang = languages[0]

        if hasattr(fetched, "to_raw_data"):
            raw = fetched.to_raw_data()
            texts = [item.get("text", "") for item in raw if isinstance(item, dict)]
            lang = getattr(fetched, "language_code", lang) or lang
        elif hasattr(fetched, "snippets"):
            texts = [getattr(s, "text", str(s)) for s in fetched.snippets]
            lang = getattr(fetched, "language_code", lang) or lang
        elif isinstance(fetched, list):
            for item in fetched:
                if isinstance(item, dict):
                    texts.append(item.get("text", ""))
                elif hasattr(item, "text"):
                    texts.append(item.text)
                else:
                    texts.append(str(item))
        else:
            texts = [str(fetched)]

        full = " ".join(t.strip() for t in texts if t and t.strip()).strip()
        if not full:
            raise RuntimeError("Transcript was empty.")
        return full, lang

    except Exception as e:
        name = type(e).__name__
        msg = str(e)
        if "TranscriptsDisabled" in name or "disabled" in msg.lower():
            raise RuntimeError(
                "Transcripts are disabled for this video. "
                "The creator has turned off captions."
            ) from e
        if "NoTranscriptFound" in name or "no transcript" in msg.lower():
            raise RuntimeError(
                "No transcript found. The video may not have captions."
            ) from e
        if "VideoUnavailable" in name:
            raise RuntimeError(
                "This video is unavailable (private, deleted, or region-blocked)."
            ) from e
        if "IpBlocked" in name or "RequestBlocked" in name or "blocked" in msg.lower():
            raise RuntimeError(
                "YouTube blocked the request (common on cloud IPs). "
                "Try running the app locally, or use another video with public captions."
            ) from e
        raise RuntimeError(f"Could not fetch transcript: {msg}") from e


def chunk_text(text: str, max_chars: int = 12000) -> List[str]:
    """Split long text into overlapping chunks that fit model context."""
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        if end < len(text):
            # break on sentence boundary if possible
            split_at = text.rfind(". ", start, end)
            if split_at > start + max_chars // 2:
                end = split_at + 1
        chunks.append(text[start:end].strip())
        start = end
    return chunks


# ---------------------------------------------------------------------------
# Generative AI summarization
# ---------------------------------------------------------------------------
SUMMARY_STYLES = {
    "Concise": (
        "Write a concise summary of the following YouTube video transcript in 3–6 bullet points. "
        "Focus only on the main ideas. Keep the total under 150 words. "
        "Do not invent information that is not in the transcript.\n\nTranscript:\n{transcript}"
    ),
    "Detailed": (
        "Write a detailed but well-structured summary of the following YouTube video transcript. "
        "Use short paragraphs or numbered points covering the introduction, key arguments, "
        "examples, and conclusion. Aim for 250–400 words. "
        "Do not invent information that is not in the transcript.\n\nTranscript:\n{transcript}"
    ),
    "ELI5": (
        "Explain the content of this YouTube video transcript as if to a 12-year-old. "
        "Use simple words, short sentences, and everyday analogies. "
        "Keep it under 200 words. Do not invent information.\n\nTranscript:\n{transcript}"
    ),
    "Key takeaways": (
        "Extract the top 5–8 key takeaways from this YouTube video transcript. "
        "Each takeaway should be one clear sentence. "
        "Do not invent information that is not in the transcript.\n\nTranscript:\n{transcript}"
    ),
}


def summarize_with_groq(transcript: str, api_key: str, style: str, model: str) -> str:
    from groq import Groq

    client = Groq(api_key=api_key)
    prompt_template = SUMMARY_STYLES[style]
    chunks = chunk_text(transcript, max_chars=14000)

    if len(chunks) == 1:
        prompt = prompt_template.format(transcript=chunks[0])
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes video transcripts accurately and clearly."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        return completion.choices[0].message.content.strip()

    # Multi-chunk: summarize each, then combine
    partials = []
    for i, chunk in enumerate(chunks, 1):
        prompt = (
            f"Summarize this part ({i}/{len(chunks)}) of a YouTube transcript "
            f"in a few clear bullet points. Keep it short.\n\n{chunk}"
        )
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes video transcripts accurately."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=512,
        )
        partials.append(completion.choices[0].message.content.strip())

    combined = "\n\n".join(partials)
    final_prompt = prompt_template.format(transcript=combined)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that merges partial summaries into one coherent summary."},
            {"role": "user", "content": final_prompt},
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    return completion.choices[0].message.content.strip()


def summarize_with_gemini(transcript: str, api_key: str, style: str, model: str) -> str:
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    gm = genai.GenerativeModel(model)
    prompt_template = SUMMARY_STYLES[style]
    chunks = chunk_text(transcript, max_chars=12000)

    if len(chunks) == 1:
        prompt = prompt_template.format(transcript=chunks[0])
        response = gm.generate_content(prompt)
        return response.text.strip()

    partials = []
    for i, chunk in enumerate(chunks, 1):
        prompt = (
            f"Summarize this part ({i}/{len(chunks)}) of a YouTube transcript "
            f"in a few clear bullet points. Keep it short.\n\n{chunk}"
        )
        response = gm.generate_content(prompt)
        partials.append(response.text.strip())

    combined = "\n\n".join(partials)
    final_prompt = prompt_template.format(transcript=combined)
    response = gm.generate_content(final_prompt)
    return response.text.strip()


def generate_summary(
    transcript: str,
    provider: str,
    api_key: str,
    style: str,
    model: str,
) -> str:
    if not api_key or not api_key.strip():
        raise RuntimeError(
            "Please enter a valid API key in the sidebar. "
            "Get a free key from Groq (console.groq.com) or Google AI Studio."
        )
    if provider == "Groq":
        return summarize_with_groq(transcript, api_key.strip(), style, model)
    if provider == "Google Gemini":
        return summarize_with_gemini(transcript, api_key.strip(), style, model)
    raise RuntimeError(f"Unknown provider: {provider}")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def main() -> None:
    # ----- Sidebar -----
    with st.sidebar:
        st.title("⚙️ Settings")
        st.markdown("---")

        provider = st.selectbox(
            "AI Provider",
            options=["Groq", "Google Gemini"],
            index=0,
            help="Groq is free & fast. Gemini also has a free tier.",
        )

        # Prefer keys stored in Streamlit secrets (Cloud-friendly & safer)
        secret_groq = ""
        secret_gemini = ""
        try:
            secret_groq = st.secrets.get("GROQ_API_KEY", "") or st.secrets.get("groq_api_key", "")
        except Exception:
            pass
        try:
            secret_gemini = st.secrets.get("GOOGLE_API_KEY", "") or st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            pass

        if provider == "Groq":
            model = st.selectbox(
                "Model",
                options=[
                    "openai/gpt-oss-20b",       # fast, free-tier (replaces deprecated llama-3.1-8b-instant)
                    "openai/gpt-oss-120b",      # stronger
                    "qwen/qwen3.8-27b",         # alternative
                    "llama-3.1-8b-instant",     # enterprise only after Aug 2026
                    "llama-3.3-70b-versatile",  # enterprise only after Aug 2026
                ],
                index=0,
                help="Use openai/gpt-oss-20b — older Llama model IDs were removed from free tier in Aug 2026",
            )
            if secret_groq:
                api_key = secret_groq
                st.success("Using Groq key from Streamlit secrets")
            else:
                api_key = st.text_input(
                    "Groq API Key",
                    type="password",
                    placeholder="gsk_...",
                    help="Get a free key at https://console.groq.com — or add GROQ_API_KEY in Streamlit Cloud secrets",
                )
                st.caption("Free tier: [console.groq.com](https://console.groq.com)")
        else:
            model = st.selectbox(
                "Model",
                options=["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
                index=0,
            )
            if secret_gemini:
                api_key = secret_gemini
                st.success("Using Gemini key from Streamlit secrets")
            else:
                api_key = st.text_input(
                    "Google AI API Key",
                    type="password",
                    placeholder="AIza...",
                    help="Get a free key at https://aistudio.google.com/apikey — or add GOOGLE_API_KEY in Streamlit Cloud secrets",
                )
                st.caption("Free tier: [aistudio.google.com](https://aistudio.google.com/apikey)")

        style = st.selectbox(
            "Summary style",
            options=list(SUMMARY_STYLES.keys()),
            index=0,
        )

        st.markdown("---")
        st.markdown(
            """
            **How it works**
            1. Paste a YouTube link  
            2. We extract the transcript (no key needed)  
            3. Generative AI creates a short summary  

            **Notes**  
            - Some videos have no captions.  
            - On Streamlit Cloud, add your key under **App settings → Secrets**.  
            - YouTube may block transcript requests from cloud IPs; if that happens, try another video or run locally.
            """
        )

    # ----- Main area -----
    st.title("📺 YouTube Summary Generator")
    st.caption("Paste a YouTube link → get a Generative-AI summary of the video")

    url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=... or https://youtu.be/...",
    )

    with st.expander("📋 Paste transcript manually (recommended on Streamlit Cloud)", expanded=False):
        st.markdown(
            """
            YouTube often **blocks automatic transcript downloads from cloud servers**.  
            Workaround:
            1. Open the video on YouTube → **⋯** → **Show transcript**
            2. Copy the text and paste it below
            3. Click **Generate Summary**
            """
        )
        manual_transcript = st.text_area(
            "Paste transcript here",
            height=160,
            placeholder="Paste the full video transcript text…",
            key="manual_transcript",
        )

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        generate = st.button("✨ Generate Summary", type="primary", use_container_width=True)

    if not generate:
        st.info(
            "Enter a YouTube URL (and optionally paste the transcript) then click **Generate Summary**. "
            "Add your free API key in the sidebar first."
        )
        with st.expander("Example videos that usually have captions"):
            st.markdown(
                """
                - https://www.youtube.com/watch?v=aircAruvnKk  (3Blue1Brown – Neural Networks)  
                - https://www.youtube.com/watch?v=8jLOx1hD3_o  (freeCodeCamp – Python)  
                - Any TED Talk or official channel video with closed captions
                """
            )
        return

    # ----- Validation -----
    video_id = extract_video_id(url) if url.strip() else None
    manual = (manual_transcript or "").strip()

    if not video_id and not manual:
        st.error("Please paste a YouTube URL and/or a transcript.")
        return

    # ----- Show video if we have an ID -----
    if video_id:
        st.subheader("Video")
        st.video(f"https://www.youtube.com/watch?v={video_id}")
        st.caption(f"Video ID: `{video_id}`")

    # ----- Transcript -----
    with st.status("Working…", expanded=True) as status:
        transcript = ""
        lang = "manual"

        if manual:
            status.update(label="Using pasted transcript…", state="running")
            transcript = manual
            lang = "pasted"
        elif video_id:
            status.update(label="Fetching transcript from YouTube…", state="running")
            try:
                transcript, lang = fetch_transcript(video_id)
            except RuntimeError as e:
                status.update(label="Auto-fetch failed", state="error")
                st.error(str(e))
                st.warning(
                    "**Tip for Streamlit Cloud:** Open the expander above, paste the transcript "
                    "from YouTube (⋯ → Show transcript → copy), then click Generate again."
                )
                return
        else:
            st.error("No transcript available.")
            return

        word_count = len(transcript.split())
        status.update(
            label=f"Transcript ready ({word_count:,} words, source={lang})",
            state="running",
        )

        if word_count < 20:
            status.update(label="Transcript too short", state="error")
            st.warning("The transcript is very short — please paste more text.")
            return

        status.update(label="Generating AI summary…", state="running")
        try:
            summary = generate_summary(transcript, provider, api_key, style, model)
        except Exception as e:
            status.update(label="Summary failed", state="error")
            st.error(f"AI summary failed: {e}")
            return

        status.update(label="Done!", state="complete")

    # ----- Results -----
    st.subheader("📝 Summary")
    st.markdown(summary)

    summary_words = len(summary.split())
    c1, c2, c3 = st.columns(3)
    c1.metric("Transcript words", f"{word_count:,}")
    c2.metric("Summary words", f"{summary_words:,}")
    c3.metric("Compression", f"{(1 - summary_words / max(word_count, 1)) * 100:.0f}%")

    st.download_button(
        label="⬇️ Download summary (.txt)",
        data=summary,
        file_name=f"youtube_summary_{video_id}.txt",
        mime="text/plain",
    )

    with st.expander("View full transcript"):
        st.text_area("Transcript", transcript, height=300)


if __name__ == "__main__":
    main()
