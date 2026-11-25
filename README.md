# hermes-gemini-api-plugin

Google AI Studio Gemini plugin (`gemini-api`) for [Hermes Agent](https://github.com/NousResearch/hermes-agent), powered end-to-end by the official [gemini-api CLI](https://github.com/google-gemini/gemini-api-cli) (Interactions API) under a single `GEMINI_API_KEY`:

1. **Image Generation (`GeminiImageGenProvider`, `image_gen.provider: gemini`)** — **Nano Banana 2.1**, **Nano Banana 2**, **Nano Banana 2 Lite**, and **Nano Banana Pro** generation and multi-reference editing via `gemini-api image` (`-i`, `--aspect-ratio`, `--image-size`, optional Google Search grounding), saving images to `$HERMES_HOME/cache/generated/images/`.
2. **Video Generation & CLI Tools (`GeminiVideoGenProvider` + 6 Tools + Skill)** — Native `video_generate` (`video_gen.provider: gemini`), Lyria music generation, Gemini 3.8 Flash single/two-speaker TTS, multimodal file & YouTube analysis, speaker-diarized transcription (`.md` / `.srt`), Gemini Deep Research interactions, and resumable Files API uploads (up to 2 GB).

## Capabilities Overview

### 1. Image Generation (`image_gen.provider: gemini`)

| Model ID | Display Name | Speed | Capabilities |
| :--- | :--- | :--- | :--- |
| `gemini-nano-banana-2.1` *(default)* | **Nano Banana 2.1** (Gemini Nano Banana 2.1) | Fast | 14 aspect ratios (including `1:4`, `4:1`, `1:8`, `8:1`), `1K` / `2K` / `4K` output resolution, Google Web and Image Search grounding, up to 14 reference images |
| `gemini-3.1-flash-image` | **Nano Banana 2** (Gemini 3.1 Flash Image) | Fast | 14 aspect ratios (including `1:4`, `4:1`, `1:8`, `8:1`), `512` / `1K` / `2K` / `4K` output resolution, Google Web and Image Search grounding, up to 14 reference images |
| `gemini-3.1-flash-lite-image` | **Nano Banana 2 Lite** (Gemini 3.1 Flash Lite Image) | Fastest | Lowest latency & cost; 10 aspect ratios, `1K` output resolution, up to 14 reference images |
| `gemini-3-pro-image` | **Nano Banana Pro** (Gemini 3 Pro Image) | Slower | Highest fidelity & reasoning; 10 aspect ratios, `1K` / `2K` / `4K` output resolution, Google Search grounding, up to 14 reference images |

### 2. Video Generation & `gemini-api` CLI Tools

| Surface | CLI Default Model / Agent (`v0.5.0`) | Description |
| :--- | :--- | :--- |
| `video_generate` (`video_gen.provider: gemini`) | `gemini-omni-1.2-flash` | Text-to-video and image-to-video generation (`--aspect-ratio`, `--resolution`, `-i`) saved to `$HERMES_HOME/cache/videos/` |
| `gemini_music` | `lyria-3.5` | Instrumental and vocal music generation saved to `$HERMES_HOME/cache/generated/audio/` |
| `gemini_tts` | `gemini-3.8-flash-tts` | 24 kHz WAV speech synthesis with single-speaker voices (`Kore`, `Puck`, etc.) and 2-speaker scripts |
| `gemini_analyze` | `gemini-4-hydrogen` | Multimodal Q&A over PDFs, videos, audio, images, CSVs, `files/<id>` handles, and YouTube URLs (seamlessly auto-uploads binary media >14 MB via Files API) |
| `gemini_transcribe` | `gemini-3.5-transcribe` | Speaker-diarized and timestamped transcripts in `md`, `text`, `json`, or `srt` captions (seamlessly auto-uploads media >14 MB) |
| `gemini_deep_research` | `deep-research-preview-04-2026` | Start (`run`, async `background: true` by default), poll (`status`), or stop (`cancel`) Gemini Deep Research & Interactions API agent runs |
| `gemini_files_upload` | Gemini Files API | Resumable upload up to 2 GB (`--wait` until `ACTIVE`), returning a reusable 48-hour `files/<id>` handle |
| Skill `gemini-api-cli` | `SKILL.md` | Read-only reference (`skill_view`) for ad-hoc `gemini-api` commands (`models`, `--dry-run`, `video --resume`, `--help`) |

## Installation

### 1. Install the Hermes Plugin

Install from the Hermes Plugin Catalog:

```bash
hermes plugins install gemini-api
```

Or install directly from GitHub:

```bash
hermes plugins install https://github.com/google-gemini/hermes-gemini-api-plugin
```

> **Migrating from `gemini-image`:** If you previously installed the v0.1.0 `gemini-image` plugin, uninstall it first (`hermes plugins uninstall gemini-image`) before installing `gemini-api` so both do not register the `gemini` image provider concurrently. Your `image_gen.gemini.*` settings and `GEMINI_API_KEY` remain compatible.

### 2. Install `gemini-api` CLI *(required)*

Install [`google-gemini/gemini-api-cli`](https://github.com/google-gemini/gemini-api-cli) (`v0.5.0`+) onto your `PATH` or `~/.hermes/bin/gemini-api` (or opt into `auto_install_cli: true` in `config.yaml` below):

```bash
# Option A: Install with Go 1.22+
go install github.com/google-gemini/gemini-api-cli/cmd/gemini-api@v0.5.0

# Option B: Download a prebuilt archive from GitHub Releases and verify checksums.txt
# https://github.com/google-gemini/gemini-api-cli/releases/tag/v0.5.0
```

## Configuration

### 1. API Key (`.env`)

Set your Google AI Studio API key ([get one here](https://aistudio.google.com/apikey)) in `~/.hermes/.env` or your shell environment:

```bash
GEMINI_API_KEY="your-api-key"
```

### 2. Provider & Plugin Settings (`~/.hermes/config.yaml`)

```yaml
image_gen:
  provider: gemini
  gemini:
    model: gemini-nano-banana-2.1   # or gemini-3.1-flash-image, gemini-3.1-flash-lite-image, gemini-3-pro-image
    image_size: 2K                  # 1K | 2K | 4K (or 512 on gemini-3.1-flash-image)
    aspect_ratio: "16:9"            # exact ratio override (e.g. 1:1, 16:9, 9:16, 21:9, 1:4, 4:1, 1:8, 8:1)
    google_search: false            # enable Google Search grounding on supported models

video_gen:
  provider: gemini
  gemini:
    model: ""                       # optional explicit override (empty uses gemini-api CLI's built-in default)

plugins:
  entries:
    gemini-api:
      settings:
        cli_path: ""                # optional explicit path to gemini-api binary
        # timeout_seconds: 180      # optional override (when omitted: 180s TTS, 300s music/analyze/transcribe/upload, 900s video/sync research)
        auto_install_cli: false     # opt-in download of pinned v0.5.0 verified against hardcoded SHA-256 digests into $HERMES_HOME/bin
```

Optional environment variable overrides for image model and endpoint:

- `GEMINI_IMAGE_MODEL` — Override the Gemini image model ID (strips any optional `google/` prefix).
- `GEMINI_BASE_URL` — Override the Gemini API base URL (`https://` required; `http://` allowed for localhost testing only).

## Security & Privacy Disclosure

- **External network calls**: All Gemini API requests (image, video, music, TTS, analyze, transcribe, Deep Research, and Files upload) are sent by `gemini-api` over HTTPS to `generativelanguage.googleapis.com` (or a validated `https://` `image_gen.gemini.base_url` / `GEMINI_BASE_URL` — cleartext non-loopback `http://` URLs are refused) authenticated via `GEMINI_API_KEY`. Remote `http(s)://` reference image URLs are fetched through Hermes's SSRF-safe HTTP client. If `auto_install_cli: true` is explicitly opted into by the user, downloads the pinned `gemini-api` `v0.5.0` release archive (capped at 64 MiB) from `github.com/google-gemini/gemini-api-cli/releases` and verifies it against hardcoded SHA-256 digests before atomic extraction.
- **Subprocess execution**: Invokes the local `gemini-api` binary (`--no-interactive -o json`, `stdin=DEVNULL`, `--` end-of-flags guard on positional prompts/paths) inside an allowlisted, secret-scrubbed environment carrying only the active `GEMINI_API_KEY` (never inheriting unrelated API keys or conflicting `GEMINI_*` / `GOOGLE_API_KEY` variables).
- **Filesystem access**: Validates all caller-supplied local file paths against `agent.file_safety.raise_if_read_blocked` and regular-file checks before reading or passing them to `gemini-api`, and writes generated outputs only under `$HERMES_HOME/cache/` (`generated/images/`, `generated/audio/`, `videos/`, `transcripts/`). Cached media persists until cleaned up by Hermes or the user.
- **Session token accounting**: Reports CLI `usage` token counts (Interactions API `total_*_tokens` and porcelain `thoughts_tokens`) to Hermes's session accounting.

## Development & Testing

Run tests using Hermes's virtual environment and validate the plugin manifest with the Hermes CLI:

```bash
PYTHONPATH=/path/to/hermes-agent:. /path/to/hermes-agent/.venv/bin/python -m pytest tests/ -v
hermes plugins validate .
```

## Acknowledgments

Built upon the initial Gemini image generation provider implementation by Wesley Simplicio ([@wesleysimplicio](https://github.com/wesleysimplicio)) in [NousResearch/hermes-agent#97576](https://github.com/NousResearch/hermes-agent/pull/97576), review feedback from [@teknium1](https://github.com/teknium1) in [NousResearch/hermes-agent#120851](https://github.com/NousResearch/hermes-agent/pull/120851), and standalone plugin extraction work by [@semirkabir](https://github.com/semirkabir) in [NousResearch/hermes-agent#127068](https://github.com/NousResearch/hermes-agent/pull/127068).

## Disclaimer

This is not an officially supported Google product. This project is not eligible for the [Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security). Use of the Gemini API is subject to the [Gemini API Additional Terms of Service](https://ai.google.dev/gemini-api/terms).
