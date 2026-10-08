# hermes-gemini-api-plugin

Native Google AI Studio Gemini (**Nano Banana**) image generation and editing backend plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent).

Calls the Gemini REST endpoint (`POST /v1beta/models/{model}:generateContent` with `responseModalities: ["TEXT", "IMAGE"]`), decodes returned `inlineData` images into `$HERMES_HOME/cache/generated/images/`, and records token usage (including Nano Banana 2.1 and Nano Banana Pro thinking tokens) into Hermes session accounting.

## Supported Models

| Model ID | Display Name | Speed | Capabilities |
| :--- | :--- | :--- | :--- |
| `gemini-nano-banana-2.1` *(default)* | **Nano Banana 2.1** (Gemini Nano Banana 2.1) | Fast | 14 aspect ratios (including `1:4`, `4:1`, `1:8`, `8:1`), `1K` / `2K` / `4K` output resolution, Google Web and Image Search grounding, up to 14 reference images |
| `gemini-3.1-flash-image` | **Nano Banana 2** (Gemini 3.1 Flash Image) | Fast | 14 aspect ratios (including `1:4`, `4:1`, `1:8`, `8:1`), `512` / `1K` / `2K` / `4K` output resolution, Google Web and Image Search grounding, up to 14 reference images |
| `gemini-3.1-flash-lite-image` | **Nano Banana 2 Lite** (Gemini 3.1 Flash Lite Image) | Fastest | Lowest latency & cost; 10 aspect ratios, `1K` output resolution, up to 14 reference images |
| `gemini-3-pro-image` | **Nano Banana Pro** (Gemini 3 Pro Image) | Slower | Highest fidelity & reasoning; 10 aspect ratios, `1K` / `2K` / `4K` output resolution, Google Search grounding, up to 14 reference images |

## Installation

Install from the Hermes Plugin Catalog:

```bash
hermes plugins install gemini-image
```

Or install directly from GitHub:

```bash
hermes plugins install https://github.com/google-gemini/hermes-gemini-api-plugin
```

## Configuration

### 1. API Key (`.env`)

Set your Google AI Studio API key ([get one here](https://aistudio.google.com/apikey)) in `~/.hermes/.env` or your shell environment:

```bash
GEMINI_API_KEY="your-api-key"
```

### 2. Provider & Generation Settings (`~/.hermes/config.yaml`)

Select `gemini` as your image generation provider (or run `hermes tools` → **Image Generation** → **Google AI Studio (direct)**):

```yaml
image_gen:
  provider: gemini
  gemini:
    model: gemini-nano-banana-2.1   # or gemini-3.1-flash-image, gemini-3.1-flash-lite-image, gemini-3-pro-image
    image_size: 2K                  # 1K | 2K | 4K (or 512 on gemini-3.1-flash-image)
    aspect_ratio: "16:9"            # exact ratio override (e.g. 1:1, 16:9, 9:16, 21:9, 1:4, 4:1, 1:8, 8:1)
    google_search: false            # enable Google Search grounding on supported models
    # Optional custom endpoint / credential routing:
    # base_url: https://generativelanguage.googleapis.com/v1beta
    # key_env: GEMINI_API_KEY
    # provider: my-custom-gemini-endpoint
```

Optional environment variable overrides for model and endpoint:

- `GEMINI_IMAGE_MODEL` — Override the Gemini image model ID (strips any optional `google/` prefix).
- `GEMINI_BASE_URL` — Override the Gemini API base URL.

## Features

- **Multi-reference image editing**: Pass `image_url` and/or `reference_image_urls` (up to 14 reference images across HTTP(S) URLs, `data:` URIs, or local file paths) to edit, restyle, or combine images.
- **Safety & size guards**: Uses Hermes's SSRF-safe HTTP client for remote URLs, blocks local credential store reads via `raise_if_read_blocked`, sniffs magic bytes (`PNG`, `JPEG`, `WebP`, `GIF`) before upload, and enforces a 25 MB per-image cap plus an 80 MB total inline payload ceiling.
- **Session token accounting**: Reports `promptTokenCount`, `candidatesTokenCount`, `thoughtsTokenCount`, and `cachedContentTokenCount` to Hermes's `record_token_usage`.

## Security & Privacy Disclosure

- **External network calls**: Sends text prompts and optional inline reference images over HTTPS to `https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent` (or a custom `image_gen.gemini.base_url`) authenticated via `GEMINI_API_KEY`. Remote `http(s)://` reference image URLs passed to `image_generate` are fetched through Hermes's SSRF-safe HTTP client.
- **Filesystem access**: Reads local reference image paths explicitly passed by the caller (guarded by `agent.file_safety.raise_if_read_blocked`) and writes generated output images to `$HERMES_HOME/cache/generated/images/`.
- **No background processes or telemetry**: Spawns no shell commands or background workers and collects no telemetry.

## Development & Testing

```bash
PYTHONPATH=/path/to/hermes-agent:. pytest tests/ -v
PYTHONPATH=/path/to/hermes-agent python -m hermes_cli.main plugins validate .
```

## Acknowledgments

Built upon the initial Gemini image generation provider implementation by Wesley Simplicio ([@wesleysimplicio](https://github.com/wesleysimplicio)) in [NousResearch/hermes-agent#97576](https://github.com/NousResearch/hermes-agent/pull/97576), review feedback from [@teknium1](https://github.com/teknium1) in [NousResearch/hermes-agent#120851](https://github.com/NousResearch/hermes-agent/pull/120851), and standalone plugin extraction work by [@semirkabir](https://github.com/semirkabir) in [NousResearch/hermes-agent#127068](https://github.com/NousResearch/hermes-agent/pull/127068).

## Disclaimer

This is not an officially supported Google product. This project is not eligible for the [Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security). Use of the Gemini API is subject to the [Gemini API Additional Terms of Service](https://ai.google.dev/gemini-api/terms).
