---
name: gemini-api-cli
description: "Headless usage patterns, flags, and JSON envelopes for google-gemini/gemini-api-cli when invoking gemini-api from Hermes."
---

# Gemini API CLI (`gemini-api`) Reference

Prefer the structured plugin tools (`image_generate`, `video_generate`, `gemini_music`, `gemini_tts`, `gemini_analyze`, `gemini_transcribe`, `gemini_deep_research`, `gemini_files_upload`) for standard media generation, multimodal analysis, transcription, and Deep Research tasks. Use the `gemini-api` CLI directly only when inspecting model catalogs, previewing request payloads with `--dry-run`, or calling advanced Interactions API, webhooks, or triggers subcommands.

## Mandatory Headless Rules

1. **Always disable interactive prompts and request JSON output**: pass `--no-interactive -o json` on every automated command so Cobra never opens a TUI picker on missing arguments.
2. **Disable SSE streaming when parsing JSON**: `generate`, `agent run`, and `agent status` stream SSE by default; add `--stream=false` whenever parsing a single JSON response object.
3. **Authenticate via `GEMINI_API_KEY`**: `gemini-api` reads `GEMINI_API_KEY` from the environment automatically; never pass API keys on the command line where process listings can expose them.
4. **Upload large local files first**: `analyze` and `transcribe` inline local files up to 14 MiB; for larger media (up to 2 GiB), run `gemini-api files upload <path> --wait -o json --no-interactive` and pass the returned `files/<id>` handle to `-i`.

## Exit Codes & Structured Errors

- `0` — Success (JSON object emitted on `stdout`).
- `1` — Runtime or upstream Gemini API error.
- `2` — CLI flag or argument validation error.
- `3` — Missing or invalid `GEMINI_API_KEY` authentication.

With `-o json` or `--agent-mode`, failures emit structured diagnostics on `stderr`:

```json
{"error": "missing required flag --id", "exit_code": 2, "hints": ["pass --id <interaction_id>"]}
```

## Common Ad-Hoc Commands

```bash
# List available Gemini models as JSON
gemini-api models list --no-interactive -o json

# Inspect exact CLI flags or request body JSON Schema without making an API call
gemini-api agent run --usage
gemini-api video --schema

# Preview the outgoing HTTP request without sending network traffic
gemini-api generate "Summarize quantum error correction" --stream=false --dry-run --no-interactive -o json

# Upload a large video or PDF once and reuse its 48-hour files/<id> handle
gemini-api files upload ./lecture.mp4 --wait --no-interactive -o json
gemini-api analyze -i files/abc123xyz "List key equations and timestamps" --no-interactive -o json

# Run and poll a background Deep Research interaction
gemini-api agent run "Compare solid-state battery electrolytes" --agent deep-research-preview-04-2026 --background --stream=false --no-interactive -o json
gemini-api agent status --id <interaction_id> --stream=false --no-interactive -o json
```
