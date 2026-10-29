# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hermes provider and tool registrations backed by ``google-gemini/gemini-api-cli``.

Registers:
- ``GeminiVideoGenProvider`` (``video_gen.provider = gemini``) via ``gemini-api video``
- Six structured tools in the ``gemini_api`` toolset:
  ``gemini_music``, ``gemini_tts``, ``gemini_analyze``, ``gemini_transcribe``,
  ``gemini_deep_research``, and ``gemini_files_upload``
- Bundled ``gemini-api-cli`` plugin skill (``SKILL.md``) for ad-hoc terminal workflows
"""

from __future__ import annotations

import base64
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent.video_gen_provider import (
    DEFAULT_ASPECT_RATIO as DEFAULT_VIDEO_ASPECT_RATIO,
    DEFAULT_RESOLUTION as DEFAULT_VIDEO_RESOLUTION,
    VideoGenProvider,
    error_response as video_error_response,
    success_response as video_success_response,
)
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)


def tool_error(message: str, **extra: Any) -> str:
    return json.dumps({"error": str(message)[:2048], **extra}, ensure_ascii=False)


def tool_result(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False)


def _load_cli_runner_module():
    pkg = __package__ or ""
    if pkg and pkg in sys.modules:
        parent = pkg.rpartition(".")[0]
        if not parent or parent in sys.modules:
            from . import cli_runner as _mod

            return _mod
    mod_name = f"{pkg}.cli_runner" if pkg else "gemini_plugin_cli_runner"
    existing = sys.modules.get(mod_name)
    if existing is not None:
        return existing
    runner_path = Path(__file__).resolve().parent / "cli_runner.py"
    spec = importlib.util.spec_from_file_location(mod_name, runner_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load cli_runner from {runner_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


_cli_runner = _load_cli_runner_module()
CLIResult = _cli_runner.CLIResult
CLI_INSTALL_HINT = _cli_runner.CLI_INSTALL_HINT
DEFAULT_CLI_TIMEOUT = _cli_runner.DEFAULT_CLI_TIMEOUT
DEFAULT_RESEARCH_TIMEOUT = _cli_runner.DEFAULT_RESEARCH_TIMEOUT
GeminiCLIRunner = _cli_runner.GeminiCLIRunner
_read_cached_config = _cli_runner._read_cached_config

# Documented CLI default models (used for Hermes picker metadata and accounting fallback when
# `-m` is omitted so `gemini-api` owns its built-in default models across CLI upgrades).
DEFAULT_VIDEO_MODEL = "gemini-omni-1.1-flash"
DEFAULT_MUSIC_MODEL = "lyria-3.5"
DEFAULT_TTS_MODEL = "gemini-3.8-flash-tts"
DEFAULT_ANALYZE_MODEL = "gemini-3.8-flash"
DEFAULT_RESEARCH_AGENT = "deep-research-preview-04-2026"

VIDEO_MODELS: Dict[str, Dict[str, Any]] = {
    "gemini-omni-1.1-flash": {
        "display": "Gemini Omni 1.1 Flash (Video)",
        "speed": "~30-120s",
        "strengths": "Conversational text-to-video and image-to-video via Gemini Interactions API",
        "price": "varies",
        "modalities": ["text", "image"],
    },
    "gemini-omni-flash-preview": {
        "display": "Gemini Omni Flash Preview (Video)",
        "speed": "~30-120s",
        "strengths": "Preview Gemini Omni video generation model via Interactions API",
        "price": "varies",
        "modalities": ["text", "image"],
    },
}

PROVIDED_TOOL_NAMES: Tuple[str, ...] = (
    "gemini_music",
    "gemini_tts",
    "gemini_analyze",
    "gemini_transcribe",
    "gemini_deep_research",
    "gemini_files_upload",
)

_TRANSCRIBE_FORMATS = {"md", "text", "json", "srt"}
_RESEARCH_ACTIONS = {"run", "status", "cancel"}


def _strip_google_prefix(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    return raw.split("/", 1)[1].strip() if raw.lower().startswith("google/") else raw


def _load_video_gen_gemini_config() -> Dict[str, Any]:
    cfg = _read_cached_config()
    video_cfg = cfg.get("video_gen") if isinstance(cfg, dict) else None
    gemini_cfg = video_cfg.get("gemini") if isinstance(video_cfg, dict) else None
    return dict(gemini_cfg) if isinstance(gemini_cfg, dict) else {}


def _resolve_explicit_video_model(caller_model: Optional[str] = None) -> Optional[str]:
    """Return an explicit Gemini video model override, or ``None`` to let ``gemini-api`` use its default.

    Foreign top-level ``video_gen.model`` IDs (e.g. ``grok-imagine-video``) are ignored so switching
    providers never sends a non-Gemini model ID to Google AI Studio.
    """
    cleaned = _strip_google_prefix(caller_model)
    if cleaned and (cleaned in VIDEO_MODELS or cleaned.startswith(("gemini-", "veo-"))):
        return cleaned
    cfg_model = _strip_google_prefix(_load_video_gen_gemini_config().get("model"))
    return cfg_model or None


def _cache_subdir(*parts: str) -> Path:
    target = get_hermes_home().joinpath("cache", *parts)
    target.mkdir(parents=True, exist_ok=True)
    return target


def _cli_failure_to_tool_error(res: CLIResult) -> str:
    extra: Dict[str, Any] = {"error_type": res.error_type or "api_error", "exit_code": res.exit_code}
    if res.hints:
        extra["hints"] = res.hints
    return tool_error(res.error or "gemini-api command failed", **extra)


def _extract_interaction_text(payload: Any) -> str:
    """Pull readable assistant output text from an Interactions API response dict."""
    if not isinstance(payload, dict):
        return ""
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    chunks: List[str] = []
    for collection_key in ("outputs", "steps"):
        items = payload.get(collection_key)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("text"), str) and item["text"].strip():
                chunks.append(item["text"].strip())
            content = item.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") != "thought":
                        txt = block.get("text")
                        if isinstance(txt, str) and txt.strip():
                            chunks.append(txt.strip())
    return "\n\n".join(chunks)


class GeminiVideoGenProvider(VideoGenProvider):
    """Hermes ``VideoGenProvider`` backed by ``gemini-api video``."""

    name = "gemini"
    display_name = "Google AI Studio"

    def __init__(
        self,
        *,
        runner: GeminiCLIRunner,
        endpoint_resolver: Callable[[], Tuple[str, str]],
        image_loader: Callable[[str], Tuple[bytes, str]],
    ) -> None:
        self._runner = runner
        self._resolve_endpoint = endpoint_resolver
        self._load_image_bytes = image_loader

    def is_available(self) -> bool:
        return self._runner.is_available()

    def list_models(self) -> List[Dict[str, Any]]:
        return [{"id": mid, **meta} for mid, meta in VIDEO_MODELS.items()]

    def default_model(self) -> Optional[str]:
        return DEFAULT_VIDEO_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Google AI Studio (gemini-api CLI)",
            "badge": "paid",
            "tag": "Conversational Gemini video generation via gemini-api CLI and GEMINI_API_KEY",
            "env_vars": [
                {
                    "key": "GEMINI_API_KEY",
                    "prompt": "Google AI Studio Gemini API key",
                    "url": "https://aistudio.google.com/apikey",
                },
            ],
        }

    def capabilities(self) -> Dict[str, Any]:
        return {
            "modalities": ["text", "image"],
            "aspect_ratios": ["16:9", "9:16", "1:1", "4:3", "3:4"],
            "resolutions": ["720p", "1080p"],
            "max_duration": 10,
            "min_duration": 4,
            "supports_audio": True,
            "supports_negative_prompt": False,
            "supports_seed": False,
            "supports_upscale": False,
            "max_reference_images": 0,
        }

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        duration: Optional[int] = None,
        aspect_ratio: str = DEFAULT_VIDEO_ASPECT_RATIO,
        resolution: str = DEFAULT_VIDEO_RESOLUTION,
        negative_prompt: Optional[str] = None,
        audio: Optional[bool] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del reference_image_urls, resolution, negative_prompt, audio, seed, kwargs
        cleaned_prompt = (prompt or "").strip()
        explicit_model = _resolve_explicit_video_model(model)
        accounting_model = explicit_model or DEFAULT_VIDEO_MODEL
        if not cleaned_prompt:
            return video_error_response(
                error="A non-empty video prompt is required.",
                error_type="invalid_argument",
                provider="gemini",
                model=accounting_model,
                prompt="",
                aspect_ratio=aspect_ratio,
            )

        _, api_key = self._resolve_endpoint()
        if not api_key:
            return video_error_response(
                error="GEMINI_API_KEY is not set. Configure GEMINI_API_KEY in ~/.hermes/.env.",
                error_type="auth_required",
                provider="gemini",
                model=accounting_model,
                prompt=cleaned_prompt,
                aspect_ratio=aspect_ratio,
            )

        out_dir = _cache_subdir("videos")
        modality = "image" if (image_url and image_url.strip()) else "text"
        cli_args: List[str] = ["video", "--out", str(out_dir)]
        if explicit_model:
            cli_args.extend(["-m", explicit_model])

        positional_args: List[str] = []
        if modality == "image" and image_url:
            ref = image_url.strip()
            if ref.startswith("files/"):
                valid_ref, ref_err, _ = self._runner.prepare_media_input(ref, allow_youtube=False)
                if ref_err is not None or not valid_ref:
                    return video_error_response(
                        error=ref_err.error if ref_err is not None else "Invalid files/<id> handle.",
                        error_type="invalid_argument",
                        provider="gemini",
                        model=accounting_model,
                        prompt=cleaned_prompt,
                        aspect_ratio=aspect_ratio,
                    )
                image_part: Dict[str, Any] = {"type": "image", "uri": valid_ref}
            else:
                try:
                    raw_bytes, mime = self._load_image_bytes(ref)
                except Exception as exc:  # noqa: BLE001
                    return video_error_response(
                        error=f"Could not load image_url for video generation: {exc}",
                        error_type="invalid_argument",
                        provider="gemini",
                        model=accounting_model,
                        prompt=cleaned_prompt,
                        aspect_ratio=aspect_ratio,
                    )
                image_part = {
                    "type": "image",
                    "mime_type": mime,
                    "data": base64.b64encode(raw_bytes).decode("ascii"),
                }
            body = json.dumps({"input": [image_part, {"type": "text", "text": cleaned_prompt}]})
            cli_args.extend(["--body", body])
        else:
            positional_args.append(cleaned_prompt)

        res = self._runner.run(
            cli_args,
            positional_args=positional_args,
            timeout=DEFAULT_RESEARCH_TIMEOUT,
            default_model=accounting_model,
        )
        if not res.ok:
            return video_error_response(
                error=res.error,
                error_type=res.error_type or "provider_error",
                provider="gemini",
                model=accounting_model,
                prompt=cleaned_prompt,
                aspect_ratio=aspect_ratio,
            )

        payload = res.data if isinstance(res.data, dict) else {}
        saved_path = str(payload.get("path") or "").strip()
        if not saved_path or not Path(saved_path).is_file():
            return video_error_response(
                error="gemini-api video completed without returning a valid output video file path.",
                error_type="invalid_response",
                provider="gemini",
                model=accounting_model,
                prompt=cleaned_prompt,
                aspect_ratio=aspect_ratio,
            )

        reported_model = str(payload.get("model") or explicit_model or DEFAULT_VIDEO_MODEL).strip()
        extra: Dict[str, Any] = {"api": "google-ai-studio"}
        if payload.get("response_id"):
            extra["response_id"] = payload["response_id"]
        if payload.get("mime_type"):
            extra["mime_type"] = payload["mime_type"]
        return video_success_response(
            video=saved_path,
            model=reported_model,
            prompt=cleaned_prompt,
            modality=modality,
            aspect_ratio=aspect_ratio,
            duration=int(duration) if duration else 0,
            provider="gemini",
            extra=extra,
        )


# ── Tool JSON Schemas ────────────────────────────────────────────────────────

GEMINI_MUSIC_SCHEMA: Dict[str, Any] = {
    "name": "gemini_music",
    "description": (
        "Generate instrumental or vocal music tracks using Google AI Studio's Lyria model "
        "via gemini-api CLI and save the audio file to the Hermes cache."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Musical prompt describing genre, instruments, tempo, mood, or vocals.",
            },
            "model": {
                "type": "string",
                "description": "Optional music model override (when omitted, gemini-api uses its built-in default).",
            },
        },
        "required": ["prompt"],
    },
}

GEMINI_TTS_SCHEMA: Dict[str, Any] = {
    "name": "gemini_tts",
    "description": (
        "Synthesize 24kHz WAV speech from text or a two-speaker script via gemini-api CLI. "
        "Supports single-speaker voices (e.g. Kore, Puck, Charon, Aoede) and "
        "two-speaker dialogues (e.g. 'Host=Kore,Guest=Puck')."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text to speak. For multi-speaker mode, prefix lines with speaker names (e.g. 'Host: Hello!').",
            },
            "voice": {
                "type": "string",
                "description": "Prebuilt voice name for single-speaker synthesis (default: Kore).",
            },
            "multi_speaker": {
                "type": "string",
                "description": "Optional two-speaker voice map 'Speaker1=Voice1,Speaker2=Voice2' (mutually exclusive with voice).",
            },
            "language": {
                "type": "string",
                "description": "Optional BCP-47 language hint (e.g. en-US, ja-JP, es-US).",
            },
            "model": {
                "type": "string",
                "description": "Optional TTS model override (when omitted, gemini-api uses its built-in default).",
            },
        },
        "required": ["text"],
    },
}

GEMINI_ANALYZE_SCHEMA: Dict[str, Any] = {
    "name": "gemini_analyze",
    "description": (
        "Ask questions about local documents, PDFs, videos, audio files, images, CSVs, 'files/<id>' "
        "handles, or YouTube URLs via gemini-api CLI. Local media over ~14MB (up to 2GB) is seamlessly "
        "uploaded via the Gemini Files API, and the resulting 'files/<id>' handle is returned in 'inputs' "
        "so follow-up questions can reuse it without re-uploading."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "inputs": {
                "type": "array",
                "items": {"type": "string"},
                "description": "One or more local file paths, 'files/<id>' handles, or YouTube URLs to analyze.",
            },
            "question": {
                "type": "string",
                "description": "Question or instruction for analyzing the supplied media (defaults to a summary).",
            },
            "system": {
                "type": "string",
                "description": "Optional system instruction to guide tone or output structure.",
            },
            "mime_type": {
                "type": "string",
                "description": "Optional MIME type override when passing a single input with an uncommon extension.",
            },
            "model": {
                "type": "string",
                "description": "Optional Gemini model override (when omitted, gemini-api uses its built-in default).",
            },
        },
        "required": ["inputs"],
    },
}

GEMINI_TRANSCRIBE_SCHEMA: Dict[str, Any] = {
    "name": "gemini_transcribe",
    "description": (
        "Transcribe local audio or video files (or 'files/<id>' handles) with speaker diarization "
        "and timestamps into Markdown, plain text, JSON, or SRT captions via gemini-api CLI. "
        "Local media over ~14MB (up to 2GB) is seamlessly uploaded through the Gemini Files API."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "inputs": {
                "type": "array",
                "items": {"type": "string"},
                "description": "One or more local audio/video paths or 'files/<id>' handles to transcribe.",
            },
            "format": {
                "type": "string",
                "enum": ["md", "text", "json", "srt"],
                "description": "Transcript output format: 'md' (default), 'text', 'json', or 'srt' subtitles.",
            },
            "speakers": {
                "type": "boolean",
                "description": "Include speaker attribution labels (default: true; ignored for 'srt').",
            },
            "timestamps": {
                "type": "boolean",
                "description": "Include timestamps (default: true; always enabled for 'srt').",
            },
            "model": {
                "type": "string",
                "description": "Optional Gemini model override (when omitted, gemini-api uses its built-in default).",
            },
        },
        "required": ["inputs"],
    },
}

GEMINI_DEEP_RESEARCH_SCHEMA: Dict[str, Any] = {
    "name": "gemini_deep_research",
    "description": (
        "Run, poll, or cancel multi-step Gemini Deep Research and Interactions API agent tasks "
        f"(default agent: {DEFAULT_RESEARCH_AGENT}) via gemini-api CLI. Defaults to asynchronous "
        "background execution so long runs return an interaction_id immediately for status polling."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["run", "status", "cancel"],
                "description": "'run' starts a research task (default), 'status' fetches an interaction by id, 'cancel' stops a background run.",
            },
            "prompt": {
                "type": "string",
                "description": "Research prompt or question (required when action='run').",
            },
            "interaction_id": {
                "type": "string",
                "description": "Interaction ID returned by a prior background run (required for 'status' and 'cancel').",
            },
            "background": {
                "type": "boolean",
                "description": "Run asynchronously and return interaction_id immediately (default: true; set false to block).",
            },
            "agent": {
                "type": "string",
                "description": f"Agent ID for action='run' (default: {DEFAULT_RESEARCH_AGENT}).",
            },
        },
        "required": [],
    },
}

GEMINI_FILES_UPLOAD_SCHEMA: Dict[str, Any] = {
    "name": "gemini_files_upload",
    "description": (
        "Upload a local media or document file (up to 2GB) to the Gemini Files API via gemini-api CLI "
        "and wait until ACTIVE, returning a reusable 48-hour 'files/<id>' handle."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Local path to the file to upload.",
            },
            "display_name": {
                "type": "string",
                "description": "Optional human-readable display name for the uploaded file.",
            },
            "mime_type": {
                "type": "string",
                "description": "Optional explicit MIME type override (e.g. 'video/mp4', 'application/pdf').",
            },
        },
        "required": ["path"],
    },
}


# ── Tool Handler Builders ────────────────────────────────────────────────────

def _string_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    items = raw if isinstance(raw, list) else [raw]
    return [str(x).strip() for x in items if isinstance(x, (str, int, float)) and str(x).strip()]


def build_tool_handlers(runner: GeminiCLIRunner) -> Dict[str, Callable[..., str]]:
    """Create ``(args, **kw) -> json_str`` handlers bound to *runner*."""

    def handle_music(args: dict, **_kw: Any) -> str:
        prompt = str(args.get("prompt") or "").strip()
        if not prompt:
            return tool_error("A non-empty 'prompt' is required for gemini_music.", error_type="invalid_argument")
        explicit_model = _strip_google_prefix(args.get("model"))
        out_dir = _cache_subdir("generated", "audio")
        cmd_args: List[str] = ["music", "--out", str(out_dir)]
        if explicit_model:
            cmd_args.extend(["-m", explicit_model])
        res = runner.run(
            cmd_args,
            positional_args=[prompt],
            timeout=300,
            default_model=explicit_model or DEFAULT_MUSIC_MODEL,
        )
        if not res.ok:
            return _cli_failure_to_tool_error(res)
        payload = res.data if isinstance(res.data, dict) else {}
        return tool_result({
            "success": True,
            "audio": payload.get("path"),
            "mime_type": payload.get("mime_type"),
            "size_bytes": payload.get("size_bytes"),
            "response_id": payload.get("response_id"),
            "model": payload.get("model") or explicit_model or DEFAULT_MUSIC_MODEL,
            "provider": "gemini",
        })

    def handle_tts(args: dict, **_kw: Any) -> str:
        text = str(args.get("text") or "").strip()
        if not text:
            return tool_error("A non-empty 'text' is required for gemini_tts.", error_type="invalid_argument")
        multi_speaker = str(args.get("multi_speaker") or "").strip()
        voice = str(args.get("voice") or "").strip()
        if multi_speaker and voice:
            return tool_error(
                "Pass either 'voice' (single-speaker) or 'multi_speaker' (two-speaker), not both.",
                error_type="invalid_argument",
            )
        explicit_model = _strip_google_prefix(args.get("model"))
        out_dir = _cache_subdir("generated", "audio")
        cmd_args: List[str] = ["tts", "--out", str(out_dir)]
        if explicit_model:
            cmd_args.extend(["-m", explicit_model])
        if multi_speaker:
            cmd_args.extend(["--multi-speaker", multi_speaker])
        elif voice:
            cmd_args.extend(["--voice", voice])
        language = str(args.get("language") or "").strip()
        if language:
            cmd_args.extend(["--language", language])
        res = runner.run(
            cmd_args,
            positional_args=[text],
            timeout=DEFAULT_CLI_TIMEOUT,
            default_model=explicit_model or DEFAULT_TTS_MODEL,
        )
        if not res.ok:
            return _cli_failure_to_tool_error(res)
        payload = res.data if isinstance(res.data, dict) else {}
        return tool_result({
            "success": True,
            "audio": payload.get("path"),
            "mime_type": payload.get("mime_type", "audio/wav"),
            "size_bytes": payload.get("size_bytes"),
            "sample_rate": payload.get("sample_rate"),
            "channels": payload.get("channels"),
            "model": payload.get("model") or explicit_model or DEFAULT_TTS_MODEL,
            "provider": "gemini",
        })

    def handle_analyze(args: dict, **_kw: Any) -> str:
        raw_inputs = _string_list(args.get("inputs"))
        if not raw_inputs:
            return tool_error("At least one entry in 'inputs' is required for gemini_analyze.", error_type="invalid_argument")
        mime_type = str(args.get("mime_type") or "").strip()
        if mime_type and len(raw_inputs) > 1:
            return tool_error("'mime_type' can only be specified when a single input is provided.", error_type="invalid_argument")

        prepared: List[str] = []
        inline_used = 0
        for item in raw_inputs:
            resolved_ref, err, added_cost = runner.prepare_media_input(
                item,
                allow_youtube=True,
                mime_type=mime_type or None,
                inline_budget_used=inline_used,
                enforce_cumulative=True,
            )
            if err is not None or not resolved_ref:
                return _cli_failure_to_tool_error(err) if err is not None else tool_error("Invalid input reference.")
            prepared.append(resolved_ref)
            inline_used += added_cost

        explicit_model = _strip_google_prefix(args.get("model"))
        cmd_args: List[str] = ["analyze"]
        if explicit_model:
            cmd_args.extend(["-m", explicit_model])
        for ref in prepared:
            cmd_args.extend(["-i", ref])
        if mime_type:
            cmd_args.extend(["--mime-type", mime_type])
        system = str(args.get("system") or "").strip()
        if system:
            cmd_args.extend(["--system", system])
        question = str(args.get("question") or "").strip()
        positional = [question] if question else []

        res = runner.run(
            cmd_args,
            positional_args=positional,
            timeout=300,
            default_model=explicit_model or DEFAULT_ANALYZE_MODEL,
        )
        if not res.ok:
            return _cli_failure_to_tool_error(res)
        payload = res.data if isinstance(res.data, dict) else {}
        return tool_result({
            "success": True,
            "text": payload.get("text", ""),
            "model": payload.get("model") or explicit_model or DEFAULT_ANALYZE_MODEL,
            "inputs": prepared,
            "mime_types": payload.get("mime_types"),
            "provider": "gemini",
        })

    def handle_transcribe(args: dict, **_kw: Any) -> str:
        raw_inputs = _string_list(args.get("inputs"))
        if not raw_inputs:
            return tool_error("At least one entry in 'inputs' is required for gemini_transcribe.", error_type="invalid_argument")
        fmt = str(args.get("format") or "md").strip().lower()
        if fmt not in _TRANSCRIBE_FORMATS:
            return tool_error(
                f"Unsupported format {fmt!r}: expected one of 'md', 'text', 'json', or 'srt'.",
                error_type="invalid_argument",
            )

        prepared: List[str] = []
        for item in raw_inputs:
            resolved_ref, err, _ = runner.prepare_media_input(item, allow_youtube=False)
            if err is not None or not resolved_ref:
                return _cli_failure_to_tool_error(err) if err is not None else tool_error("Invalid input reference.")
            prepared.append(resolved_ref)

        explicit_model = _strip_google_prefix(args.get("model"))
        out_dir = _cache_subdir("transcripts")
        cmd_args: List[str] = ["transcribe", "--format", fmt, "--out", str(out_dir)]
        if explicit_model:
            cmd_args.extend(["-m", explicit_model])
        for ref in prepared:
            cmd_args.extend(["-i", ref])
        if args.get("speakers") is False and fmt != "srt":
            cmd_args.append("--no-speakers")
        if args.get("timestamps") is False and fmt != "srt":
            cmd_args.append("--no-timestamps")

        res = runner.run(
            cmd_args,
            timeout=300,
            default_model=explicit_model or DEFAULT_ANALYZE_MODEL,
        )
        if not res.ok:
            return _cli_failure_to_tool_error(res)
        payload = res.data if isinstance(res.data, dict) else {}
        results = payload.get("results") if isinstance(payload.get("results"), list) else []
        transcripts: List[Dict[str, Any]] = []
        for entry in results:
            if not isinstance(entry, dict):
                continue
            item_out = dict(entry)
            out_path = entry.get("path")
            if isinstance(out_path, str) and Path(out_path).is_file():
                try:
                    text_body = Path(out_path).read_text(encoding="utf-8", errors="replace")
                    if len(text_body) <= 32768:
                        item_out["content"] = text_body
                except OSError:
                    pass
            transcripts.append(item_out)
        return tool_result({
            "success": True,
            "format": payload.get("format") or fmt,
            "model": payload.get("model") or explicit_model or DEFAULT_ANALYZE_MODEL,
            "results": transcripts,
            "provider": "gemini",
        })

    def handle_deep_research(args: dict, **_kw: Any) -> str:
        action = str(args.get("action") or "run").strip().lower()
        if action not in _RESEARCH_ACTIONS:
            return tool_error(
                f"Unsupported action {action!r}: expected 'run', 'status', or 'cancel'.",
                error_type="invalid_argument",
            )
        if action == "run":
            prompt = str(args.get("prompt") or "").strip()
            if not prompt:
                return tool_error("A non-empty 'prompt' is required when action='run'.", error_type="invalid_argument")
            agent_id = str(args.get("agent") or "").strip() or DEFAULT_RESEARCH_AGENT
            raw_bg = args.get("background")
            background = True if raw_bg is None else bool(raw_bg)
            cmd_args = ["agent", "run", "--agent", agent_id, "--stream=false"]
            if background:
                cmd_args.append("--background")
            timeout = 60 if background else DEFAULT_RESEARCH_TIMEOUT
            res = runner.run(
                cmd_args,
                positional_args=[prompt],
                timeout=timeout,
                default_model=agent_id,
            )
            if not res.ok:
                return _cli_failure_to_tool_error(res)
            payload = res.data if isinstance(res.data, dict) else {}
            return tool_result({
                "success": True,
                "action": "run",
                "interaction_id": payload.get("id"),
                "status": payload.get("status"),
                "agent": agent_id,
                "background": background,
                "output_text": _extract_interaction_text(payload),
                "provider": "gemini",
            })

        interaction_id = str(args.get("interaction_id") or "").strip()
        if not interaction_id:
            return tool_error(
                f"'interaction_id' is required when action={action!r}.",
                error_type="invalid_argument",
            )
        if interaction_id.startswith("-"):
            return tool_error(
                f"Invalid 'interaction_id' {interaction_id!r}.",
                error_type="invalid_argument",
            )
        if action == "status":
            res = runner.run(["agent", "status", "--id", interaction_id, "--stream=false"], timeout=60)
            if not res.ok:
                return _cli_failure_to_tool_error(res)
            payload = res.data if isinstance(res.data, dict) else {}
            return tool_result({
                "success": True,
                "action": "status",
                "interaction_id": payload.get("id") or interaction_id,
                "status": payload.get("status"),
                "output_text": _extract_interaction_text(payload),
                "provider": "gemini",
            })

        res = runner.run(["agent", "cancel", "--id", interaction_id], timeout=60)
        if not res.ok:
            return _cli_failure_to_tool_error(res)
        payload = res.data if isinstance(res.data, dict) else {}
        return tool_result({
            "success": True,
            "action": "cancel",
            "interaction_id": payload.get("id") or interaction_id,
            "status": payload.get("status") or "cancelled",
            "provider": "gemini",
        })

    def handle_files_upload(args: dict, **_kw: Any) -> str:
        path = str(args.get("path") or "").strip()
        if not path:
            return tool_error("A non-empty 'path' is required for gemini_files_upload.", error_type="invalid_argument")
        res = runner.upload_file(
            path,
            display_name=args.get("display_name"),
            mime_type=args.get("mime_type"),
            wait=True,
        )
        if not res.ok:
            return _cli_failure_to_tool_error(res)
        payload = res.data if isinstance(res.data, dict) else {}
        return tool_result({
            "success": True,
            "name": payload.get("name"),
            "uri": payload.get("uri"),
            "mime_type": payload.get("mime_type"),
            "display_name": payload.get("display_name"),
            "size_bytes": payload.get("size_bytes"),
            "state": payload.get("state"),
            "provider": "gemini",
        })

    return {
        "gemini_music": handle_music,
        "gemini_tts": handle_tts,
        "gemini_analyze": handle_analyze,
        "gemini_transcribe": handle_transcribe,
        "gemini_deep_research": handle_deep_research,
        "gemini_files_upload": handle_files_upload,
    }


def register_cli_integrations(
    ctx: Any,
    *,
    endpoint_resolver: Callable[[], Tuple[str, str]],
    image_loader: Callable[[str], Tuple[bytes, str]],
    user_agent: str,
) -> GeminiCLIRunner:
    """Register ``GeminiVideoGenProvider``, structured CLI tools, and ``SKILL.md`` on *ctx*."""
    runner = GeminiCLIRunner(endpoint_resolver=endpoint_resolver, user_agent=user_agent)

    if hasattr(ctx, "register_video_gen_provider"):
        ctx.register_video_gen_provider(
            GeminiVideoGenProvider(
                runner=runner,
                endpoint_resolver=endpoint_resolver,
                image_loader=image_loader,
            )
        )

    handlers = build_tool_handlers(runner)
    tool_specs = (
        ("gemini_music", GEMINI_MUSIC_SCHEMA, "🎵"),
        ("gemini_tts", GEMINI_TTS_SCHEMA, "🗣️"),
        ("gemini_analyze", GEMINI_ANALYZE_SCHEMA, "🔎"),
        ("gemini_transcribe", GEMINI_TRANSCRIBE_SCHEMA, "📝"),
        ("gemini_deep_research", GEMINI_DEEP_RESEARCH_SCHEMA, "🧠"),
        ("gemini_files_upload", GEMINI_FILES_UPLOAD_SCHEMA, "📤"),
    )
    for tool_name, schema, emoji in tool_specs:
        ctx.register_tool(
            name=tool_name,
            toolset="gemini_api",
            schema=schema,
            handler=handlers[tool_name],
            check_fn=runner.is_available,
            requires_env=["GEMINI_API_KEY"],
            description=schema["description"],
            emoji=emoji,
        )

    skill_path = Path(__file__).resolve().parent / "SKILL.md"
    if hasattr(ctx, "register_skill") and skill_path.is_file():
        ctx.register_skill(
            "gemini-api-cli",
            skill_path,
            description="Non-interactive command reference for google-gemini/gemini-api-cli",
        )
    return runner
