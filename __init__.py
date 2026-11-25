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

"""Google AI Studio Gemini plugin for Hermes, backed end-to-end by ``google-gemini/gemini-api-cli``.

Provides:
- ``GeminiImageGenProvider`` (``image_gen.provider = gemini``) via ``gemini-api image`` (Interactions API)
- ``GeminiVideoGenProvider`` (``video_gen.provider = gemini``) via ``gemini-api video`` (Interactions API)
- Six structured tools in the ``gemini_api`` toolset (``gemini_music``, ``gemini_tts``,
  ``gemini_analyze``, ``gemini_transcribe``, ``gemini_deep_research``, ``gemini_files_upload``)
- Bundled ``gemini-api-cli`` skill (``SKILL.md``)

Image model selection: caller ``model`` (honoured only when it names a Gemini catalog entry) →
``GEMINI_IMAGE_MODEL`` → ``image_gen.gemini.model`` → ``gemini-api image``'s built-in default
(:data:`DEFAULT_MODEL`). Top-level ``image_gen.model`` is deliberately NOT honoured because it is
provider-agnostic and may hold another backend's ID.
"""

from __future__ import annotations

import base64
import importlib
import importlib.metadata
import importlib.util
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import hermes_constants
from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    normalize_reference_images,
    resolve_aspect_ratio,
    success_response,
)
from agent.secret_scope import get_secret
from hermes_cli.config import get_compatible_custom_providers, load_config, load_config_readonly
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = "gemini-nano-banana-2.1"
PLUGIN_VERSION = "0.2.0"
_FALLBACK_CLI_VERSION = "0.5.0"
_PROMPT_REQUIRED = "Prompt is required and must be a non-empty string"


def _load_sibling_module(module_leaf: str):
    """Import a sibling plugin module across package, loader, and standalone test contexts."""
    pkg = __package__ or __name__
    parent = pkg.rpartition(".")[0]
    if pkg in sys.modules and (not parent or parent in sys.modules):
        try:
            return importlib.import_module(f".{module_leaf}", package=pkg)
        except ImportError:
            pass
    mod_name = f"{pkg}.{module_leaf}"
    existing = sys.modules.get(mod_name)
    if existing is not None:
        return existing
    mod_path = Path(__file__).resolve().parent / f"{module_leaf}.py"
    if not mod_path.is_file():
        return None
    spec = importlib.util.spec_from_file_location(mod_name, mod_path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _resolve_hermes_base_version() -> str:
    """Resolve the running Hermes base version using only public package metadata and install stamps."""
    try:
        hermes_root = Path(hermes_constants.__file__).resolve().parent
        stamp_path = hermes_root / "install-stamp.json"
        if stamp_path.is_file():
            stamp = json.loads(stamp_path.read_text(encoding="utf-8-sig"))
            if isinstance(stamp, dict):
                for key in ("baseVersion", "base_version", "version"):
                    val = str(stamp.get(key) or "").strip()
                    if val and val.lower() != "unknown":
                        return val
    except Exception:  # noqa: BLE001
        pass
    try:
        dist_ver = importlib.metadata.version("hermes-agent").strip()
        if dist_ver and dist_ver != "0.0.0":
            return dist_ver
    except Exception:  # noqa: BLE001
        pass
    return "0.0.0"


def _pinned_cli_version() -> str:
    runner_mod = _load_sibling_module("cli_runner")
    return str(getattr(runner_mod, "PINNED_CLI_VERSION", _FALLBACK_CLI_VERSION))


HERMES_VERSION = f"hermes-agent/{_resolve_hermes_base_version()}"
USER_AGENT = f"{HERMES_VERSION} (gemini-api-plugin/{PLUGIN_VERSION}; gemini-api-cli/{_pinned_cli_version()})"
CLI_USER_AGENT = USER_AGENT

_ASPECT_RATIOS = {"landscape": "16:9", "square": "1:1", "portrait": "9:16"}
_GEMINI_RATIOS_10 = (
    "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9",
)
_GEMINI_RATIOS_14 = _GEMINI_RATIOS_10 + ("1:4", "4:1", "1:8", "8:1")
_MAX_REFERENCE_IMAGES = 14
_MAX_INPUT_IMAGE_BYTES = 25 * 1024 * 1024  # 25 MB per reference image

_MIME_TO_SUFFIX = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}

MODELS: Dict[str, Dict[str, Any]] = {
    "gemini-nano-banana-2.1": {
        "display": "Nano Banana 2.1 (Gemini Nano Banana 2.1)",
        "speed": "Fast",
        "strengths": "Best balance — 14 aspect ratios, 1K/2K/4K resolution, Google Web and Image Search grounding, up to 14 refs",
        "api_model": "gemini-nano-banana-2.1",
        "aspect_ratios": _GEMINI_RATIOS_14,
        "resolutions": ("1K", "2K", "4K"),
        "supports_search": True,
        "max_refs": 14,
    },
    "gemini-3.1-flash-image": {
        "display": "Nano Banana 2 (Gemini 3.1 Flash Image)",
        "speed": "Fast",
        "strengths": "Previous generation — 14 aspect ratios, 512/1K/2K/4K resolution, Google Web and Image Search grounding, up to 14 refs",
        "api_model": "gemini-3.1-flash-image",
        "aspect_ratios": _GEMINI_RATIOS_14,
        "resolutions": ("512", "1K", "2K", "4K"),
        "supports_search": True,
        "max_refs": 14,
    },
    "gemini-3.1-flash-lite-image": {
        "display": "Nano Banana 2 Lite (Gemini 3.1 Flash Lite Image)",
        "speed": "Fastest",
        "strengths": "Lowest latency & cost; 10 aspect ratios, 1K output, up to 14 reference images",
        "api_model": "gemini-3.1-flash-lite-image",
        "aspect_ratios": _GEMINI_RATIOS_10,
        "resolutions": ("1K",),
        "supports_search": False,
        "max_refs": 14,
    },
    "gemini-3-pro-image": {
        "display": "Nano Banana Pro (Gemini 3 Pro Image)",
        "speed": "Slower",
        "strengths": "Highest fidelity & reasoning — 10 aspect ratios, 1K/2K/4K, Google Search, up to 14 refs",
        "api_model": "gemini-3-pro-image",
        "aspect_ratios": _GEMINI_RATIOS_10,
        "resolutions": ("1K", "2K", "4K"),
        "supports_search": True,
        "max_refs": 14,
    },
}


def _load_image_gen_config(sub: Optional[str] = None) -> Dict[str, Any]:
    """Read ``image_gen`` (or ``image_gen.<sub>``) via public ``hermes_cli.config.load_config``."""
    label = "image_gen" if sub is None else f"image_gen.{sub}"
    try:
        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        if sub is not None:
            section = section.get(sub) if isinstance(section, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not load %s config: %s", label, exc)
        return {}


def _collect_source_images(
    image_url: Optional[str],
    reference_image_urls: Optional[List[str]],
    limit: Optional[int] = None,
) -> List[str]:
    sources: List[str] = []
    if isinstance(image_url, str) and image_url.strip():
        sources.append(image_url.strip())
    sources.extend(normalize_reference_images(reference_image_urls) or [])
    return sources[:limit] if limit is not None else sources


def _error_factory(
    provider: str, aspect: str, *, model: str = "", prompt: str = ""
) -> Callable[..., Dict[str, Any]]:
    def fail(error: str, error_type: str, **override: Any) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = dict(provider=provider, model=model, prompt=prompt, aspect_ratio=aspect)
        kwargs.update(override)
        return error_response(error=error, error_type=error_type, **kwargs)

    return fail


def _prompt_required_error(provider: str, aspect: str) -> Dict[str, Any]:
    return _error_factory(provider, aspect)(_PROMPT_REQUIRED, "invalid_argument")


def _strip_google_prefix(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    return raw.split("/", 1)[1].strip() if raw.lower().startswith("google/") else raw


def _custom_model_meta(model_id: str) -> Dict[str, Any]:
    """Conservative metadata for an uncatalogued Gemini image model ID."""
    return {
        "display": model_id,
        "api_model": model_id,
        "aspect_ratios": _GEMINI_RATIOS_10,
        "resolutions": (),
        "supports_search": False,
        "max_refs": _MAX_REFERENCE_IMAGES,
    }


def _resolve_model_selection(
    caller_model: Optional[str] = None, cfg: Optional[Dict[str, Any]] = None
) -> Tuple[str, Dict[str, Any], Optional[str]]:
    """Return ``(model_id, meta, explicit_override)``.

    ``explicit_override`` is ``None`` when no caller/env/config override is active, allowing
    ``gemini-api image`` to omit ``-m`` and own its built-in default across CLI upgrades.
    """
    cleaned = _strip_google_prefix(caller_model)
    if cleaned and cleaned in MODELS:
        return cleaned, MODELS[cleaned], cleaned
    if cleaned:
        logger.debug(
            "Ignoring image model id %r: not a Gemini model. Set image_gen.gemini.model or "
            "GEMINI_IMAGE_MODEL to choose one.",
            cleaned,
        )

    env_override = _strip_google_prefix(os.environ.get("GEMINI_IMAGE_MODEL"))
    if env_override:
        return env_override, MODELS.get(env_override) or _custom_model_meta(env_override), env_override

    scoped = _load_image_gen_config("gemini") if cfg is None else cfg
    cfg_model = _strip_google_prefix(scoped.get("model") if isinstance(scoped, dict) else None)
    if cfg_model:
        return cfg_model, MODELS.get(cfg_model) or _custom_model_meta(cfg_model), cfg_model

    return DEFAULT_MODEL, MODELS[DEFAULT_MODEL], None


def _resolve_model(caller_model: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    """``(model_id, meta)`` — caller catalog entry → ``GEMINI_IMAGE_MODEL`` → ``image_gen.gemini.model`` → default."""
    model_id, meta, _ = _resolve_model_selection(caller_model)
    return model_id, meta


def _named_endpoint(name: str) -> Tuple[str, str]:
    """``(base_url, api_key)`` of the user-declared custom endpoint *name* via public ``hermes_cli.config``."""
    target = name.strip().lower()
    if not target:
        return "", ""
    try:
        cfg = load_config_readonly() or {}
        providers = get_compatible_custom_providers(cfg) or []
    except Exception:  # noqa: BLE001
        providers = []

    entry = next(
        (
            p for p in providers
            if isinstance(p, dict) and str(p.get("name") or "").strip().lower() == target
        ),
        None,
    )
    if not entry:
        logger.warning("image_gen.gemini.provider %r matches no custom endpoint in providers:", name)
        return "", ""
    key_env = str(entry.get("key_env") or "").strip()
    api_key = str(entry.get("api_key") or "").strip() or (get_secret(key_env) if key_env else None) or ""
    return str(entry.get("base_url") or "").strip().rstrip("/"), api_key


def _resolve_endpoint(cfg: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
    """``(base_url, api_key)`` — ``image_gen.gemini.base_url`` → named endpoint → ``GEMINI_BASE_URL`` →
    :data:`BASE_URL`; the env var named by ``image_gen.gemini.key_env`` → named endpoint →
    ``GEMINI_API_KEY``. Shared across image, video, and CLI tools."""
    cfg = _load_image_gen_config("gemini") if cfg is None else cfg
    named = str(cfg.get("provider") or "").strip()
    named_base, named_key = _named_endpoint(named) if named else ("", "")
    base_url = (
        str(cfg.get("base_url") or "").strip().rstrip("/")
        or named_base
        or os.environ.get("GEMINI_BASE_URL", "").strip().rstrip("/")
        or BASE_URL
    )
    key_env = str(cfg.get("key_env") or "").strip()
    api_key = (
        (get_secret(key_env) if key_env else None)
        or named_key
        or get_secret("GEMINI_API_KEY")
        or ""
    ).strip()
    return base_url, api_key


def _resolve_exact_aspect_ratio(
    meta: Dict[str, Any], aspect_ratio: Optional[str], cfg: Optional[Dict[str, Any]] = None
) -> Tuple[str, str]:
    """``(semantic_aspect, wire_aspect)`` — an exact ratio (caller argument → ``image_gen.gemini.aspect_ratio``)
    wins when supported by *meta*; otherwise the semantic landscape/square/portrait mapping applies."""
    semantic = resolve_aspect_ratio(aspect_ratio)
    supported: Tuple[str, ...] = tuple(meta.get("aspect_ratios") or _GEMINI_RATIOS_10)
    raw_arg = (aspect_ratio or "").strip()
    cfg = _load_image_gen_config("gemini") if cfg is None else cfg
    for candidate in (
        raw_arg if raw_arg in supported else None,
        cfg.get("aspect_ratio"),
    ):
        if isinstance(candidate, str) and candidate.strip() in supported:
            exact = candidate.strip()
            w, h = (int(x) for x in exact.split(":"))
            return ("square" if w == h else ("landscape" if w > h else "portrait")), exact
    return semantic, _ASPECT_RATIOS[semantic]


def _resolve_image_size(
    meta: Dict[str, Any], upscale: bool = False, cfg: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """``--image-size`` (``512``/``1K``/``2K``/``4K``) when *meta* supports it."""
    supported: Tuple[str, ...] = tuple(meta.get("resolutions") or ())
    if not supported:
        return None
    cfg = _load_image_gen_config("gemini") if cfg is None else cfg
    for candidate in (
        "4K" if upscale and "4K" in supported else ("2K" if upscale and "2K" in supported else None),
        cfg.get("image_size") or cfg.get("resolution"),
    ):
        if isinstance(candidate, str) and candidate.strip():
            normalized = candidate.strip().upper()
            if normalized in supported:
                return normalized
    return None


def _resolve_google_search(meta: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None) -> bool:
    """Whether ``tools: [{"type": "google_search"}]`` grounding is enabled when *meta* supports it."""
    if not meta.get("supports_search"):
        return False
    cfg = _load_image_gen_config("gemini") if cfg is None else cfg
    return bool(cfg.get("google_search") or cfg.get("search_grounding"))


def _sniff_mime(data: bytes) -> str:
    """Detect supported image MIME type from magic bytes."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    raise ValueError("not a recognised image (expected PNG, JPEG, WebP or GIF)")


def _cap_message(ref: str) -> str:
    return f"exceeds the {_MAX_INPUT_IMAGE_BYTES // (1024 * 1024)}MB per-image cap: {ref}"


def _load_image_bytes(ref: str) -> Tuple[bytes, str]:
    """Load ``(data, mime_type)`` from a URL, data URI, or local path with SSRF and file-safety guards."""
    ref = ref.strip()
    lower = ref.lower()
    if lower.startswith(("http://", "https://")):
        from tools.url_safety import create_ssrf_safe_client

        chunks: List[bytes] = []
        total = 0
        with create_ssrf_safe_client(timeout=30.0) as client:
            with client.stream("GET", ref) as resp:
                resp.raise_for_status()
                for chunk in resp.iter_bytes():
                    total += len(chunk)
                    if total > _MAX_INPUT_IMAGE_BYTES:
                        raise ValueError(f"Image URL {_cap_message(ref)}")
                    chunks.append(chunk)
        raw = b"".join(chunks)
        return raw, _sniff_mime(raw)
    if lower.startswith("data:"):
        header, sep, b64 = ref.partition(",")
        if not sep or ";base64" not in header.lower():
            raise ValueError("Unsupported data URI for image input (expected base64-encoded image)")
        b64 = "".join(b64.split())
        if len(b64) > (_MAX_INPUT_IMAGE_BYTES // 3) * 4 + 4:
            raise ValueError(f"Image data URI {_cap_message('data:')}")
        raw = base64.b64decode(b64, validate=True)
        if len(raw) > _MAX_INPUT_IMAGE_BYTES:
            raise ValueError(f"Image data URI {_cap_message('data:')}")
        return raw, _sniff_mime(raw)
    from agent.file_safety import raise_if_read_blocked

    raise_if_read_blocked(ref)
    path = Path(os.path.expanduser(ref))
    if not path.is_file():
        raise ValueError(f"Image input path is not a regular file or does not exist: {ref}")
    if path.stat().st_size > _MAX_INPUT_IMAGE_BYTES:
        raise ValueError(f"Image input path {_cap_message(ref)}")
    raw = path.read_bytes()
    if len(raw) > _MAX_INPUT_IMAGE_BYTES:
        raise ValueError(f"Image input path {_cap_message(ref)}")
    return raw, _sniff_mime(raw)


class GeminiImageGenProvider(ImageGenProvider):
    """Hermes ``ImageGenProvider`` backed by ``gemini-api image`` (Interactions API)."""

    models = MODELS

    def __init__(self, *, runner: Any = None) -> None:
        if runner is None:
            runner_mod = _load_sibling_module("cli_runner")
            if runner_mod is None:
                raise ImportError("Unable to load sibling cli_runner module")
            runner = runner_mod.GeminiCLIRunner(
                endpoint_resolver=_resolve_endpoint,
                user_agent=CLI_USER_AGENT,
            )
        self._runner = runner

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def display_name(self) -> str:
        return "Google AI Studio"

    def list_models(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for model_id, meta in MODELS.items():
            rows.append({
                "id": model_id,
                "display": meta.get("display", model_id),
                "speed": meta.get("speed", ""),
                "strengths": meta.get("strengths", ""),
                "price": "varies",
            })
        return rows

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Google AI Studio (gemini-api CLI)",
            "badge": "paid",
            "tag": "Gemini (Nano Banana) image generation & editing via gemini-api CLI and GEMINI_API_KEY",
            "env_vars": [
                {
                    "key": "GEMINI_API_KEY",
                    "prompt": "Google AI Studio Gemini API key",
                    "url": "https://aistudio.google.com/apikey",
                },
            ],
        }

    def is_available(self) -> bool:
        return bool(self._runner.is_available())

    def capabilities(self) -> Dict[str, Any]:
        _, meta = _resolve_model()
        resolutions = set(meta.get("resolutions") or ())
        return {
            "modalities": ["text", "image"],
            "max_reference_images": int(meta.get("max_refs") or _MAX_REFERENCE_IMAGES),
            "supports_upscale": bool(resolutions & {"2K", "4K"}),
        }

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        *,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        cfg = _load_image_gen_config("gemini")
        model_id, meta, explicit_model = _resolve_model_selection(kwargs.get("model"), cfg)
        aspect, wire_aspect = _resolve_exact_aspect_ratio(meta, aspect_ratio, cfg)
        if not prompt:
            return _prompt_required_error("gemini", aspect)

        base_url, api_key = _resolve_endpoint(cfg)
        if not api_key:
            return _error_factory("gemini", aspect)(
                "GEMINI_API_KEY is not set (and image_gen.gemini.key_env is empty). "
                "Run `hermes tools` → Image Generation → Google AI Studio to configure.",
                "auth_required",
            )

        fail = _error_factory("gemini", aspect, model=model_id, prompt=prompt)
        runner_mod = _load_sibling_module("cli_runner")
        if runner_mod is not None and hasattr(runner_mod, "validate_base_url"):
            try:
                runner_mod.validate_base_url(base_url)
            except ValueError as exc:
                return fail(str(exc), "invalid_argument")

        out_dir = get_hermes_home() / "cache" / "generated" / "images"
        out_dir.mkdir(parents=True, exist_ok=True)

        cli_args: List[str] = ["image", "--out", str(out_dir)]
        if explicit_model:
            cli_args.extend(["-m", explicit_model])
        cli_args.extend(["--aspect-ratio", wire_aspect])

        image_size = _resolve_image_size(meta, bool(kwargs.get("upscale")), cfg)
        if image_size:
            cli_args.extend(["--image-size", image_size])

        use_google_search = _resolve_google_search(meta, cfg)
        if use_google_search:
            cli_args.extend(["--body", json.dumps({"tools": [{"type": "google_search"}]})])

        max_refs = int(meta.get("max_refs") or _MAX_REFERENCE_IMAGES)
        sources = _collect_source_images(image_url, reference_image_urls, limit=max_refs)
        is_edit = bool(sources)

        tmp_files: List[Path] = []
        inline_used = 0
        default_timeout = int(getattr(runner_mod, "DEFAULT_CLI_TIMEOUT", 180)) if runner_mod else 180
        try:
            for ref in sources:
                if ref.startswith("files/"):
                    valid_ref, ref_err, added_cost = self._runner.prepare_media_input(
                        ref,
                        allow_youtube=False,
                        inline_budget_used=inline_used,
                        enforce_cumulative=True,
                    )
                elif ref.lower().startswith(("http://", "https://", "data:")):
                    try:
                        raw_bytes, mime = _load_image_bytes(ref)
                    except Exception as exc:  # noqa: BLE001
                        return fail(f"Could not load reference image: {exc}", "invalid_argument")
                    suffix = _MIME_TO_SUFFIX.get(mime, ".png")
                    with tempfile.NamedTemporaryFile(
                        prefix="hermes-image-ref-", suffix=suffix, delete=False
                    ) as tmp_file:
                        tmp_path = Path(tmp_file.name)
                        tmp_file.write(raw_bytes)
                    if os.name != "nt":
                        tmp_path.chmod(0o600)
                    tmp_files.append(tmp_path)
                    valid_ref, ref_err, added_cost = self._runner.prepare_media_input(
                        str(tmp_path),
                        allow_youtube=False,
                        mime_type=mime,
                        inline_budget_used=inline_used,
                        enforce_cumulative=True,
                    )
                else:
                    try:
                        _, mime = _load_image_bytes(ref)
                    except Exception as exc:  # noqa: BLE001
                        return fail(f"Could not load reference image: {exc}", "invalid_argument")
                    valid_ref, ref_err, added_cost = self._runner.prepare_media_input(
                        ref,
                        allow_youtube=False,
                        mime_type=mime,
                        inline_budget_used=inline_used,
                        enforce_cumulative=True,
                    )

                if ref_err is not None or not valid_ref:
                    return fail(
                        ref_err.error if ref_err is not None else "Invalid reference image input.",
                        (ref_err.error_type if ref_err is not None else "") or "invalid_argument",
                    )
                cli_args.extend(["-i", valid_ref])
                inline_used += added_cost

            res = self._runner.run(
                cli_args,
                positional_args=[prompt],
                default_timeout=default_timeout,
                default_model=meta["api_model"],
            )
        finally:
            for tmp_path in tmp_files:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass

        if not res.ok:
            return fail(res.error or "gemini-api image failed", res.error_type or "provider_error")

        payload = res.data if isinstance(res.data, dict) else {}
        saved_path = str(payload.get("path") or "").strip()
        if not saved_path or not Path(saved_path).is_file():
            return fail(
                "gemini-api image completed without returning a valid output image file path.",
                "invalid_response",
            )

        reported_model = str(payload.get("model") or model_id).strip()
        extra: Dict[str, Any] = {"api": "google-ai-studio", "exact_aspect_ratio": wire_aspect}
        if image_size:
            extra["image_size"] = image_size
        if kwargs.get("upscale") and image_size in ("2K", "4K"):
            extra["upscaled"] = True
        if use_google_search:
            extra["google_search"] = True
        if payload.get("response_id"):
            extra["response_id"] = payload["response_id"]
        if payload.get("mime_type"):
            extra["mime_type"] = payload["mime_type"]
        return success_response(
            image=saved_path,
            model=reported_model,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="gemini",
            modality="image" if is_edit else "text",
            extra=extra,
        )


def _load_cli_tools_module():
    """Import sibling ``cli_tools`` across package, loader, and standalone test contexts."""
    return _load_sibling_module("cli_tools")


def register(ctx) -> None:
    """Plugin entry point — wire CLI-backed Gemini image/video providers, tools, and skill."""
    runner_mod = _load_sibling_module("cli_runner")
    shared_runner = (
        runner_mod.GeminiCLIRunner(endpoint_resolver=_resolve_endpoint, user_agent=CLI_USER_AGENT)
        if runner_mod is not None
        else None
    )
    ctx.register_image_gen_provider(GeminiImageGenProvider(runner=shared_runner))
    cli_tools = _load_cli_tools_module()
    if cli_tools is not None:
        cli_tools.register_cli_integrations(
            ctx,
            endpoint_resolver=_resolve_endpoint,
            image_loader=_load_image_bytes,
            user_agent=CLI_USER_AGENT,
            runner=shared_runner,
        )
