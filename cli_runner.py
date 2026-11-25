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

"""Subprocess runner, binary discovery, and opt-in pinned installer for ``gemini-api``.

Wraps ``google-gemini/gemini-api-cli`` in non-interactive JSON mode (``--no-interactive -o json``)
with Hermes profile-scoped ``GEMINI_API_KEY`` isolation, PR #2 ``User-Agent`` attribution,
``agent.file_safety`` local-path checks, ``--`` positional argument flag-injection guards, and
structured error / token-usage extraction.
"""

from __future__ import annotations

import errno
import hashlib
import io
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

PINNED_CLI_VERSION = "0.5.0"
# Verified SHA-256 digests from https://github.com/google-gemini/gemini-api-cli/releases/download/v0.5.0/checksums.txt
PINNED_CLI_SHA256: Dict[str, str] = {
    "gemini-api_Darwin_arm64.tar.gz": "ae31afd60148edc4b8550811d406b875c4b74164ca96a5782ff47bf60057aef7",
    "gemini-api_Darwin_x86_64.tar.gz": "006c1f58e2c2711b9a7c6d537e4db4f930df49402035bcdfdefdd25384d07989",
    "gemini-api_Linux_arm64.tar.gz": "cea63e70982a06f81069318049bfca6db49042649c5ae635b3433cb3a2420d20",
    "gemini-api_Linux_x86_64.tar.gz": "fce229dd482ec1caff15cd147a293513ffb5a52390394e52136f85b4a497c742",
    "gemini-api_Windows_arm64.zip": "a64df8f4b0ec394de4a42b8be96831942ec850463aacc77d41c6b04f07af2963",
    "gemini-api_Windows_x86_64.zip": "d2bc738fb8c83f663e1fca5e3382876dcbde54880c7375cfcb49351298b264ec",
}
CLI_INLINE_LIMIT_BYTES = 14 * 1024 * 1024        # ~14 MiB single-file raw binary limit (~19 MiB base64)
CLI_CUMULATIVE_INLINE_BYTES = 19 * 1024 * 1024   # 19 MiB cumulative base64 request budget in gemini-api
CLI_MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024    # 2 GiB Gemini Files API ceiling
DEFAULT_CLI_TIMEOUT = 180
DEFAULT_RESEARCH_TIMEOUT = 900

_MAX_CHECKSUMS_BYTES = 64 * 1024        # 64 KiB download ceiling for checksums.txt
_MAX_ARCHIVE_BYTES = 64 * 1024 * 1024   # 64 MiB download ceiling for gemini-api release archives
_MAX_BINARY_BYTES = 128 * 1024 * 1024   # 128 MiB extracted binary ceiling
_CLI_CACHE_TTL_SECONDS = 5.0
_MAX_CACHE_ENTRIES = 64

# Safe non-secret process environment variables forwarded to the gemini-api child process.
# Unrelated secrets (OpenAI, Anthropic, GitHub, etc.), HERMES_* state, and conflicting
# GEMINI_* / GOOGLE_API_KEY variables are never inherited.
_SAFE_CHILD_ENV_KEYS = frozenset({
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "TERM", "COLORTERM",
    "TMPDIR", "TEMP", "TMP", "LANG", "LANGUAGE", "TZ",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "all_proxy", "no_proxy",
    "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME", "XDG_RUNTIME_DIR",
    "SYSTEMROOT", "COMSPEC", "WINDIR", "PATHEXT",
    "APPDATA", "LOCALAPPDATA", "USERPROFILE", "HOMEDRIVE", "HOMEPATH",
})

CLI_INSTALL_HINT = (
    f"gemini-api CLI binary not found. Install v{PINNED_CLI_VERSION} with "
    f"`go install github.com/google-gemini/gemini-api-cli/cmd/gemini-api@v{PINNED_CLI_VERSION}` "
    f"or place the release binary from https://github.com/google-gemini/gemini-api-cli/releases/tag/v{PINNED_CLI_VERSION} "
    "into ~/.hermes/bin/gemini-api (or set plugins.entries.gemini-api.settings.cli_path in config.yaml)."
)

_DEFAULT_SERVER_URL = "https://generativelanguage.googleapis.com"
_API_VERSION_SUFFIXES = ("/v1beta", "/v1alpha", "/v1")
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_YOUTUBE_HOST_RE = re.compile(r"^(?:www\.|m\.|music\.)?(?:youtube\.com|youtu\.be)$", re.IGNORECASE)
_FILE_ID_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,38}[a-z0-9])?$")
# Extensions that gemini-api sends strictly as inline text blocks or rejects in analyze/transcribe
# (must NOT be auto-uploaded to Files API — let gemini-api report its fast local diagnostic instead).
_NON_UPLOADABLE_ANALYZE_EXTS = frozenset({
    ".txt", ".md", ".html", ".htm", ".xml", ".json", ".js", ".ts", ".py", ".go", ".css", ".yaml", ".yml",
    ".amr", ".wma", ".mkv", ".rtf",
})

_cache_lock = threading.Lock()
_install_lock = threading.Lock()
_config_cache: Dict[str, Tuple[int, int, Dict[str, Any]]] = {}
_binary_cache: Dict[Tuple[str, str, str], Tuple[float, Optional[str]]] = {}


@dataclass
class CLIResult:
    """Normalized result from a ``gemini-api`` subprocess invocation."""

    ok: bool
    exit_code: int
    data: Any = None
    stdout: str = ""
    stderr: str = ""
    error: str = ""
    error_type: str = ""
    hints: List[str] = field(default_factory=list)


def clear_cli_cache() -> None:
    """Clear short-lived binary and config caches (primarily for unit tests)."""
    with _cache_lock:
        _config_cache.clear()
        _binary_cache.clear()


def _read_cached_config() -> Dict[str, Any]:
    """Read ``$HERMES_HOME/config.yaml`` keyed by ``(mtime_ns, size)`` to avoid per-turn YAML thrash."""
    home = get_hermes_home()
    cfg_path = home / "config.yaml"
    cache_key = str(cfg_path)
    try:
        st = cfg_path.stat()
    except OSError:
        with _cache_lock:
            _config_cache.pop(cache_key, None)
        return {}
    with _cache_lock:
        cached = _config_cache.get(cache_key)
    if cached is not None and cached[0] == st.st_mtime_ns and cached[1] == st.st_size:
        return cached[2]
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly() or {}
    except Exception:  # noqa: BLE001
        return {}
    normalized = cfg if isinstance(cfg, dict) else {}
    with _cache_lock:
        if len(_config_cache) >= _MAX_CACHE_ENTRIES:
            _config_cache.clear()
        _config_cache[cache_key] = (st.st_mtime_ns, st.st_size, normalized)
    return normalized


def load_plugin_settings(plugin_ids: Sequence[str] = ("gemini-api", "gemini-image")) -> Dict[str, Any]:
    """Read ``plugins.entries.<id>.settings`` (or legacy ``config``) from ``config.yaml``."""
    cfg = _read_cached_config()
    entries = ((cfg.get("plugins") or {}).get("entries")) if isinstance(cfg, dict) else None
    if not isinstance(entries, dict):
        return {}
    for pid in plugin_ids:
        entry = entries.get(pid)
        if not isinstance(entry, dict):
            continue
        for key in ("settings", "config"):
            block = entry.get(key)
            if isinstance(block, dict) and block:
                return dict(block)
    return {}


def _is_executable_file(path: Path) -> bool:
    try:
        return path.is_file() and os.access(path, os.X_OK)
    except OSError:
        return False


def _binary_filename() -> str:
    return "gemini-api.exe" if sys.platform.lower() == "win32" else "gemini-api"


def resolve_cli_binary(cli_path: Optional[str] = None) -> Optional[str]:
    """Locate an executable ``gemini-api`` binary without downloading anything.

    Order: explicit *cli_path* → ``plugins.entries.gemini-api.settings.cli_path`` →
    ``$HERMES_HOME/bin/gemini-api`` → ``PATH`` → common user Go / local bin directories.
    """
    settings = load_plugin_settings() if cli_path is None else {}
    configured = str((cli_path if cli_path is not None else settings.get("cli_path")) or "").strip()
    if configured:
        candidate = Path(os.path.expanduser(configured))
        return str(candidate) if _is_executable_file(candidate) else None

    hermes_home = get_hermes_home()
    path_env = os.environ.get("PATH", "")
    binary_name = _binary_filename()
    cache_key = (str(hermes_home), path_env, binary_name)
    now = time.monotonic()
    with _cache_lock:
        cached = _binary_cache.get(cache_key)
    if cached is not None and (now - cached[0]) < _CLI_CACHE_TTL_SECONDS:
        if cached[1] is None or _is_executable_file(Path(cached[1])):
            return cached[1]

    resolved: Optional[str] = None
    hermes_bin = hermes_home / "bin" / binary_name
    if _is_executable_file(hermes_bin):
        resolved = str(hermes_bin)
    else:
        found_on_path = shutil.which("gemini-api")
        if found_on_path:
            resolved = found_on_path
        else:
            home = Path.home()
            for fallback in (
                home / ".local" / "bin" / binary_name,
                home / "go" / "bin" / binary_name,
                Path("/usr/local/bin") / binary_name,
            ):
                if _is_executable_file(fallback):
                    resolved = str(fallback)
                    break

    with _cache_lock:
        if len(_binary_cache) >= _MAX_CACHE_ENTRIES:
            _binary_cache.clear()
        _binary_cache[cache_key] = (now, resolved)
    return resolved


def _platform_release_asset(version: str = PINNED_CLI_VERSION) -> Tuple[str, str]:
    """Return ``(archive_filename, binary_name)`` matching GoReleaser assets on GitHub Releases."""
    del version  # GitHub release assets omit version from filename (e.g. gemini-api_Linux_x86_64.tar.gz)
    sys_name = sys.platform.lower()
    if sys_name.startswith("linux"):
        os_slug, ext, binary_name = "Linux", "tar.gz", "gemini-api"
    elif sys_name == "darwin":
        os_slug, ext, binary_name = "Darwin", "tar.gz", "gemini-api"
    elif sys_name == "win32":
        os_slug, ext, binary_name = "Windows", "zip", "gemini-api.exe"
    else:
        raise ValueError(f"Unsupported OS for gemini-api release archive: {sys.platform}")

    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        arch_slug = "x86_64"
    elif machine in ("aarch64", "arm64"):
        arch_slug = "arm64"
    else:
        raise ValueError(f"Unsupported CPU architecture for gemini-api release archive: {machine}")

    return f"gemini-api_{os_slug}_{arch_slug}.{ext}", binary_name


def _expected_sha256_from_checksums(checksums_text: str, asset_name: str) -> str:
    for raw_line in checksums_text.splitlines():
        parts = raw_line.strip().split()
        if len(parts) >= 2 and parts[1].lstrip("*") == asset_name:
            digest = parts[0].strip().lower()
            if len(digest) == 64 and all(c in "0123456789abcdef" for c in digest):
                return digest
    raise ValueError(f"Asset {asset_name!r} not found in release checksums.txt")


def _read_bounded_archive(client: Any, archive_url: str, max_bytes: Optional[int] = None) -> bytes:
    """Download *archive_url* while enforcing a hard streaming byte ceiling."""
    limit = _MAX_ARCHIVE_BYTES if max_bytes is None else max_bytes
    stream_fn = getattr(client, "stream", None)
    if callable(stream_fn):
        chunks: List[bytes] = []
        total = 0
        with stream_fn("GET", archive_url) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_bytes():
                total += len(chunk)
                if total > limit:
                    raise ValueError(f"Release download exceeded {limit} byte safety limit")
                chunks.append(chunk)
        return b"".join(chunks)

    resp = client.get(archive_url)
    resp.raise_for_status()
    raw_bytes = resp.content
    if len(raw_bytes) > limit:
        raise ValueError(f"Release download exceeded {limit} byte safety limit")
    return raw_bytes


def _extract_binary_from_archive(archive_bytes: bytes, asset_name: str, binary_name: str) -> bytes:
    """Extract *binary_name* from a ``.tar.gz`` or ``.zip`` release archive with size caps."""
    if asset_name.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
            info = next(
                (i for i in zf.infolist() if not i.is_dir() and Path(i.filename).name == binary_name),
                None,
            )
            if info is None:
                raise ValueError(f"Binary {binary_name!r} not found in archive {asset_name}")
            if info.file_size > _MAX_BINARY_BYTES:
                raise ValueError(
                    f"Extracted {binary_name!r} exceeds {_MAX_BINARY_BYTES // (1024 * 1024)} MiB limit"
                )
            with zf.open(info, "r") as extracted:
                payload = extracted.read(_MAX_BINARY_BYTES + 1)
    else:
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tf:
            member = next(
                (m for m in tf.getmembers() if m.isfile() and Path(m.name).name == binary_name),
                None,
            )
            if member is None:
                raise ValueError(f"Binary {binary_name!r} not found in archive {asset_name}")
            if member.size > _MAX_BINARY_BYTES:
                raise ValueError(
                    f"Extracted {binary_name!r} exceeds {_MAX_BINARY_BYTES // (1024 * 1024)} MiB limit"
                )
            extracted = tf.extractfile(member)
            if extracted is None:
                raise ValueError(f"Could not extract {binary_name!r} from archive {asset_name}")
            payload = extracted.read(_MAX_BINARY_BYTES + 1)

    if len(payload) > _MAX_BINARY_BYTES:
        raise ValueError(f"Extracted {binary_name!r} exceeds {_MAX_BINARY_BYTES // (1024 * 1024)} MiB limit")
    if not payload:
        raise ValueError(f"Extracted {binary_name!r} binary was empty")
    return payload


def install_cli_binary(
    *,
    version: str = PINNED_CLI_VERSION,
    target_dir: Optional[Path] = None,
    expected_sha256: Optional[str] = None,
) -> Path:
    """Download and SHA-256-verify a pinned ``gemini-api`` release into ``$HERMES_HOME/bin``.

    For the pinned release (:data:`PINNED_CLI_VERSION`), verification uses the hardcoded digest in
    :data:`PINNED_CLI_SHA256` rather than trusting a remote ``checksums.txt``. Never called
    automatically unless ``plugins.entries.gemini-api.settings.auto_install_cli`` is explicitly
    set to ``true`` by the user (off by default per Hermes catalog policy).
    """
    from tools.url_safety import create_ssrf_safe_client

    ver = version.lstrip("v")
    asset_name, binary_name = _platform_release_asset(ver)
    base = f"https://github.com/google-gemini/gemini-api-cli/releases/download/v{ver}"
    checksums_url = f"{base}/checksums.txt"
    archive_url = f"{base}/{asset_name}"

    pinned_sha = (
        expected_sha256.strip().lower()
        if isinstance(expected_sha256, str) and expected_sha256.strip()
        else (PINNED_CLI_SHA256.get(asset_name) if ver == PINNED_CLI_VERSION else None)
    )

    with _install_lock:
        with create_ssrf_safe_client(timeout=60.0, follow_redirects=True) as client:
            if pinned_sha:
                expected_sha = pinned_sha
            else:
                cs_bytes = _read_bounded_archive(client, checksums_url, max_bytes=_MAX_CHECKSUMS_BYTES)
                expected_sha = _expected_sha256_from_checksums(
                    cs_bytes.decode("utf-8", errors="replace"), asset_name
                )
            archive_bytes = _read_bounded_archive(client, archive_url)

        actual_sha = hashlib.sha256(archive_bytes).hexdigest().lower()
        if actual_sha != expected_sha:
            raise ValueError(
                f"SHA-256 mismatch for {asset_name}: expected {expected_sha}, got {actual_sha}"
            )

        payload = _extract_binary_from_archive(archive_bytes, asset_name, binary_name)

        dest_dir = Path(target_dir) if target_dir is not None else (get_hermes_home() / "bin")
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / binary_name

        tmp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=dest_dir, prefix=f".{binary_name}.tmp.", delete=False
            ) as tmp_file:
                tmp_path = Path(tmp_file.name)
                tmp_file.write(payload)
            if os.name != "nt":
                tmp_path.chmod(0o755)
            tmp_path.replace(dest_path)
            tmp_path = None
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass

        clear_cli_cache()
        return dest_path


def validate_base_url(base_url: str) -> str:
    """Validate and normalize a Gemini API base URL.

    Refuses cleartext ``http://`` URLs (except loopback hosts for local testing), leading dashes,
    embedded credentials, query parameters, and fragments so ``GEMINI_API_KEY`` is never sent over
    unencrypted links or flag-injected values.
    """
    cleaned = (base_url or "").strip().rstrip("/")
    if not cleaned:
        return ""
    if cleaned.startswith("-"):
        raise ValueError(f"Invalid Gemini base_url {base_url!r}: must be an https:// URL")

    parsed = urlsplit(cleaned)
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower()
    if not scheme or not host or host.startswith("-"):
        raise ValueError(f"Invalid Gemini base_url {base_url!r}: must include scheme and hostname")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Invalid Gemini base_url: embedded credentials, query parameters, and fragments are not allowed")
    if scheme != "https" and not (scheme == "http" and host in _LOOPBACK_HOSTS):
        raise ValueError(
            f"Refusing insecure Gemini base_url {base_url!r}: use https:// (http:// is allowed only for localhost)"
        )
    return cleaned


def server_flags_from_base_url(base_url: str) -> List[str]:
    """Translate validated ``https://host/v1beta`` endpoint URLs into ``gemini-api`` flags."""
    cleaned = validate_base_url(base_url)
    if not cleaned:
        return []

    api_version: Optional[str] = None
    for suffix in _API_VERSION_SUFFIXES:
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)].rstrip("/")
            if suffix != "/v1beta":
                api_version = suffix.lstrip("/")
            break
    flags: List[str] = []
    if cleaned and cleaned != _DEFAULT_SERVER_URL:
        flags.extend(["--server-url", cleaned])
    if api_version:
        flags.extend(["--api-version", api_version])
    return flags


def _build_child_env(binary_path: str, api_key: str) -> Dict[str, str]:
    """Construct a scrubbed, allowlisted environment for ``gemini-api`` subprocesses.

    Avoids inheriting unrelated process secrets (OpenAI, Anthropic, GitHub, etc.), Hermes profile
    state, or conflicting ``GEMINI_*`` / ``GOOGLE_API_KEY`` variables (such as ``GEMINI_ACCESS_TOKEN``,
    which takes precedence over ``GEMINI_API_KEY`` in ``gemini-api``).
    """
    env: Dict[str, str] = {}
    for key, val in os.environ.items():
        if key in _SAFE_CHILD_ENV_KEYS or key.startswith("LC_"):
            env[key] = val
    env["GEMINI_API_KEY"] = api_key
    env["NO_COLOR"] = "1"
    bin_parent = str(Path(binary_path).parent)
    current_path = env.get("PATH", "")
    if bin_parent and bin_parent not in current_path.split(os.pathsep):
        env["PATH"] = f"{bin_parent}{os.pathsep}{current_path}" if current_path else bin_parent
    return env


def _exit_code_to_error_type(exit_code: int) -> str:
    if exit_code == 3:
        return "auth_required"
    if exit_code == 2:
        return "invalid_argument"
    return "api_error"


def _parse_cli_error(stderr: str, stdout: str, exit_code: int) -> Tuple[str, List[str]]:
    """Extract ``(message, hints)`` from ``gemini-api`` JSON stderr or fallback text."""
    for stream in (stderr, stdout):
        text = (stream or "").strip()
        if not text:
            continue
        for candidate in (text, text.splitlines()[-1]):
            try:
                parsed = json.loads(candidate)
            except (ValueError, TypeError):
                continue
            if isinstance(parsed, dict) and parsed.get("error"):
                raw_err = parsed["error"]
                msg = str(raw_err.get("message") if isinstance(raw_err, dict) else raw_err)
                hints_raw = parsed.get("hints")
                hints = [str(h) for h in hints_raw if isinstance(h, str)] if isinstance(hints_raw, list) else []
                if any("files upload" in h for h in hints):
                    hints.append("Call gemini_files_upload first and pass the returned files/<id> handle.")
                return msg, hints
    fallback = (stderr or stdout or f"gemini-api exited with code {exit_code}").strip()
    return fallback, []


def _usage_namespace(usage_obj: Dict[str, Any]) -> Optional[SimpleNamespace]:
    """Normalize ``gemini-api`` Interactions API, porcelain, or camelCase usage dicts for accounting."""
    def _num(*keys: str) -> int:
        for k in keys:
            val = usage_obj.get(k)
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                return int(val)
        return 0

    prompt = _num("total_input_tokens", "input_tokens", "prompt_tokens", "promptTokenCount")
    output = _num("total_output_tokens", "output_tokens", "completion_tokens", "candidatesTokenCount")
    thoughts = _num(
        "total_thought_tokens", "thoughts_tokens", "thought_tokens", "reasoning_tokens", "thoughtsTokenCount"
    )
    total = _num("total_tokens", "totalTokenCount") or (prompt + output + thoughts)
    cached = _num("total_cached_tokens", "cached_tokens", "cached_input_tokens", "cachedContentTokenCount")
    if not (prompt or output or thoughts or total):
        return None
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=output + thoughts,
        total_tokens=total,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
    )


def record_gemini_token_usage(usage_ns: Any, *, model: str, base_url: str) -> None:
    """Best-effort bridge to Hermes's session token accounting (sole pending public-API seam)."""
    if usage_ns is None:
        return
    try:
        from plugins.image_gen._common import record_token_usage
    except ImportError:
        return
    record_token_usage(usage_ns, model=model, provider="gemini", base_url=base_url)


def maybe_record_cli_usage(data: Any, *, default_model: str, base_url: str) -> None:
    """Record ``usage`` / ``usageMetadata`` from a ``gemini-api`` JSON payload when present."""
    if not isinstance(data, dict):
        return

    top_model = str(data.get("model") or default_model or "gemini").strip()
    usage_entries: List[Tuple[str, Dict[str, Any]]] = []
    for key in ("usage", "usageMetadata", "usage_metadata"):
        if isinstance(data.get(key), dict):
            usage_entries.append((top_model, data[key]))
    if isinstance(data.get("results"), list):
        for item in data["results"]:
            if isinstance(item, dict) and isinstance(item.get("usage"), dict):
                item_model = str(item.get("model") or top_model).strip()
                usage_entries.append((item_model, item["usage"]))
    for entry_model, raw_usage in usage_entries:
        ns = _usage_namespace(raw_usage)
        if ns is not None:
            record_gemini_token_usage(ns, model=entry_model, base_url=base_url)


class GeminiCLIRunner:
    """Executes ``gemini-api`` subcommands with structured JSON output and safety checks."""

    def __init__(
        self,
        *,
        endpoint_resolver: Callable[[], Tuple[str, str]],
        user_agent: str,
        cli_path: Optional[str] = None,
    ) -> None:
        self._resolve_endpoint = endpoint_resolver
        self._user_agent = user_agent
        self._cli_path = cli_path

    def is_available(self) -> bool:
        """Fast availability probe shared across ``GeminiVideoGenProvider`` and all 6 CLI tools."""
        if not self._resolve_endpoint()[1]:
            return False
        if resolve_cli_binary(self._cli_path):
            return True
        return load_plugin_settings().get("auto_install_cli") is True

    def ensure_binary(self) -> Tuple[Optional[str], Optional[str]]:
        """Return ``(binary_path, error_message)``; auto-installs only when opted in via settings."""
        binary = resolve_cli_binary(self._cli_path)
        if binary:
            return binary, None
        settings = load_plugin_settings()
        if settings.get("auto_install_cli") is True:
            try:
                installed = install_cli_binary()
                return str(installed), None
            except Exception as exc:  # noqa: BLE001
                return None, f"Automatic gemini-api CLI install failed ({exc}). {CLI_INSTALL_HINT}"
        return None, CLI_INSTALL_HINT

    def run(
        self,
        args: Sequence[str],
        *,
        positional_args: Sequence[str] = (),
        timeout: Optional[int] = None,
        default_timeout: int = DEFAULT_CLI_TIMEOUT,
        default_model: str = "",
        cwd: Optional[Path] = None,
    ) -> CLIResult:
        """Run ``gemini-api <args> --no-interactive -o json -- <positional_args>`` and parse JSON."""
        base_url, api_key = self._resolve_endpoint()
        if not api_key:
            return CLIResult(
                ok=False,
                exit_code=3,
                error="GEMINI_API_KEY is not set. Configure GEMINI_API_KEY in ~/.hermes/.env.",
                error_type="auth_required",
            )

        try:
            server_flags = server_flags_from_base_url(base_url)
        except ValueError as exc:
            return CLIResult(
                ok=False,
                exit_code=2,
                error=str(exc),
                error_type="invalid_argument",
            )

        binary, bin_err = self.ensure_binary()
        if not binary:
            return CLIResult(
                ok=False,
                exit_code=127,
                error=bin_err or CLI_INSTALL_HINT,
                error_type="missing_dependency",
            )

        settings = load_plugin_settings()
        effective_timeout = timeout
        if effective_timeout is None:
            cfg_timeout = settings.get("timeout_seconds")
            effective_timeout = (
                int(cfg_timeout)
                if isinstance(cfg_timeout, int) and not isinstance(cfg_timeout, bool) and cfg_timeout > 0
                else default_timeout
            )

        cmd: List[str] = [
            binary,
            *args,
            "--no-interactive",
            "-o",
            "json",
            "--header",
            f"User-Agent: {self._user_agent}",
            *server_flags,
        ]
        if positional_args:
            cmd.append("--")
            cmd.extend(positional_args)

        env = _build_child_env(binary, api_key)
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=effective_timeout,
                cwd=str(cwd) if cwd is not None else None,
                env=env,
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            return CLIResult(
                ok=False,
                exit_code=124,
                error=f"gemini-api timed out after {effective_timeout}s",
                error_type="timeout",
            )
        except OSError as exc:
            if isinstance(exc, (FileNotFoundError, PermissionError)) or exc.errno in (
                errno.ENOENT,
                errno.EACCES,
                errno.ENOEXEC,
            ):
                return CLIResult(
                    ok=False,
                    exit_code=127,
                    error=f"Failed to execute gemini-api ({exc}). {CLI_INSTALL_HINT}",
                    error_type="missing_dependency",
                )
            return CLIResult(
                ok=False,
                exit_code=1,
                error=f"Failed to execute gemini-api ({exc}).",
                error_type="execution_error",
            )

        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        if proc.returncode != 0:
            msg, hints = _parse_cli_error(stderr, stdout, proc.returncode)
            if hints:
                msg = f"{msg} (Hints: {'; '.join(hints)})"
            return CLIResult(
                ok=False,
                exit_code=proc.returncode,
                stdout=stdout,
                stderr=stderr,
                error=msg,
                error_type=_exit_code_to_error_type(proc.returncode),
                hints=hints,
            )

        parsed: Any = None
        if stdout:
            try:
                parsed = json.loads(stdout)
            except ValueError:
                parsed = {"raw_output": stdout}
        maybe_record_cli_usage(parsed, default_model=default_model, base_url=base_url)
        return CLIResult(ok=True, exit_code=0, data=parsed, stdout=stdout, stderr=stderr)

    def _validate_local_file(self, raw_path: str) -> Tuple[Optional[Path], Optional[CLIResult]]:
        """Check *raw_path* (and its symlink-resolved target) against ``raise_if_read_blocked``."""
        from agent.file_safety import raise_if_read_blocked

        cleaned = (raw_path or "").strip()
        if not cleaned:
            return None, CLIResult(ok=False, exit_code=2, error="File path is required.", error_type="invalid_argument")
        try:
            # Check raw string first (catches Windows NT-namespace prefixes before normalization),
            # then resolve symlinks so a symlink pointing to a blocked credential file cannot bypass it.
            raise_if_read_blocked(cleaned)
            path = Path(os.path.expanduser(cleaned)).resolve()
            raise_if_read_blocked(str(path))
        except Exception as exc:  # noqa: BLE001
            return None, CLIResult(ok=False, exit_code=2, error=str(exc), error_type="invalid_argument")

        if not path.is_file():
            return None, CLIResult(
                ok=False,
                exit_code=2,
                error=f"Input path is not a regular file or does not exist: {cleaned}",
                error_type="invalid_argument",
            )
        return path, None

    def upload_file(
        self,
        local_path: str,
        *,
        display_name: Optional[str] = None,
        mime_type: Optional[str] = None,
        wait: bool = True,
        timeout: Optional[int] = None,
        default_timeout: int = 300,
    ) -> CLIResult:
        """Validate *local_path* against ``raise_if_read_blocked`` and run ``files upload``."""
        path, err = self._validate_local_file(local_path)
        if err is not None or path is None:
            return err or CLIResult(ok=False, exit_code=2, error="Invalid file path.", error_type="invalid_argument")

        size = path.stat().st_size
        if size <= 0:
            return CLIResult(
                ok=False, exit_code=2, error=f"Input file is empty: {local_path}", error_type="invalid_argument"
            )
        if size > CLI_MAX_UPLOAD_BYTES:
            return CLIResult(
                ok=False,
                exit_code=2,
                error=f"Input file exceeds the 2GB Gemini Files API limit: {local_path}",
                error_type="invalid_argument",
            )

        args: List[str] = ["files", "upload"]
        if wait:
            args.append("--wait")
        if display_name and display_name.strip():
            dn = display_name.strip()
            if dn.startswith("-"):
                return CLIResult(
                    ok=False,
                    exit_code=2,
                    error="display_name cannot start with a dash.",
                    error_type="invalid_argument",
                )
            args.extend(["--display-name", dn])
        if mime_type and mime_type.strip():
            mt = mime_type.strip()
            if mt.startswith("-"):
                return CLIResult(
                    ok=False,
                    exit_code=2,
                    error="mime_type cannot start with a dash.",
                    error_type="invalid_argument",
                )
            args.extend(["--mime-type", mt])
        return self.run(args, positional_args=[str(path)], timeout=timeout, default_timeout=default_timeout)

    def prepare_media_input(
        self,
        raw_input: str,
        *,
        allow_youtube: bool = True,
        mime_type: Optional[str] = None,
        inline_budget_used: int = 0,
        enforce_cumulative: bool = False,
    ) -> Tuple[Optional[str], Optional[CLIResult], int]:
        """Validate a local path, ``files/<id>`` handle, or YouTube URL; seamlessly auto-upload >14 MiB binary media.

        Returns ``(resolved_ref, error_result, inline_cost_added)``. Plain-text files and unsupported container
        extensions stay local so ``gemini-api`` runs its own fast local validation without wasted uploads.
        """
        ref = (raw_input or "").strip()
        if not ref:
            return (
                None,
                CLIResult(
                    ok=False, exit_code=2, error="Media input reference cannot be empty.", error_type="invalid_argument"
                ),
                0,
            )
        if ref.startswith("files/"):
            file_id = ref[len("files/"):]
            if not _FILE_ID_RE.fullmatch(file_id):
                return (
                    None,
                    CLIResult(
                        ok=False,
                        exit_code=2,
                        error=f"Invalid Gemini Files API handle {ref!r}: expected 'files/<lowercase-alnum-id>'.",
                        error_type="invalid_argument",
                    ),
                    0,
                )
            return ref, None, 0

        lower = ref.lower()
        if lower.startswith(("http://", "https://")):
            parsed = urlsplit(ref)
            host = (parsed.hostname or "").lower()
            if allow_youtube and _YOUTUBE_HOST_RE.fullmatch(host):
                return ref, None, 0
            return (
                None,
                CLIResult(
                    ok=False,
                    exit_code=2,
                    error=(
                        f"Unsupported remote URL {ref!r}: pass a local file path, a 'files/<id>' handle, "
                        + ("or a YouTube URL." if allow_youtube else "not a remote URL.")
                    ),
                    error_type="invalid_argument",
                ),
                0,
            )

        path, err = self._validate_local_file(ref)
        if err is not None or path is None:
            return None, err, 0

        size = path.stat().st_size
        ext = path.suffix.lower()
        mime_clean = (mime_type or "").strip().lower()
        is_text_override = mime_clean.startswith("text/") and mime_clean != "text/csv"
        can_auto_upload = not is_text_override and (bool(mime_clean) or ext not in _NON_UPLOADABLE_ANALYZE_EXTS)

        encoded_cost = ((size + 2) // 3) * 4 if can_auto_upload else size
        exceeds_single = size > CLI_INLINE_LIMIT_BYTES
        exceeds_cumulative = enforce_cumulative and (inline_budget_used + encoded_cost) > CLI_CUMULATIVE_INLINE_BYTES

        if can_auto_upload and (exceeds_single or exceeds_cumulative):
            uploaded = self.upload_file(str(path), mime_type=mime_type, wait=True)
            if not uploaded.ok:
                return None, uploaded, 0
            file_name = (uploaded.data or {}).get("name") if isinstance(uploaded.data, dict) else None
            if not isinstance(file_name, str) or not file_name.startswith("files/"):
                return (
                    None,
                    CLIResult(
                        ok=False,
                        exit_code=1,
                        error="gemini-api files upload succeeded but returned no files/<id> handle",
                        error_type="invalid_response",
                    ),
                    0,
                )
            return file_name, None, 0

        return str(path), None, encoded_cost

    def prepare_image_input(
        self,
        ref: str,
        *,
        image_loader: Callable[[str], Tuple[bytes, str]],
        local_mime_sniffer: Optional[Callable[[str], str]] = None,
        mime_to_suffix: Dict[str, str],
        tmp_files: List[Path],
        tmp_prefix: str = "hermes-image-ref-",
        inline_budget_used: int = 0,
        enforce_cumulative: bool = False,
    ) -> Tuple[Optional[str], Optional[CLIResult], int]:
        """Resolve a ``files/<id>``, remote URL, data URI, or local image path for ``gemini-api -i``.

        Remote URLs and data URIs are written to ``0o600`` temporary files appended to *tmp_files*
        for caller cleanup. Local image paths use *local_mime_sniffer* (when provided) so only the
        leading header bytes are read instead of buffering up to 25 MB in memory.
        """
        cleaned = (ref or "").strip()
        if cleaned.startswith("files/"):
            return self.prepare_media_input(
                cleaned,
                allow_youtube=False,
                inline_budget_used=inline_budget_used,
                enforce_cumulative=enforce_cumulative,
            )
        if cleaned.lower().startswith(("http://", "https://", "data:")):
            try:
                raw_bytes, mime = image_loader(cleaned)
            except Exception as exc:  # noqa: BLE001
                return (
                    None,
                    CLIResult(ok=False, exit_code=2, error=str(exc), error_type="invalid_argument"),
                    0,
                )
            suffix = mime_to_suffix.get(mime, ".png")
            with tempfile.NamedTemporaryFile(prefix=tmp_prefix, suffix=suffix, delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)
                tmp_file.write(raw_bytes)
            if os.name != "nt":
                tmp_path.chmod(0o600)
            tmp_files.append(tmp_path)
            return self.prepare_media_input(
                str(tmp_path),
                allow_youtube=False,
                mime_type=mime,
                inline_budget_used=inline_budget_used,
                enforce_cumulative=enforce_cumulative,
            )
        try:
            mime = local_mime_sniffer(cleaned) if local_mime_sniffer is not None else image_loader(cleaned)[1]
        except Exception as exc:  # noqa: BLE001
            return (
                None,
                CLIResult(ok=False, exit_code=2, error=str(exc), error_type="invalid_argument"),
                0,
            )
        return self.prepare_media_input(
            cleaned,
            allow_youtube=False,
            mime_type=mime,
            inline_budget_used=inline_budget_used,
            enforce_cumulative=enforce_cumulative,
        )
