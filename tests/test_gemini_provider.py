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

"""Tests for the Google AI Studio (Gemini / Nano Banana) image_gen provider backed by ``gemini-api image``."""

from __future__ import annotations

import base64
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest
import yaml

_PLUGIN_ROOT = Path(__file__).resolve().parent.parent
_INIT_PATH = _PLUGIN_ROOT / "__init__.py"
_SPEC = importlib.util.spec_from_file_location(
    "hermes_plugins.gemini_api_img_test",
    _INIT_PATH,
    submodule_search_locations=[str(_PLUGIN_ROOT)],
)
assert _SPEC is not None and _SPEC.loader is not None
gemini_plugin = importlib.util.module_from_spec(_SPEC)
sys.modules["hermes_plugins.gemini_api_img_test"] = gemini_plugin
_SPEC.loader.exec_module(gemini_plugin)
cli_runner = gemini_plugin._load_sibling_module("cli_runner")

# 1×1 transparent PNG — valid image bytes for magic-byte sniffing
_PNG_HEX = (
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000d49444154789c6300010000000500010d0a2db40000000049454e44"
    "ae426082"
)


def _b64_png() -> str:
    return base64.b64encode(bytes.fromhex(_PNG_HEX)).decode("ascii")


def _make_fake_binary(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    binary = bin_dir / "gemini-api"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)
    return binary


def _completed_proc(
    payload: Any = None,
    *,
    returncode: int = 0,
    stderr_payload: Any = None,
) -> subprocess.CompletedProcess[str]:
    stdout = json.dumps(payload) if payload is not None else ""
    stderr = json.dumps(stderr_payload) if stderr_payload is not None else ""
    return subprocess.CompletedProcess(args=["gemini-api"], returncode=returncode, stdout=stdout, stderr=stderr)


def _write_fake_output(tmp_path: Path, name: str = "gemini-image-1.png") -> Path:
    out_file = tmp_path / "cache" / "generated" / "images" / name
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_bytes(bytes.fromhex(_PNG_HEX))
    return out_file


@pytest.fixture(autouse=True)
def _tmp_hermes_home(tmp_path, monkeypatch):
    cli_runner.clear_cli_cache()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for var in (
        "GEMINI_API_KEY", "GEMINI_IMAGE_MODEL", "GEMINI_BASE_URL",
        "GEMINI_IMAGE_ASPECT_RATIO", "GEMINI_IMAGE_SIZE", "GEMINI_IMAGE_GOOGLE_SEARCH",
    ):
        monkeypatch.delenv(var, raising=False)
    yield tmp_path
    cli_runner.clear_cli_cache()


@pytest.fixture
def fake_cli_bin(tmp_path):
    return _make_fake_binary(tmp_path)


@pytest.fixture
def provider(monkeypatch, fake_cli_bin):
    monkeypatch.setenv("GEMINI_API_KEY", "AQ.test-key")
    runner = cli_runner.GeminiCLIRunner(
        endpoint_resolver=gemini_plugin._resolve_endpoint,
        user_agent=gemini_plugin.CLI_USER_AGENT,
        cli_path=str(fake_cli_bin),
    )
    return gemini_plugin.GeminiImageGenProvider(runner=runner)


# ── Metadata & Setup Schema ─────────────────────────────────────────────────


class TestMetadata:
    def test_name(self, provider):
        assert provider.name == "gemini"

    def test_default_model(self, provider):
        assert provider.default_model() == gemini_plugin.DEFAULT_MODEL
        assert provider.default_model().startswith("gemini-")

    def test_picker_matches_resolvable_catalog(self, provider):
        ids = [m["id"] for m in provider.list_models()]
        assert set(ids) == set(provider.models)
        assert set(ids) >= {
            "gemini-nano-banana-2.1",
            "gemini-3.1-flash-image",
            "gemini-3.1-flash-lite-image",
            "gemini-3-pro-image",
        }
        assert provider.default_model() in ids

    def test_catalog_entries_have_display_speed_strengths(self, provider):
        for entry in provider.list_models():
            assert "Nano Banana" in entry["display"]
            assert entry["speed"]
            assert entry["strengths"]

    def test_setup_schema_uses_paid_badge_and_single_gemini_api_key(self, provider, monkeypatch):
        from hermes_cli.tools_config import _provider_env_ready

        schema = provider.get_setup_schema()
        assert schema["badge"] == "paid"
        assert [v["key"] for v in schema["env_vars"]] == ["GEMINI_API_KEY"]

        monkeypatch.setenv("GEMINI_API_KEY", "AQ.gemini-only")
        assert _provider_env_ready(schema) is True


# ── Availability ────────────────────────────────────────────────────────────


class TestAvailability:
    def test_no_api_key_unavailable(self, monkeypatch, fake_cli_bin):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        assert gemini_plugin.GeminiImageGenProvider().is_available() is False

    def test_api_key_and_binary_required_for_availability(self, monkeypatch, tmp_path):
        monkeypatch.setenv("GEMINI_API_KEY", "AQ.test")
        monkeypatch.setattr(shutil, "which", lambda _name: None)
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "empty-home")
        with patch.object(cli_runner, "_is_executable_file", return_value=False):
            assert gemini_plugin.GeminiImageGenProvider().is_available() is False

        _make_fake_binary(tmp_path)
        cli_runner.clear_cli_cache()
        assert gemini_plugin.GeminiImageGenProvider().is_available() is True


# ── Model resolution ────────────────────────────────────────────────────────


class TestModelResolution:
    def test_env_var_override_and_google_prefix_stripping(self, monkeypatch):
        monkeypatch.setenv("GEMINI_IMAGE_MODEL", "google/gemini-3-pro-image")
        model_id, meta = gemini_plugin._resolve_model()
        assert model_id == "gemini-3-pro-image"
        assert meta["api_model"] == "gemini-3-pro-image"

        assert gemini_plugin._strip_google_prefix("--evil-flag") is None
        assert gemini_plugin._strip_google_prefix("google/--evil-flag") is None
        assert gemini_plugin._strip_google_prefix("google/") is None

    def test_config_gemini_model_and_foreign_top_level_ignored(self, tmp_path):
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"image_gen": {"model": "gpt-image-2-medium", "gemini": {"model": "gemini-3.1-flash-lite-image"}}})
        )
        model_id, _ = gemini_plugin._resolve_model()
        assert model_id == "gemini-3.1-flash-lite-image"

        (tmp_path / "config.yaml").write_text(yaml.safe_dump({"image_gen": {"model": "gpt-image-2-medium"}}))
        model_id, _ = gemini_plugin._resolve_model()
        assert model_id == gemini_plugin.DEFAULT_MODEL


# ── Endpoint / credential routing ───────────────────────────────────────────


class TestEndpointConfig:
    def test_config_base_url_and_key_env_reach_availability_and_endpoint(self, monkeypatch, tmp_path, fake_cli_bin):
        monkeypatch.setenv("CUSTOM_GEMINI_TOKEN", "custom-secret")
        (tmp_path / "config.yaml").write_text(yaml.safe_dump({"image_gen": {"gemini": {
            "base_url": "https://proxy.example.com/v1beta/", "key_env": "CUSTOM_GEMINI_TOKEN"}}}))
        assert gemini_plugin.GeminiImageGenProvider().is_available() is True
        assert gemini_plugin._resolve_endpoint() == ("https://proxy.example.com/v1beta", "custom-secret")


# ── Source image loading ────────────────────────────────────────────────────


class TestSourceImageLoading:
    def test_load_image_bytes_blocks_credential_store_and_symlinks(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        auth_json = hermes_home / "auth.json"
        auth_json.write_text('{"api_key":"sk-secret"}', encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        with pytest.raises(ValueError, match="credential store"):
            gemini_plugin._load_image_bytes(str(auth_json))

        # A symlink pointing to a blocked credential file must also be rejected after resolution.
        symlink_path = tmp_path / "innocent.png"
        symlink_path.symlink_to(auth_json)
        with pytest.raises(ValueError, match="credential store"):
            gemini_plugin._load_image_bytes(str(symlink_path))
        with pytest.raises(ValueError, match="credential store"):
            gemini_plugin._sniff_local_image_mime(str(symlink_path))

    def test_load_image_bytes_allows_legit_local_image(self, tmp_path):
        img = tmp_path / "pic.png"
        img.write_bytes(bytes.fromhex(_PNG_HEX))

        data, mime = gemini_plugin._load_image_bytes(str(img))
        assert data == bytes.fromhex(_PNG_HEX)
        assert mime == "image/png"

    def test_load_image_bytes_decodes_whitespace_wrapped_data_uri_and_rejects_corrupt_b64(self):
        raw_b64 = _b64_png()
        wrapped_b64 = f"{raw_b64[:24]}\r\n  {raw_b64[24:48]}\n\t{raw_b64[48:]}"
        data, mime = gemini_plugin._load_image_bytes(f"data:image/png;base64,{wrapped_b64}")
        assert data == bytes.fromhex(_PNG_HEX)
        assert mime == "image/png"

        with pytest.raises(Exception):
            gemini_plugin._load_image_bytes("data:image/png;base64,!!!not-valid-base64!!!")


# ── Generate (via gemini-api image CLI) ─────────────────────────────────────


class TestGenerate:
    def test_empty_prompt_rejected(self, provider):
        result = provider.generate("   ", aspect_ratio="square")
        assert result["success"] is False
        assert result["error_type"] == "invalid_argument"

    def test_missing_api_key(self, monkeypatch, fake_cli_bin):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        result = gemini_plugin.GeminiImageGenProvider().generate("a yellow banana")
        assert result["success"] is False
        assert result["error_type"] == "auth_required"

    def test_cli_image_saves_to_cache_and_omits_default_m_flag(self, provider, tmp_path):
        out_img = _write_fake_output(tmp_path, "gemini-image-out.png")
        proc = _completed_proc({
            "path": str(out_img),
            "kind": "image",
            "mime_type": "image/png",
            "size_bytes": out_img.stat().st_size,
            "response_id": "int_img_1",
            "status": "completed",
        })

        with patch("subprocess.run", return_value=proc) as mock_run:
            result = provider.generate("a cute nano banana", aspect_ratio="landscape")

        assert result["success"] is True
        assert result["image"] == str(out_img)
        assert result["model"] == gemini_plugin.DEFAULT_MODEL
        assert result["aspect_ratio"] == "landscape"
        assert result["exact_aspect_ratio"] == "16:9"
        assert result["provider"] == "gemini"
        assert result["modality"] == "text"
        assert result["response_id"] == "int_img_1"

        cmd = mock_run.call_args.args[0]
        assert cmd[1] == "image"
        assert "-m" not in cmd
        assert cmd[cmd.index("--out") + 1] == str(tmp_path / "cache" / "generated" / "images")
        assert cmd[cmd.index("--aspect-ratio") + 1] == "16:9"
        assert cmd[cmd.index("--") + 1:] == ["a cute nano banana"]
        header_val = cmd[cmd.index("--header") + 1]
        assert header_val == f"User-Agent: {gemini_plugin.CLI_USER_AGENT}"
        assert mock_run.call_args.kwargs["env"]["GEMINI_API_KEY"] == "AQ.test-key"

    def test_token_usage_reaches_session_accounting(self, provider, tmp_path):
        from agent import aux_accounting

        recorded: List[Any] = []

        class _DB:
            def record_auxiliary_usage(self, *args, **kwargs):
                recorded.append((args, kwargs))

        out_img = _write_fake_output(tmp_path)
        proc = _completed_proc({
            "path": str(out_img),
            "kind": "image",
            "mime_type": "image/png",
            "usage": {
                "total_input_tokens": 42,
                "total_output_tokens": 1290,
                "total_thought_tokens": 668,
                "total_tokens": 2000,
            },
        })
        token = aux_accounting.set_accounting_context(_DB(), "sess-gemini-img-1")
        try:
            with patch("subprocess.run", return_value=proc):
                result = provider.generate("a yellow banana", aspect_ratio="landscape")
        finally:
            aux_accounting.reset_accounting_context(token)

        assert result["success"] is True
        ((session_id, task), kwargs), = recorded
        assert (session_id, task) == ("sess-gemini-img-1", "image_generation")
        assert (kwargs["model"], kwargs["billing_provider"]) == (gemini_plugin.DEFAULT_MODEL, "gemini")
        assert kwargs["input_tokens"] == 42
        assert kwargs["output_tokens"] == 1290 + 668

    @pytest.mark.parametrize(
        "model_id,expected_upscale",
        [
            ("gemini-nano-banana-2.1", True),
            ("gemini-3.1-flash-image", True),
            ("gemini-3.1-flash-lite-image", False),
            ("gemini-3-pro-image", True),
        ],
    )
    def test_upscale_advertised_only_when_model_has_a_rung_above_1k(
        self, provider, monkeypatch, model_id, expected_upscale
    ):
        monkeypatch.setenv("GEMINI_IMAGE_MODEL", model_id)
        assert provider.capabilities()["supports_upscale"] is expected_upscale

    def test_config_exact_aspect_ratio_image_size_and_google_search_override_defaults(
        self, provider, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("GEMINI_IMAGE_ASPECT_RATIO", "4:5")
        monkeypatch.setenv("GEMINI_IMAGE_SIZE", "512")
        monkeypatch.setenv("GEMINI_IMAGE_GOOGLE_SEARCH", "false")
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({
                "image_gen": {
                    "gemini": {
                        "aspect_ratio": "21:9",
                        "image_size": "2K",
                        "google_search": True,
                    }
                }
            })
        )
        out_img = _write_fake_output(tmp_path)
        proc = _completed_proc({"path": str(out_img), "mime_type": "image/png"})

        with patch("subprocess.run", return_value=proc) as mock_run:
            result = provider.generate("ultrawide banner", aspect_ratio="landscape")

        assert result["success"] is True
        assert result["exact_aspect_ratio"] == "21:9"
        assert result["image_size"] == "2K"
        assert result["google_search"] is True
        cmd = mock_run.call_args.args[0]
        assert cmd[cmd.index("--aspect-ratio") + 1] == "21:9"
        assert cmd[cmd.index("--image-size") + 1] == "2K"
        assert json.loads(cmd[cmd.index("--body") + 1]) == {"tools": [{"type": "google_search"}]}

    def test_reference_images_passed_via_i_flag_and_data_uri_cleaned_up(self, provider, tmp_path):
        ref_path = tmp_path / "ref.png"
        ref_path.write_bytes(bytes.fromhex(_PNG_HEX))
        data_uri = f"data:image/png;base64,{_b64_png()}"
        out_img = _write_fake_output(tmp_path)
        proc = _completed_proc({"path": str(out_img), "mime_type": "image/png"})

        seen_inputs: List[str] = []

        def _capture(argv, **_kwargs):
            for idx, tok in enumerate(argv):
                if tok == "-i":
                    seen_inputs.append(argv[idx + 1])
            assert Path(seen_inputs[1]).is_file()
            return proc

        with patch("subprocess.run", side_effect=_capture):
            result = provider.generate(
                "make it cyberpunk",
                image_url=str(ref_path),
                reference_image_urls=[data_uri, "files/ref42"],
            )

        assert result["success"] is True
        assert result["modality"] == "image"
        assert seen_inputs[0] == str(ref_path)
        assert seen_inputs[2] == "files/ref42"
        assert not Path(seen_inputs[1]).exists()  # temp file cleaned up after CLI exit

    def test_caller_model_honoured_only_when_it_names_a_catalog_entry(self, provider, tmp_path):
        out_img = _write_fake_output(tmp_path)
        proc = _completed_proc({"path": str(out_img), "mime_type": "image/png"})

        with patch("subprocess.run", return_value=proc) as mock_run:
            result = provider.generate("a yellow banana", model="gemini-3-pro-image")
        assert result["model"] == "gemini-3-pro-image"
        cmd = mock_run.call_args.args[0]
        assert cmd[cmd.index("-m") + 1] == "gemini-3-pro-image"

        with patch("subprocess.run", return_value=proc) as mock_run:
            result = provider.generate("a yellow banana", model="gpt-image-2-medium")
        assert result["success"] is True
        assert result["model"] == gemini_plugin.DEFAULT_MODEL
        assert "-m" not in mock_run.call_args.args[0]

    def test_provider_scoped_custom_model_passes_through_conservatively(self, provider, tmp_path):
        (tmp_path / "config.yaml").write_text(yaml.safe_dump({
            "image_gen": {"gemini": {
                "model": "gemini-unlisted-test-image", "image_size": "4K", "google_search": True}}}))
        out_img = _write_fake_output(tmp_path)
        proc = _completed_proc({"path": str(out_img), "mime_type": "image/png"})

        with patch("subprocess.run", return_value=proc) as mock_run:
            result = provider.generate("a yellow banana")

        assert result["success"] is True
        assert result["model"] == "gemini-unlisted-test-image"
        cmd = mock_run.call_args.args[0]
        assert cmd[cmd.index("-m") + 1] == "gemini-unlisted-test-image"
        assert "--image-size" not in cmd
        assert "--body" not in cmd

    @pytest.mark.parametrize(
        "model_id,expected_wire_ratio,expected_search,expected_512_size",
        [
            ("gemini-nano-banana-2.1", "1:4", True, None),
            ("gemini-3.1-flash-image", "1:4", True, "512"),
            ("gemini-3.1-flash-lite-image", "16:9", False, None),
            ("gemini-3-pro-image", "16:9", True, None),
        ],
    )
    def test_per_model_gating_for_extreme_aspect_ratios_and_google_search(
        self, provider, tmp_path, model_id, expected_wire_ratio, expected_search, expected_512_size
    ):
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"image_gen": {"gemini": {"google_search": True, "image_size": "512"}}})
        )
        out_img = _write_fake_output(tmp_path)
        proc = _completed_proc({"path": str(out_img), "mime_type": "image/png"})

        with patch("subprocess.run", return_value=proc) as mock_run:
            result = provider.generate("panoramic poster", aspect_ratio="1:4", model=model_id)

        assert result["success"] is True
        assert result["exact_aspect_ratio"] == expected_wire_ratio
        assert result.get("google_search", False) is expected_search
        assert result.get("image_size") == expected_512_size
        cmd = mock_run.call_args.args[0]
        assert cmd[cmd.index("--aspect-ratio") + 1] == expected_wire_ratio
        if expected_512_size:
            assert cmd[cmd.index("--image-size") + 1] == expected_512_size
        else:
            assert "--image-size" not in cmd
        assert ("--body" in cmd) is expected_search

    def test_reference_images_over_inline_budget_auto_upload_via_files_api(
        self, provider, monkeypatch, tmp_path
    ):
        """Oversized reference image batches seamlessly auto-upload through Gemini Files API."""
        monkeypatch.setattr(cli_runner, "CLI_CUMULATIVE_INLINE_BYTES", 128)
        ref1 = tmp_path / "r1.png"
        ref2 = tmp_path / "r2.png"
        ref1.write_bytes(bytes.fromhex(_PNG_HEX))
        ref2.write_bytes(bytes.fromhex(_PNG_HEX) + b"\x00" * 128)
        out_img = _write_fake_output(tmp_path)

        upload_proc = _completed_proc({"name": "files/imgref2", "state": "ACTIVE"})
        image_proc = _completed_proc({"path": str(out_img), "mime_type": "image/png"})
        with patch("subprocess.run", side_effect=[upload_proc, image_proc]) as mock_run:
            result = provider.generate("combine these", reference_image_urls=[str(ref1), str(ref2)])

        assert result["success"] is True
        assert mock_run.call_args_list[0].args[0][1:3] == ["files", "upload"]
        img_cmd = mock_run.call_args_list[1].args[0]
        assert "files/imgref2" in img_cmd

    def test_non_image_local_file_rejected_before_cli_call(self, provider, tmp_path):
        decoy = tmp_path / "notes.png"
        decoy.write_text("this is not an image")

        with patch("subprocess.run") as mock_run:
            result = provider.generate("edit this", image_url=str(decoy))

        assert result["success"] is False
        assert "not a recognised image" in result["error"]
        mock_run.assert_not_called()

    def test_cli_auth_error_and_missing_output_file_surfaced(self, provider, tmp_path):
        err_proc = _completed_proc(
            returncode=3,
            stderr_payload={"error": "API key not valid", "exit_code": 3},
        )
        with patch("subprocess.run", return_value=err_proc):
            result = provider.generate("a cat")

        assert result["success"] is False
        assert result["error_type"] == "auth_required"
        assert "API key not valid" in result["error"]

        missing_proc = _completed_proc({"path": str(tmp_path / "does-not-exist.png")})
        with patch("subprocess.run", return_value=missing_proc):
            missing = provider.generate("a cat")
        assert missing["success"] is False
        assert missing["error_type"] == "invalid_response"

    def test_cleartext_remote_base_url_rejected_before_cli_call(self, provider, monkeypatch):
        monkeypatch.setenv("GEMINI_BASE_URL", "http://evil.example.com/v1beta")
        with patch("subprocess.run") as mock_run:
            result = provider.generate("a cat")

        assert result["success"] is False
        assert result["error_type"] == "invalid_argument"
        assert "https://" in result["error"]
        mock_run.assert_not_called()


# ── Plugin registration ─────────────────────────────────────────────────────


class TestPluginRegistration:
    def test_user_installed_plugin_registers_in_image_gen_registry(self, tmp_path):
        from agent.image_gen_registry import get_provider
        from hermes_cli.plugins import get_plugin_manager

        user_plugin_dir = tmp_path / "plugins" / "gemini-api"
        user_plugin_dir.mkdir(parents=True)
        for fname in ("plugin.yaml", "__init__.py", "cli_runner.py", "cli_tools.py", "SKILL.md"):
            src = _PLUGIN_ROOT / fname
            if src.is_file():
                shutil.copy2(src, user_plugin_dir / fname)
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"plugins": {"enabled": ["gemini-api"]}})
        )

        mgr = get_plugin_manager()
        mgr.discover_and_load()
        assert "gemini-api" in mgr._plugins
        assert mgr._plugins["gemini-api"].enabled is True
        registered = get_provider("gemini")
        assert registered is not None
        assert registered.name == "gemini"
