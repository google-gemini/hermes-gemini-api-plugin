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

"""Tests for the gemini-api CLI integration (video provider, 6 tools, skill, security, and validation)."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest
import yaml

_PLUGIN_ROOT = Path(__file__).resolve().parent.parent

_PNG_HEX = (
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000d49444154789c6300010000000500010d0a2db40000000049454e44"
    "ae426082"
)


def _load_plugin_package():
    pkg_name = "hermes_plugins_test_pkg"
    spec = importlib.util.spec_from_file_location(
        pkg_name,
        _PLUGIN_ROOT / "__init__.py",
        submodule_search_locations=[str(_PLUGIN_ROOT)],
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = mod
    spec.loader.exec_module(mod)
    return mod


gemini_plugin = _load_plugin_package()
cli_tools = gemini_plugin._load_cli_tools_module()
cli_runner = cli_tools._cli_runner


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
    raw_stdout: str = "",
    raw_stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    stdout = raw_stdout if raw_stdout else (json.dumps(payload) if payload is not None else "")
    stderr = raw_stderr if raw_stderr else (json.dumps(stderr_payload) if stderr_payload is not None else "")
    return subprocess.CompletedProcess(args=["gemini-api"], returncode=returncode, stdout=stdout, stderr=stderr)


@pytest.fixture(autouse=True)
def _isolated_hermes_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cli_runner.clear_cli_cache()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("GEMINI_API_KEY", "AQ.test-key")
    monkeypatch.delenv("GEMINI_BASE_URL", raising=False)
    monkeypatch.delenv("GEMINI_IMAGE_MODEL", raising=False)
    yield tmp_path
    cli_runner.clear_cli_cache()


@pytest.fixture()
def fake_cli_bin(tmp_path: Path) -> Path:
    return _make_fake_binary(tmp_path)


@pytest.fixture()
def runner(fake_cli_bin: Path):
    return cli_runner.GeminiCLIRunner(
        endpoint_resolver=gemini_plugin._resolve_endpoint,
        user_agent=gemini_plugin.USER_AGENT,
        cli_path=str(fake_cli_bin),
    )


class TestBinaryResolutionAndInstaller:
    def test_resolves_hermes_home_bin_before_path_with_ttl_cache(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        hermes_bin = _make_fake_binary(tmp_path)
        calls = 0

        def _counting_which(_name: str):
            nonlocal calls
            calls += 1
            return "/usr/bin/other-gemini-api"

        monkeypatch.setattr(shutil, "which", _counting_which)
        assert cli_runner.resolve_cli_binary() == str(hermes_bin)
        assert calls == 0

    def test_settings_cli_path_override_and_missing_binary(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        custom_bin = tmp_path / "custom" / "gemini-api"
        custom_bin.parent.mkdir(parents=True)
        custom_bin.write_text("#!/bin/sh\n", encoding="utf-8")
        custom_bin.chmod(0o755)
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"plugins": {"entries": {"gemini-api": {"settings": {"cli_path": str(custom_bin)}}}}}),
            encoding="utf-8",
        )
        assert cli_runner.resolve_cli_binary() == str(custom_bin)

        # Non-existent configured cli_path fails closed instead of falling back silently.
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"plugins": {"entries": {"gemini-api": {"settings": {"cli_path": str(tmp_path / "nope")}}}}}),
            encoding="utf-8",
        )
        assert cli_runner.resolve_cli_binary() is None

        # Without binary anywhere and auto_install_cli default false, runner returns actionable hint.
        (tmp_path / "config.yaml").unlink()
        cli_runner.clear_cli_cache()
        monkeypatch.setattr(shutil, "which", lambda _name: None)
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "empty-home")
        bare_runner = cli_runner.GeminiCLIRunner(
            endpoint_resolver=gemini_plugin._resolve_endpoint,
            user_agent=gemini_plugin.USER_AGENT,
        )
        with patch.object(cli_runner, "_is_executable_file", return_value=False):
            res = bare_runner.run(["models", "list"])
        assert res.ok is False
        assert res.error_type == "missing_dependency"
        assert "go install github.com/google-gemini/gemini-api-cli" in res.error

    def test_server_flags_strip_v1beta_and_reject_cleartext_remote_urls(self, runner, monkeypatch: pytest.MonkeyPatch):
        assert cli_runner.server_flags_from_base_url("https://generativelanguage.googleapis.com/v1beta") == []
        assert cli_runner.server_flags_from_base_url("https://proxy.example.com/v1beta/") == [
            "--server-url",
            "https://proxy.example.com",
        ]
        assert cli_runner.server_flags_from_base_url("https://proxy.example.com/v1alpha") == [
            "--server-url",
            "https://proxy.example.com",
            "--api-version",
            "v1alpha",
        ]
        assert cli_runner.server_flags_from_base_url("http://127.0.0.1:8080/v1beta") == [
            "--server-url",
            "http://127.0.0.1:8080",
        ]

        # Cleartext non-loopback HTTP base_url is refused before spawning the CLI with GEMINI_API_KEY.
        monkeypatch.setenv("GEMINI_BASE_URL", "http://evil.example.com/v1beta")
        with patch("subprocess.run") as mock_run:
            res = runner.run(["models", "list"])
        assert res.ok is False
        assert res.error_type == "invalid_argument"
        assert "https://" in res.error
        mock_run.assert_not_called()

    def test_install_cli_binary_verifies_sha256_and_enforces_size_cap(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr("platform.machine", lambda: "x86_64")

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            payload = b"#!/bin/sh\necho gemini-api 0.4.4\n"
            info = tarfile.TarInfo(name="gemini-api")
            info.size = len(payload)
            info.mode = 0o755
            tf.addfile(info, io.BytesIO(payload))
        archive_bytes = buf.getvalue()
        good_sha = hashlib.sha256(archive_bytes).hexdigest()
        asset_name = f"gemini-api_{cli_runner.PINNED_CLI_VERSION}_linux_amd64.tar.gz"

        def _mock_client_ctx(checksums_body: str, chunks: List[bytes] | None = None):
            client = MagicMock()
            cs_resp = MagicMock(text=checksums_body)
            cs_resp.raise_for_status.return_value = None
            client.get.return_value = cs_resp

            stream_resp = MagicMock()
            stream_resp.raise_for_status.return_value = None
            stream_resp.iter_bytes.return_value = chunks if chunks is not None else [archive_bytes]
            stream_cm = MagicMock()
            stream_cm.__enter__.return_value = stream_resp
            stream_cm.__exit__.return_value = False
            client.stream.return_value = stream_cm

            cm = MagicMock()
            cm.__enter__.return_value = client
            cm.__exit__.return_value = False
            return cm

        with patch("tools.url_safety.create_ssrf_safe_client", return_value=_mock_client_ctx(f"{good_sha}  {asset_name}\n")):
            installed = cli_runner.install_cli_binary(target_dir=tmp_path / "installed")
        assert installed.is_file()
        assert os.access(installed, os.X_OK)

        bad_sha = "0" * 64
        with patch("tools.url_safety.create_ssrf_safe_client", return_value=_mock_client_ctx(f"{bad_sha}  {asset_name}\n")):
            with pytest.raises(ValueError, match="SHA-256 mismatch"):
                cli_runner.install_cli_binary(target_dir=tmp_path / "tampered")
        assert not (tmp_path / "tampered" / "gemini-api").exists()

        monkeypatch.setattr(cli_runner, "_MAX_ARCHIVE_BYTES", 8)
        with patch("tools.url_safety.create_ssrf_safe_client", return_value=_mock_client_ctx(f"{good_sha}  {asset_name}\n")):
            with pytest.raises(ValueError, match="safety limit"):
                cli_runner.install_cli_binary(target_dir=tmp_path / "oversized")


class TestRunnerExecutionAndSafety:
    def test_invokes_headless_json_with_pr2_user_agent_and_double_dash_guard(self, runner):
        from agent import aux_accounting

        recorded: List[Any] = []

        class _DB:
            def record_auxiliary_usage(self, *args, **kwargs):
                recorded.append((args, kwargs))

        proc = _completed_proc({
            "model": cli_tools.DEFAULT_ANALYZE_MODEL,
            "text": "Analysis complete",
            "usage": {"input_tokens": 120, "output_tokens": 80, "thought_tokens": 30, "total_tokens": 230},
        })
        token = aux_accounting.set_accounting_context(_DB(), "sess-cli-1")
        try:
            with patch("subprocess.run", return_value=proc) as mock_run:
                res = runner.run(
                    ["analyze", "-i", "files/abc123"],
                    positional_args=["--server-url=https://evil.example - summarize this"],
                    default_model=cli_tools.DEFAULT_ANALYZE_MODEL,
                )
        finally:
            aux_accounting.reset_accounting_context(token)

        assert res.ok is True
        cmd = mock_run.call_args.args[0]
        assert "--no-interactive" in cmd
        assert cmd[cmd.index("-o") + 1] == "json"
        header_val = cmd[cmd.index("--header") + 1]
        assert header_val == f"User-Agent: {gemini_plugin.USER_AGENT}"
        assert gemini_plugin.HERMES_VERSION in header_val
        assert f"gemini-api/{gemini_plugin.PLUGIN_VERSION}" in header_val
        # Positional prompt with leading `--` sits strictly after `--` so Cobra never parses it as a flag.
        dd_idx = cmd.index("--")
        assert dd_idx > cmd.index("--no-interactive")
        assert cmd[dd_idx + 1:] == ["--server-url=https://evil.example - summarize this"]
        assert mock_run.call_args.kwargs["stdin"] == subprocess.DEVNULL
        assert mock_run.call_args.kwargs["env"]["GEMINI_API_KEY"] == "AQ.test-key"
        assert len(recorded) == 1

    @pytest.mark.parametrize(
        ("code", "expected_type"),
        [(1, "api_error"), (2, "invalid_argument"), (3, "auth_required")],
    )
    def test_exit_codes_and_structured_stderr_hints(self, runner, code: int, expected_type: str):
        proc = _completed_proc(
            returncode=code,
            stderr_payload={"error": "bad request", "exit_code": code, "hints": ["try --stream=false"]},
        )
        with patch("subprocess.run", return_value=proc):
            res = runner.run(["generate"], positional_args=["hi"])
        assert res.ok is False
        assert res.error_type == expected_type
        assert "bad request" in res.error
        assert "try --stream=false" in res.error

    def test_blocks_credential_store_and_seamlessly_auto_uploads_large_binary_media(
        self, runner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        auth_json = tmp_path / "auth.json"
        auth_json.write_text('{"secret":"value"}', encoding="utf-8")

        with patch("subprocess.run") as mock_run:
            ref, err, _ = runner.prepare_media_input(str(auth_json))
        assert ref is None
        assert err is not None and err.error_type == "invalid_argument"
        assert "credential store" in err.error.lower()
        mock_run.assert_not_called()

        # Malformed files/<id> traversal attempt is rejected locally.
        bad_ref, bad_err, _ = runner.prepare_media_input("files/../../etc/passwd")
        assert bad_ref is None and bad_err is not None and bad_err.error_type == "invalid_argument"

        # Oversized binary media automatically promotes via `files upload --wait -- <path>`.
        monkeypatch.setattr(cli_runner, "CLI_INLINE_LIMIT_BYTES", 64)
        big_media = tmp_path / "lecture.mp4"
        big_media.write_bytes(b"\x00" * 256)

        upload_proc = _completed_proc({"name": "files/lec999", "state": "ACTIVE"})
        with patch("subprocess.run", return_value=upload_proc) as mock_run:
            ref, err, cost = runner.prepare_media_input(str(big_media))
        assert err is None
        assert ref == "files/lec999"
        assert cost == 0
        upload_cmd = mock_run.call_args.args[0]
        assert upload_cmd[1:3] == ["files", "upload"]
        assert "--wait" in upload_cmd
        assert upload_cmd[upload_cmd.index("--") + 1:] == [str(big_media)]

        # Plain-text files are NOT uploaded to Files API (gemini-api sends text inline only).
        big_txt = tmp_path / "notes.txt"
        big_txt.write_text("x" * 256, encoding="utf-8")
        with patch("subprocess.run") as mock_run:
            txt_ref, txt_err, _ = runner.prepare_media_input(str(big_txt))
        assert txt_err is None
        assert txt_ref == str(big_txt)
        mock_run.assert_not_called()


class TestVideoGenProviderAndTools:
    def test_video_provider_omits_default_m_flag_and_supports_explicit_override(self, runner, tmp_path: Path):
        provider = cli_tools.GeminiVideoGenProvider(
            runner=runner,
            endpoint_resolver=gemini_plugin._resolve_endpoint,
            image_loader=gemini_plugin._load_image_bytes,
        )
        assert provider.default_model() == cli_tools.DEFAULT_VIDEO_MODEL
        ids = {m["id"] for m in provider.list_models()}
        assert ids >= {cli_tools.DEFAULT_VIDEO_MODEL, "gemini-omni-flash-preview"}

        out_mp4 = tmp_path / "cache" / "videos" / "clip.mp4"
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        out_mp4.write_bytes(b"fake-mp4-bytes")

        # Foreign top-level model (`grok-imagine-video`) is ignored; `-m` is omitted so CLI owns default!
        proc = _completed_proc({
            "path": str(out_mp4),
            "kind": "video",
            "mime_type": "video/mp4",
            "size_bytes": 14,
            "response_id": "int_vid_1",
            "status": "completed",
        })
        with patch("subprocess.run", return_value=proc) as mock_run:
            result = provider.generate("a paper boat in the rain", model="grok-imagine-video")
        assert result["success"] is True
        assert result["video"] == str(out_mp4)
        assert result["model"] == cli_tools.DEFAULT_VIDEO_MODEL
        assert result["modality"] == "text"
        assert result["response_id"] == "int_vid_1"
        cmd = mock_run.call_args.args[0]
        assert cmd[1] == "video"
        assert "-m" not in cmd
        assert cmd[cmd.index("--") + 1:] == ["a paper boat in the rain"]

        # Explicit Gemini video model passes `-m` and image-to-video passes base64 image + text via `--body`.
        frame = tmp_path / "frame.png"
        frame.write_bytes(bytes.fromhex(_PNG_HEX))
        with patch("subprocess.run", return_value=proc) as mock_run:
            i2v = provider.generate("animate gentle waves", image_url=str(frame), model="gemini-omni-flash-preview")
        assert i2v["success"] is True
        assert i2v["modality"] == "image"
        assert i2v["model"] == "gemini-omni-flash-preview"
        i2v_cmd = mock_run.call_args.args[0]
        assert i2v_cmd[i2v_cmd.index("-m") + 1] == "gemini-omni-flash-preview"
        body = json.loads(i2v_cmd[i2v_cmd.index("--body") + 1])
        assert body["input"][0]["type"] == "image"
        assert body["input"][0]["mime_type"] == "image/png"
        assert body["input"][1] == {"type": "text", "text": "animate gentle waves"}

    def test_music_and_tts_tools_delegate_default_model_and_pass_overrides(self, runner, tmp_path: Path):
        handlers = cli_tools.build_tool_handlers(runner)

        music_file = tmp_path / "track.mp3"
        music_file.write_bytes(b"mp3")
        with patch(
            "subprocess.run",
            return_value=_completed_proc({"path": str(music_file), "mime_type": "audio/mpeg", "size_bytes": 3}),
        ) as mock_run:
            music_res = json.loads(handlers["gemini_music"]({"prompt": "- upbeat jazz piano"}))
        assert music_res["success"] is True
        assert music_res["audio"] == str(music_file)
        assert music_res["model"] == cli_tools.DEFAULT_MUSIC_MODEL
        music_cmd = mock_run.call_args.args[0]
        assert "-m" not in music_cmd
        assert music_cmd[music_cmd.index("--") + 1:] == ["- upbeat jazz piano"]

        # Mutually exclusive voice + multi_speaker rejected locally.
        conflict = json.loads(
            handlers["gemini_tts"]({"text": "A: Hi", "voice": "Kore", "multi_speaker": "A=Kore,B=Puck"})
        )
        assert conflict["error_type"] == "invalid_argument"

        wav_file = tmp_path / "speech.wav"
        wav_file.write_bytes(b"wav")
        with patch(
            "subprocess.run",
            return_value=_completed_proc(
                {"path": str(wav_file), "model": cli_tools.DEFAULT_TTS_MODEL, "sample_rate": 24000, "channels": 1}
            ),
        ) as mock_run:
            tts_res = json.loads(
                handlers["gemini_tts"]({
                    "text": "Host: Welcome!\nGuest: Thanks!",
                    "multi_speaker": "Host=Kore,Guest=Puck",
                    "language": "en-US",
                })
            )
        assert tts_res["success"] is True
        assert tts_res["audio"] == str(wav_file)
        assert tts_res["model"] == cli_tools.DEFAULT_TTS_MODEL
        tts_cmd = mock_run.call_args.args[0]
        assert "-m" not in tts_cmd
        assert "--multi-speaker" in tts_cmd
        assert "Host=Kore,Guest=Puck" in tts_cmd
        assert "--language" in tts_cmd

    def test_analyze_transcribe_deep_research_and_files_upload(
        self, runner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        handlers = cli_tools.build_tool_handlers(runner)

        # Cumulative multi-file inline budget promotes the second PDF seamlessly and surfaces files/<id>.
        monkeypatch.setattr(cli_runner, "CLI_CUMULATIVE_INLINE_BYTES", 32)
        doc1 = tmp_path / "part1.pdf"
        doc2 = tmp_path / "part2.pdf"
        doc1.write_bytes(b"%PDF-1.4 one")
        doc2.write_bytes(b"%PDF-1.4 two-overflow")

        upload_proc = _completed_proc({"name": "files/part2id", "state": "ACTIVE"})
        analyze_proc = _completed_proc({"model": cli_tools.DEFAULT_ANALYZE_MODEL, "text": "Key takeaway: 42%"})
        with patch("subprocess.run", side_effect=[upload_proc, analyze_proc]) as mock_run:
            ana = json.loads(
                handlers["gemini_analyze"]({
                    "inputs": [str(doc1), str(doc2), "https://youtu.be/dQw4w9WgXcQ"],
                    "question": "Compare the reports and video",
                    "system": "Be concise",
                })
            )
        assert ana["success"] is True
        assert ana["text"] == "Key takeaway: 42%"
        assert ana["inputs"] == [str(doc1), "files/part2id", "https://youtu.be/dQw4w9WgXcQ"]
        ana_cmd = mock_run.call_args_list[1].args[0]
        assert ana_cmd[1] == "analyze"
        assert "-m" not in ana_cmd
        assert "files/part2id" in ana_cmd
        assert ana_cmd[ana_cmd.index("--") + 1:] == ["Compare the reports and video"]

        # Transcribe reads back generated SRT file content when <=32KB.
        audio = tmp_path / "clip.mp3"
        audio.write_bytes(b"audio-bytes")
        srt_out = tmp_path / "clip.srt"
        srt_out.write_text("1\n00:00:00,000 --> 00:00:02,000\nHello world\n", encoding="utf-8")
        with patch(
            "subprocess.run",
            return_value=_completed_proc(
                {"format": "srt", "model": cli_tools.DEFAULT_ANALYZE_MODEL, "results": [{"input": str(audio), "path": str(srt_out)}]}
            ),
        ):
            tr = json.loads(handlers["gemini_transcribe"]({"inputs": [str(audio)], "format": "srt"}))
        assert tr["success"] is True
        assert "Hello world" in tr["results"][0]["content"]

        # Deep research defaults to background=True on action="run", with explicit background=False supported.
        with patch(
            "subprocess.run",
            return_value=_completed_proc({"id": "int_res_42", "status": "in_progress"}),
        ) as mock_run:
            bg = json.loads(handlers["gemini_deep_research"]({"action": "run", "prompt": "Survey solid-state batteries"}))
        assert bg["interaction_id"] == "int_res_42"
        assert bg["background"] is True
        bg_cmd = mock_run.call_args.args[0]
        assert "--background" in bg_cmd and "--stream=false" in bg_cmd
        assert bg_cmd[bg_cmd.index("--agent") + 1] == cli_tools.DEFAULT_RESEARCH_AGENT

        with patch(
            "subprocess.run",
            return_value=_completed_proc(
                {"id": "int_res_42", "status": "completed", "steps": [{"content": [{"type": "text", "text": "Final synthesis"}]}]}
            ),
        ) as mock_run:
            fg = json.loads(
                handlers["gemini_deep_research"]({"action": "run", "prompt": "Quick check", "background": False})
            )
        assert fg["background"] is False
        assert "--background" not in mock_run.call_args.args[0]
        assert fg["output_text"] == "Final synthesis"

        with patch(
            "subprocess.run",
            return_value=_completed_proc(
                {"id": "int_res_42", "status": "completed", "steps": [{"content": [{"type": "text", "text": "Final synthesis"}]}]}
            ),
        ):
            st = json.loads(handlers["gemini_deep_research"]({"action": "status", "interaction_id": "int_res_42"}))
        assert st["status"] == "completed"
        assert st["output_text"] == "Final synthesis"

        with patch("subprocess.run", return_value=_completed_proc({"id": "int_res_42", "status": "cancelled"})):
            cn = json.loads(handlers["gemini_deep_research"]({"action": "cancel", "interaction_id": "int_res_42"}))
        assert cn["status"] == "cancelled"

        # Explicit files_upload tool returns reusable files/<id> handle.
        sample = tmp_path / "deck.pdf"
        sample.write_bytes(b"%PDF-1.4 deck")
        with patch(
            "subprocess.run",
            return_value=_completed_proc({"name": "files/deck77", "uri": "https://x/files/deck77", "state": "ACTIVE"}),
        ):
            up = json.loads(handlers["gemini_files_upload"]({"path": str(sample), "display_name": "Q4 Deck"}))
        assert up["success"] is True
        assert up["name"] == "files/deck77"


class TestPluginManifestAndValidatorParity:
    def test_plugin_registers_video_provider_tools_and_skill(self, tmp_path: Path, fake_cli_bin: Path):
        from agent.video_gen_registry import get_provider as get_video_provider
        from hermes_cli.plugins import get_plugin_manager
        from tools.registry import registry

        user_plugin_dir = tmp_path / "plugins" / "gemini-api"
        user_plugin_dir.mkdir(parents=True)
        for fname in ("plugin.yaml", "__init__.py", "cli_runner.py", "cli_tools.py", "SKILL.md"):
            shutil.copy2(_PLUGIN_ROOT / fname, user_plugin_dir / fname)
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({
                "plugins": {
                    "enabled": ["gemini-api"],
                    "entries": {"gemini-api": {"settings": {"cli_path": str(fake_cli_bin)}}},
                }
            }),
            encoding="utf-8",
        )

        mgr = get_plugin_manager()
        mgr.discover_and_load()
        assert "gemini-api" in mgr._plugins
        assert mgr._plugins["gemini-api"].enabled is True

        vid_provider = get_video_provider("gemini")
        assert vid_provider is not None
        assert vid_provider.is_available() is True

        registered_names = set(registry.get_all_tool_names())
        for tool_name in cli_tools.PROVIDED_TOOL_NAMES:
            assert tool_name in registered_names

        assert mgr.list_plugin_skills("gemini-api") == ["gemini-api-cli"]
        skill_path = mgr.find_plugin_skill("gemini-api:gemini-api-cli")
        assert skill_path is not None and skill_path.exists()

    def test_hermes_plugin_validator_and_security_scanner_pass(self):
        from hermes_cli.plugin_validate import validate_plugin_dir
        from tools.plugin_guard import scan_plugin

        report = validate_plugin_dir(_PLUGIN_ROOT)
        assert report.ok, f"Validation failed: {report.failures}"
        assert not report.warnings, f"Unexpected validation warnings: {report.warnings}"

        scan = scan_plugin(_PLUGIN_ROOT, source="unit-test")
        assert scan.verdict == "safe", f"Unexpected findings: {scan.findings}"
