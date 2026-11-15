"""The Claude Code plugin and its marketplace point at the packaged skills, in sync."""

import json
import struct
from pathlib import Path

from researchforge import __version__
from researchforge.claude.installer import list_packaged_skills

REPO_ROOT = Path(__file__).resolve().parents[2]
MARKETPLACE = REPO_ROOT / ".claude-plugin" / "marketplace.json"


def _marketplace() -> dict[str, object]:
    data: dict[str, object] = json.loads(MARKETPLACE.read_text())
    return data


def _plugin_root() -> Path:
    plugins = _marketplace()["plugins"]
    assert isinstance(plugins, list) and len(plugins) == 1
    source = plugins[0]["source"]
    assert source.startswith("./") and ".." not in source, source
    return REPO_ROOT / str(source)


class TestClaudePlugin:
    def test_marketplace_names_one_plugin_called_researchforge(self) -> None:
        data = _marketplace()
        assert data["name"] == "researchforge"
        assert isinstance(data["owner"], dict) and data["owner"]["name"]
        plugins = data["plugins"]
        assert isinstance(plugins, list)
        assert [p["name"] for p in plugins] == ["researchforge"]

    def test_plugin_source_is_the_packaged_skills_folder(self) -> None:
        """One copy of the skills: the plugin loads the same files pip installs."""
        root = _plugin_root()
        packaged = Path(__file__).resolve().parents[2] / "src" / "researchforge" / "claude"
        assert root.resolve() == packaged.resolve()
        on_disk = {p.parent.name for p in (root / "skills").glob("*/SKILL.md")}
        assert on_disk == set(list_packaged_skills())

    def test_manifest_version_matches_the_package(self) -> None:
        manifest = json.loads((_plugin_root() / ".claude-plugin" / "plugin.json").read_text())
        assert manifest["name"] == "researchforge"
        assert manifest["version"] == __version__

    def test_icon_is_a_square_png_the_directory_accepts(self) -> None:
        icon = _plugin_root() / ".claude-plugin" / "icon.png"
        raw = icon.read_bytes()
        assert raw[:8] == b"\x89PNG\r\n\x1a\n"
        width, height = struct.unpack(">II", raw[16:24])
        assert width == height and 512 <= width <= 2048
        assert len(raw) < 2 * 1024 * 1024
