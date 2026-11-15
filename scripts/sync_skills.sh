#!/usr/bin/env bash
# sync_skills.sh — keep ~/.claude/skills/ and the package source in sync.
#
# Usage:
#   ./scripts/sync_skills.sh           # package source → ~/.claude/skills/ (default)
#   ./scripts/sync_skills.sh --push    # ~/.claude/skills/ → package source (use after
#                                      # editing installed skills directly, e.g. in a
#                                      # Claude Code session)
#   ./scripts/sync_skills.sh --check   # diff only; exit 1 if any differ
#
# After running with no args / --push, automatically calls:
#   researchforge claude install --user --force
# so the hash manifest is updated and future installs report correct state.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PKG_SKILLS="$REPO_ROOT/src/researchforge/claude/skills"
USER_SKILLS="$HOME/.claude/skills"

# ── Argument parsing ─────────────────────────────────────────────────────────
MODE="install"   # default: pkg → user
for arg in "$@"; do
  case "$arg" in
    --push)   MODE="push"   ;;
    --check)  MODE="check"  ;;
    --help|-h)
      sed -n '2,20p' "$0" | sed 's/^# //'
      exit 0
      ;;
  esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────
differs=0
changed=0

check_pair() {
  local name="$1"
  local pkg="$PKG_SKILLS/$name/SKILL.md"
  local usr="$USER_SKILLS/$name/SKILL.md"

  if [ ! -f "$pkg" ] && [ ! -f "$usr" ]; then
    return
  fi

  if [ ! -f "$pkg" ]; then
    echo "MISSING in pkg : $name  (only in ~/.claude/skills/)"
    differs=1
    return
  fi

  if [ ! -f "$usr" ]; then
    echo "MISSING in user: $name  (only in package)"
    differs=1
    return
  fi

  if ! diff -q "$pkg" "$usr" > /dev/null 2>&1; then
    echo "DIFFERS        : $name"
    differs=1
  fi
}

# ── Main ──────────────────────────────────────────────────────────────────────
echo "=== ResearchForge skill sync ==="
echo "  Package : $PKG_SKILLS"
echo "  Installed: $USER_SKILLS"
echo ""

case "$MODE" in

  check)
    # Find all skill names from both directories
    all_skills=()
    while IFS= read -r d; do
      all_skills+=("$(basename "$d")")
    done < <(find "$PKG_SKILLS" "$USER_SKILLS" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort -u)

    for name in "${all_skills[@]}"; do
      check_pair "$name"
    done

    if [ "$differs" -eq 0 ]; then
      echo "✅ All skills in sync."
    else
      echo ""
      echo "❌ Some skills differ. Run without --check to sync."
      exit 1
    fi
    ;;

  push)
    # ~/.claude/skills/ → package source
    echo "Direction: ~/.claude/skills/ → package (--push)"
    echo ""

    if [ ! -d "$USER_SKILLS" ]; then
      echo "❌ $USER_SKILLS not found."
      exit 1
    fi

    for skill_dir in "$USER_SKILLS"/*/; do
      name="$(basename "$skill_dir")"
      src="$USER_SKILLS/$name/SKILL.md"
      dst_dir="$PKG_SKILLS/$name"
      dst="$dst_dir/SKILL.md"

      [ -f "$src" ] || continue

      mkdir -p "$dst_dir"
      if [ ! -f "$dst" ] || ! diff -q "$src" "$dst" > /dev/null 2>&1; then
        cp "$src" "$dst"
        echo "  ↑ $name"
        changed=$((changed + 1))
      else
        echo "  = $name  (unchanged)"
      fi
    done

    echo ""
    echo "Pushed $changed skill(s) to package source."
    echo ""
    echo "Running: researchforge claude install --user --force"
    researchforge claude install --user --force
    ;;

  install)
    # package source → ~/.claude/skills/ (via official installer)
    echo "Direction: package → ~/.claude/skills/"
    echo ""

    # First push any skills that are ahead in the installed location
    # (safety net — catches sessions where we edited ~/.claude/skills/ directly)
    ahead=0
    for skill_dir in "$USER_SKILLS"/*/; do
      name="$(basename "$skill_dir")"
      src="$USER_SKILLS/$name/SKILL.md"
      dst="$PKG_SKILLS/$name/SKILL.md"

      [ -f "$src" ] || continue
      [ -f "$dst" ] || continue

      if ! diff -q "$src" "$dst" > /dev/null 2>&1; then
        echo "  ⚠ $name differs — copying ~/.claude/skills/ → package first"
        cp "$src" "$dst"
        ahead=$((ahead + 1))
      fi
    done

    if [ "$ahead" -gt 0 ]; then
      echo ""
      echo "  Synced $ahead skill(s) back to package source."
      echo ""
    fi

    echo "Running: researchforge claude install --user --force"
    researchforge claude install --user --force
    ;;

esac
