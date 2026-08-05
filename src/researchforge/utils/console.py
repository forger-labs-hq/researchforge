"""Shared Rich console and ResearchForge brand palette.

Palette — Academic Violet (Option A):
    Primary   #7C3AED   violet  — brand
    Accent    #F59E0B   amber   — forge fire / highlights
    Success   #10B981   emerald — ok / pass
    Error     #EF4444   red     — fail / missing required
    Warning   #F59E0B   amber   — caveats
    Muted     #9CA3AF   grey    — secondary text
    Path      #60A5FA   blue    — file / directory paths
"""

from __future__ import annotations

from rich.console import Console
from rich.theme import Theme

_RF_THEME = Theme(
    {
        "rf.primary": "bold #7C3AED",
        "rf.accent": "bold #F59E0B",
        "rf.success": "#10B981",
        "rf.error": "bold #EF4444",
        "rf.warning": "#F59E0B",
        "rf.muted": "dim #9CA3AF",
        "rf.path": "#60A5FA",
    }
)

console = Console(theme=_RF_THEME, highlight=False)

# ---------------------------------------------------------------------------
# Fox mascot — ASCII art in brand colours
# Ears / face in amber; brand name in bold violet; tagline muted grey.
# ---------------------------------------------------------------------------

_FOX_ART = (
    "[#F59E0B]  /\\   /\\ [/]\n"
    "[#F59E0B] ( o   o )[/]"
    "  [bold #7C3AED]ResearchForge[/]  [dim #9CA3AF]{version}[/]\n"
    "[#F59E0B]  \\_____/[/] "
    "  [dim #9CA3AF]From papers to proof.[/]"
)


def print_banner() -> None:
    """Print the ResearchForge banner to stdout.

    Shows the logo path hint when the image is installed, ASCII fox otherwise.
    (Terminal output can't render images — the real logo appears in the hub UI.)
    """
    from researchforge import __version__

    console.print(_FOX_ART.format(version=f"v{__version__}"))
