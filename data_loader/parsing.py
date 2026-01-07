from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class SpectroscopyMeta:
    pulse: str              # lorentzian / broad / ...
    alpha: float | None
    echo: bool
    date: str
    path: Path


_LOR_RE = re.compile(
    r"(?P<pulse>lorentzian)"
    r"(_echo)?"
    r"(?P<alpha>[0-9.]+)?"
)


def parse_metadata(path: Path) -> SpectroscopyMeta:
    name = path.stem

    echo = "_echo" in name
    pulse = "broad" if "broad" in name else "lorentzian"

    alpha = None
    m = re.search(r"([0-9.]+)$", name)
    if m:
        alpha = float(m.group(1))

    date = path.parent.name  # e.g. "5-2-25"

    return SpectroscopyMeta(
        pulse=pulse,
        alpha=alpha,
        echo=echo,
        date=date,
        path=path,
    )
