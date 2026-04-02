from __future__ import annotations

from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def load_project_dotenv(*, override: bool = False) -> None:
    """Load a ``.env`` file from the project root into ``os.environ`` if the file exists.

    Existing environment variables are kept unless ``override=True``.
    """
    from dotenv import load_dotenv

    load_dotenv(_PACKAGE_ROOT / ".env", override=override)
