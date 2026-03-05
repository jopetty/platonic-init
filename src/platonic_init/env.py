from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def _find_repo_root(start: Path | None = None) -> Path:
    cur = (start or Path.cwd()).resolve()
    for candidate in [cur, *cur.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return cur


def load_project_env(start: Path | None = None) -> Path | None:
    root = _find_repo_root(start)
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
        return env_path
    return None
