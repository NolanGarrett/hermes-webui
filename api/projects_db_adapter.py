"""Read-only bridge to Hermes Agent's per-profile projects database."""

from __future__ import annotations

import importlib
import sqlite3
from collections.abc import Iterable
from contextlib import closing
from pathlib import Path
from typing import Any

from api import profiles


_SQLITE_TIMEOUT_SECONDS = 1.0


def _requested_profile(profile_name: str | None) -> str | None:
    name = profiles.get_active_profile_name() if profile_name is None else profile_name
    if not isinstance(name, str) or not profiles._PROFILE_ID_RE.fullmatch(name):
        return None
    if profiles._is_isolated_profile_mode() and name != profiles._isolated_profile_name():
        return None
    return name


def _profile_home(profile_name: str) -> Path | None:
    home = Path(profiles.get_hermes_home_for_profile(profile_name)).expanduser()
    if profiles._is_root_profile(profile_name):
        return home if home.is_dir() else None
    if home.name != profile_name or home.parent.name != "profiles" or not home.is_dir():
        return None
    return home


def _open_read_only(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(
        f"{db_path.resolve().as_uri()}?mode=ro",
        uri=True,
        timeout=_SQLITE_TIMEOUT_SECONDS,
    )
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
    except Exception:
        conn.close()
        raise
    return conn


def _folder_dict(folder: Any) -> dict:
    to_dict = getattr(folder, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        if not isinstance(value, dict):
            raise TypeError("project folder to_dict() must return a dict")
        return dict(value)
    if isinstance(folder, dict):
        return dict(folder)
    raise TypeError("incompatible project folder DTO")


def _project_dict(project: Any, profile_name: str) -> dict:
    project_id = getattr(project, "id", None)
    slug = getattr(project, "slug", None)
    name = getattr(project, "name", None)
    if not project_id or not slug or not name:
        raise TypeError("native project is missing required identity fields")
    return {
        "project_id": project_id,
        "native_project_id": project_id,
        "slug": slug,
        "name": name,
        "description": getattr(project, "description", None),
        "icon": getattr(project, "icon", None),
        "color": getattr(project, "color", None),
        "board_slug": getattr(project, "board_slug", None),
        "primary_path": getattr(project, "primary_path", None),
        "folders": [_folder_dict(folder) for folder in getattr(project, "folders", [])],
        "profile": profile_name,
        "created_at": getattr(project, "created_at", None),
        "archived": bool(getattr(project, "archived", False)),
        "project_source": "hermes-agent",
        "read_only": True,
    }


def load_native_projects(*, profile_name: str | None = None) -> list[dict] | None:
    """Return active native projects without creating or migrating their store."""
    try:
        resolved_profile = _requested_profile(profile_name)
        if resolved_profile is None:
            return None
        home = _profile_home(resolved_profile)
        if home is None:
            return None
        db_path = home / "projects.db"
        if not db_path.is_file():
            return None
        projects_db = importlib.import_module("hermes_cli.projects_db")
        with closing(_open_read_only(db_path)) as conn:
            try:
                projects = projects_db.list_projects(conn, include_archived=False)
            except TypeError:
                projects = projects_db.list_projects(conn)
            return [_project_dict(project, resolved_profile) for project in projects]
    except Exception:
        return None


def native_project_ids_for_paths(
    paths: Iterable[str | None], *, profile_name: str | None = None
) -> dict[str, str] | None:
    """Resolve paths to native project ids through the upstream matcher."""
    try:
        resolved_profile = _requested_profile(profile_name)
        if resolved_profile is None:
            return None
        home = _profile_home(resolved_profile)
        if home is None:
            return None
        db_path = home / "projects.db"
        if not db_path.is_file():
            return None
        projects_db = importlib.import_module("hermes_cli.projects_db")
        distinct_paths = list(
            dict.fromkeys(path for path in paths if isinstance(path, str) and path.strip())
        )
        result: dict[str, str] = {}
        with closing(_open_read_only(db_path)) as conn:
            for path in distinct_paths:
                try:
                    project = projects_db.project_for_path(
                        conn, path, include_archived=False
                    )
                except TypeError:
                    project = projects_db.project_for_path(conn, path)
                if project is not None:
                    project_id = getattr(project, "id", None)
                    if not project_id:
                        raise TypeError("native project is missing its id")
                    result[path] = project_id
        return result
    except Exception:
        return None
