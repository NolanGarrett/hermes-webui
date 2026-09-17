"""Regression: a freshly-committed state.db session must invalidate the CLI-session
cache immediately, even when the file-stat stamp would collide (the root cause of
the recurring test_gateway_sync flake).

The CLI-session cache (_CLI_SESSIONS_CACHE) and the session-list cache were keyed
on (st_mtime_ns, st_size) of state.db + its WAL sidecars. Under WAL-mode writes
those stamps can collide, serving a stale cache. The fix adds a commit-reliable
content fingerprint (_sqlite_content_fingerprint) that advances on every commit.
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path


def _use_active_home(monkeypatch, tmp_path):
    from api import models, profiles

    monkeypatch.setattr(profiles, "get_active_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(profiles, "get_active_profile_name", lambda: "default")
    monkeypatch.setattr(
        models,
        "_default_claude_code_projects_dir",
        lambda: tmp_path / "claude-projects",
    )
    return models


def test_single_profile_cache_key_tracks_projects_db_creation_and_change(
    monkeypatch, tmp_path
):
    models = _use_active_home(monkeypatch, tmp_path)
    projects_db = tmp_path / "projects.db"

    assert not projects_db.exists()
    key_missing = models._resolve_cli_sessions_context()[3]

    projects_db.write_bytes(b"first")
    key_created = models._resolve_cli_sessions_context()[3]

    projects_db.write_bytes(b"second-longer")
    key_changed = models._resolve_cli_sessions_context()[3]

    assert key_missing != key_created
    assert key_created != key_changed


def test_projects_db_change_invalidates_fixed_streaming_cache_key(monkeypatch, tmp_path):
    models = _use_active_home(monkeypatch, tmp_path)
    monkeypatch.setattr(models, "_active_stream_ids", lambda: {"fixed-stream"})
    projects_db = tmp_path / "projects.db"
    projects_db.write_bytes(b"first")

    key_before = models._resolve_cli_sessions_context()[3]
    projects_db.write_bytes(b"second-longer")
    key_after = models._resolve_cli_sessions_context()[3]

    assert key_before != key_after


def test_missing_projects_db_cache_fingerprint_is_read_only(monkeypatch, tmp_path):
    models = _use_active_home(monkeypatch, tmp_path)
    projects_db = tmp_path / "projects.db"

    models._resolve_cli_sessions_context()

    assert not projects_db.exists()
    assert not Path(f"{projects_db}-wal").exists()
    assert not Path(f"{projects_db}-shm").exists()


def test_content_fingerprint_advances_on_commit():
    """The cache key's content fingerprint must change after any commit, even
    when mtime/size would not reliably change (the WAL-collision flake source).
    """
    from api.models import _sqlite_file_stat_cache_key, _sqlite_content_fingerprint

    d = tempfile.mkdtemp()
    p = Path(d) / "state.db"
    conn = sqlite3.connect(str(p))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE sessions(id TEXT, source TEXT, started_at REAL)")
    conn.execute("CREATE TABLE messages(id INTEGER PRIMARY KEY, session_id TEXT)")
    conn.commit()

    fp_before = _sqlite_content_fingerprint(p)
    key_before = _sqlite_file_stat_cache_key(p)

    conn.execute("INSERT INTO sessions VALUES ('gw_new_001', 'weixin', 1)")
    conn.execute("INSERT INTO messages (session_id) VALUES ('gw_new_001')")
    conn.commit()

    fp_after = _sqlite_content_fingerprint(p)
    key_after = _sqlite_file_stat_cache_key(p)
    conn.close()

    assert fp_before != fp_after, (
        "content fingerprint must advance after a commit so a freshly-inserted "
        "CLI/gateway session is never served from a stale cache"
    )
    # The full cache key (which embeds the fingerprint) must therefore differ too.
    assert key_before != key_after


def test_content_fingerprint_detects_message_only_change():
    """An in-place session row update or message-only insert must also change the
    fingerprint (sessions COUNT/MAX alone could miss a same-rowid REPLACE).
    """
    from api.models import _sqlite_content_fingerprint

    d = tempfile.mkdtemp()
    p = Path(d) / "state.db"
    conn = sqlite3.connect(str(p))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE sessions(id TEXT, source TEXT, started_at REAL)")
    conn.execute("CREATE TABLE messages(id INTEGER PRIMARY KEY, session_id TEXT)")
    conn.execute("INSERT INTO sessions VALUES ('s1', 'weixin', 1)")
    conn.commit()

    fp_before = _sqlite_content_fingerprint(p)
    # Add a message to the existing session (sessions table unchanged).
    conn.execute("INSERT INTO messages (session_id) VALUES ('s1')")
    conn.commit()
    fp_after = _sqlite_content_fingerprint(p)
    conn.close()

    assert fp_before != fp_after, (
        "fingerprint must cover the messages table so a message-only commit "
        "(e.g. a continued gateway conversation) invalidates the cache"
    )


def test_content_fingerprint_safe_on_missing_or_empty_db():
    """The fingerprint must not raise on a missing path or a db without the
    expected tables — it returns None / zeroed parts so the stat fallback applies.
    """
    from api.models import _sqlite_content_fingerprint

    assert _sqlite_content_fingerprint(Path("/nonexistent/state.db")) is None

    d = tempfile.mkdtemp()
    p = Path(d) / "empty.db"
    conn = sqlite3.connect(str(p))
    conn.execute("CREATE TABLE unrelated(x)")
    conn.commit()
    conn.close()
    # No sessions/messages tables → None parts (MAX(rowid) on a missing table),
    # no exception. Shape is a 2-tuple of (sessions_max_rowid, messages_max_rowid).
    fp = _sqlite_content_fingerprint(p)
    assert fp == (None, None)
