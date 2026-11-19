"""SQLite HTTP response caching layer."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

_LOCAL_DATA_DIR = Path(os.path.abspath(__file__)).parent.parent / "data"
_PARENT_SHARED_DB = Path(os.path.abspath(__file__)).parent.parent.parent / "data" / "cache.db"
if not (_LOCAL_DATA_DIR / "cache.db").exists() and _PARENT_SHARED_DB.exists():
    DATA_DIR = _PARENT_SHARED_DB.parent
else:
    DATA_DIR = _LOCAL_DATA_DIR
DB_PATH = DATA_DIR / "cache.db"

_DB_LOCK = threading.Lock()


def _get_connection() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS http_cache (
            cache_key TEXT PRIMARY KEY,
            payload TEXT NOT NULL,
            fetched_at REAL NOT NULL
        )
        """
    )
    conn.commit()
    return conn


_CONN: Optional[sqlite3.Connection] = None


def _conn() -> sqlite3.Connection:
    global _CONN
    if _CONN is None:
        _CONN = _get_connection()
    return _CONN


def get_cached(key: str, max_age_seconds: float) -> Optional[Any]:
    """Return cached JSON-decoded payload if present and younger than max_age_seconds."""
    now = time.time()
    with _DB_LOCK:
        cur = _conn().execute(
            "SELECT payload, fetched_at FROM http_cache WHERE cache_key = ?",
            (key,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    payload_str, fetched_at = row
    if max_age_seconds >= 0 and (now - float(fetched_at)) > max_age_seconds:
        return None
    try:
        return json.loads(payload_str)
    except json.JSONDecodeError:
        return None


def get_cached_timestamp(key: str) -> Optional[float]:
    """Return the UNIX timestamp when a cache key was last stored."""
    with _DB_LOCK:
        cur = _conn().execute(
            "SELECT fetched_at FROM http_cache WHERE cache_key = ?",
            (key,),
        )
        row = cur.fetchone()
    return float(row[0]) if row else None


def set_cached(key: str, data: Any) -> None:
    """Persist JSON-serializable data under cache_key with current timestamp."""
    payload_str = json.dumps(data, separators=(",", ":"))
    now = time.time()
    with _DB_LOCK:
        _conn().execute(
            """
            INSERT INTO http_cache (cache_key, payload, fetched_at)
            VALUES (?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                payload = excluded.payload,
                fetched_at = excluded.fetched_at
            """,
            (key, payload_str, now),
        )
        _conn().commit()
