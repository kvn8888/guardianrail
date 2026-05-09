from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AuditEvent:
    session_id: str
    prompt: str
    response: str
    model_id: str
    sae_release: str
    layer: int
    action: str
    rule_name: str
    feature_id: int | None = None
    feature_label: str | None = None
    activation: float | None = None
    threshold: float | None = None
    metadata: dict[str, Any] | None = None


SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  session_id TEXT NOT NULL,
  prompt TEXT NOT NULL,
  response TEXT NOT NULL,
  model_id TEXT NOT NULL,
  sae_release TEXT NOT NULL,
  layer INTEGER NOT NULL,
  feature_id INTEGER,
  feature_label TEXT,
  activation REAL,
  threshold REAL,
  action TEXT NOT NULL,
  rule_name TEXT NOT NULL,
  metadata_json TEXT
);
"""


def connect(db_path: str | Path = "artifacts/guardianrail.sqlite3") -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute(SCHEMA)
    conn.commit()
    return conn


def write_event(conn: sqlite3.Connection, event: AuditEvent) -> int:
    payload = asdict(event)
    metadata = payload.pop("metadata")
    created_at = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        """
        INSERT INTO audit_events (
          created_at, session_id, prompt, response, model_id, sae_release, layer,
          feature_id, feature_label, activation, threshold, action, rule_name, metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            created_at,
            payload["session_id"],
            payload["prompt"],
            payload["response"],
            payload["model_id"],
            payload["sae_release"],
            payload["layer"],
            payload["feature_id"],
            payload["feature_label"],
            payload["activation"],
            payload["threshold"],
            payload["action"],
            payload["rule_name"],
            json.dumps(metadata or {}, sort_keys=True),
        ),
    )
    conn.commit()
    return int(cursor.lastrowid)


def read_events(conn: sqlite3.Connection, limit: int = 50) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT *
        FROM audit_events
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [dict(row) for row in rows]

