from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class AgentSession:
    sender: str
    pending_action: str | None = None
    pending_workflow: str | None = None
    idempotency_key: str | None = None
    job_id: str | None = None
    adapter_id: str | None = None


class SessionStore:
    """Small persistent state store. Document contents are deliberately excluded."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._initialize()

    def get(self, sender: str) -> AgentSession:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM asi1_sessions WHERE sender = ?", (sender,)
            ).fetchone()
        if row is None:
            return AgentSession(sender=sender)
        return AgentSession(**dict(row))

    def save(self, session: AgentSession) -> None:
        values = asdict(session)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO asi1_sessions(
                    sender, pending_action, pending_workflow, idempotency_key,
                    job_id, adapter_id
                ) VALUES(
                    :sender, :pending_action, :pending_workflow, :idempotency_key,
                    :job_id, :adapter_id
                )
                ON CONFLICT(sender) DO UPDATE SET
                    pending_action = excluded.pending_action,
                    pending_workflow = excluded.pending_workflow,
                    idempotency_key = excluded.idempotency_key,
                    job_id = excluded.job_id,
                    adapter_id = excluded.adapter_id
                """,
                values,
            )

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS asi1_sessions(
                    sender TEXT PRIMARY KEY,
                    pending_action TEXT,
                    pending_workflow TEXT,
                    idempotency_key TEXT,
                    job_id TEXT,
                    adapter_id TEXT
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection
