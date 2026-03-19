from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


class TurnLogger:
    """Append-only JSONL logger for bandit training tuples."""

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._path, "a")

    def log(
        self,
        session_id: str,
        turn: int,
        context: list,
        action: str,
        reward: float,
    ) -> None:
        record = {
            "session_id": session_id,
            "turn": turn,
            "context": context,
            "action": action,
            "reward": reward,
        }
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def close(self) -> None:
        if self._file and not self._file.closed:
            self._file.close()
