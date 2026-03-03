from __future__ import annotations

import time
from pathlib import Path


LOG_PATH = Path("logs.txt")


def log_with_timestamp(line: str, path: Path = LOG_PATH) -> None:
    """Append a log line with a millisecond timestamp prefix."""
    ts_ms = int(time.time() * 1000)
    entry = f"{ts_ms}\t{line}\n"
    path.write_text("", encoding="utf-8") if not path.exists() else None
    with path.open("a", encoding="utf-8") as f:
        f.write(entry)


def log_and_print(line: str, path: Path = LOG_PATH) -> None:
    print(line)
    log_with_timestamp(line, path)
