from __future__ import annotations

from dataclasses import dataclass
import shutil
import time
from pathlib import Path
from typing import Callable, Optional


@dataclass
class LabelLatch:
    active: str
    ever_received: bool = False
    last_timestamp_ns: Optional[int] = None
    change_count: int = 0

    def update(self, label: str, timestamp_ns: int) -> None:
        label = label.strip()
        if not label:
            return
        if label != self.active:
            self.change_count += 1
        self.active = label
        self.ever_received = True
        self.last_timestamp_ns = int(timestamp_ns)


@dataclass
class DiskSpaceGuard:
    min_free_gb: float = 10.0
    period_sec: float = 5.0
    enabled: bool = True
    last_check_monotonic: float = 0.0
    last_free_gb: Optional[float] = None

    def available(self, path: Path, now: Optional[float] = None) -> bool:
        now = time.monotonic() if now is None else float(now)
        if now - self.last_check_monotonic >= self.period_sec or self.last_free_gb is None:
            self.last_check_monotonic = now
            self.last_free_gb = shutil.disk_usage(path).free / (1024 ** 3)
        return not self.enabled or self.last_free_gb >= self.min_free_gb


def wait_for_thread_shutdown(
    thread,
    warning_after_sec: float,
    on_warning: Optional[Callable[[float], None]] = None,
    poll_sec: float = 0.2,
) -> float:
    """Wait until a writer thread really exits before its file handles are closed.

    ``warning_after_sec`` is deliberately only a warning threshold.  Returning while
    the thread is alive would allow it to write to already-closed CSV/debug handles.
    """
    started = time.monotonic()
    warned = False
    while thread.is_alive():
        thread.join(timeout=max(0.01, float(poll_sec)))
        elapsed = time.monotonic() - started
        if not warned and elapsed >= max(0.0, float(warning_after_sec)):
            warned = True
            if on_warning is not None:
                on_warning(elapsed)
    return time.monotonic() - started
