"""
scheduler.py - Background sleep. While the proxy serves requests, a daemon thread periodically
consolidates every space that is due (enough pending turns, or idle). Failures are logged and
retried on the next poll; they never reach a request.
"""

import logging
import threading

logger = logging.getLogger(__name__)


class SleepScheduler:
    def __init__(self, memory, poll_seconds: float):
        self.memory = memory
        self.poll_seconds = poll_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def run_once(self, now=None) -> list[str]:
        """Sleeps every due space; returns the spaces consolidated."""
        if self.memory.llm is None or self.memory.model is None:
            return []  # the model is learned from the first proxied request
        done = []
        for space in self.memory.due_spaces(now):
            try:
                self.memory.sleep(space, now=now)
                done.append(space)
            except Exception:
                logger.exception("sleep failed for space %r; will retry", space)
        return done

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="smruti-sleep", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stop.wait(self.poll_seconds):
            self.run_once()
