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
        try:
            due = self.memory.due_spaces(now)
        except Exception:
            logger.exception("checking for due spaces failed; will retry")
            return done
        for space in due:
            try:
                self.memory.sleep(space, now=now)
                done.append(space)
            except Exception:
                logger.exception("sleep failed for space %r; will retry", space)
        return done

    def flush(self, now=None) -> list[str]:
        """Runs a consolidation pass on any space with pending records (e.g. on shutdown or startup catch-up)."""
        if self.memory.llm is None or self.memory.model is None:
            return []
        done = []
        try:
            pending_spaces = [space for space, pending, _ in self.memory.store.pending_stats() if pending > 0]
        except Exception:
            return done
        for space in pending_spaces:
            try:
                self.memory.sleep(space, now=now)
                done.append(space)
            except Exception:
                logger.debug("immediate sleep flush for space %r failed", space)
        return done

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="orma-sleep", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        # 1. Startup catch-up: wait 1.5s for upstream runtime to stabilize, then flush any pending records
        if not self._stop.wait(1.5):
            try:
                consolidated = self.flush()
                if consolidated:
                    logger.info("startup sleep catch-up consolidated spaces: %s", consolidated)
            except Exception:
                logger.debug("startup sleep catch-up skipped; will retry in normal loop")

        # 2. Main periodic sleep loop
        while not self._stop.wait(self.poll_seconds):
            try:
                self.run_once()
            except Exception:
                logger.exception("sleep iteration failed; will retry")
