"""orma - local episode-first long-term memory for small LLMs."""

from orma.config import OrmaConfig, SmrutiConfig
from orma.memory import Memory

__version__ = "0.2.2"
__all__ = ["Memory", "OrmaConfig", "SmrutiConfig"]
