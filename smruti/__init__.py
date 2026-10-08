"""
smruti (स्मृति) - Biologically Inspired, Dual-Valence Cognitive Memory for Autonomous AI Agents.
"""

from smruti.config import SmrutiConfig, get_config, smrutiConfig
from smruti.engine.consolidator import Consolidator
from smruti.engine.cortex import Cortex
from smruti.engine.inhibitory import InhibitoryGate
from smruti.engine.stream import StreamBuffer
from smruti.framework import Smruti, SmrutiInhibitionError, smruti, smrutiInhibitionError
from smruti.models import ActionStatus, AntiMemory, CorticalRule, Episode, InhibitionResult
from smruti.storage.db import DatabaseManager, get_db

# Developer ergonomic alias
SmrutiMemory = Smruti

__version__ = "0.2.0"
__all__ = [
    "ActionStatus",
    "AntiMemory",
    "Consolidator",
    "Cortex",
    "CorticalRule",
    "DatabaseManager",
    "Episode",
    "InhibitionResult",
    "InhibitoryGate",
    "Smruti",
    "SmrutiConfig",
    "SmrutiInhibitionError",
    "smruti",
    "smrutiConfig",
    "smrutiInhibitionError",
    "StreamBuffer",
    "get_config",
    "get_db",
]
