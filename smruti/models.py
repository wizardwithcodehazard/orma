"""
models.py - Pydantic schemas and data contracts for smruti memory engine.
"""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ActionStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    ERROR = "error"
    INTERCEPTED = "intercepted"

class ValenceType(str, Enum):
    POSITIVE = "positive"      # Excitatory / promoted heuristics
    INHIBITORY = "inhibitory"  # Anti-memory / blocked patterns

class Episode(BaseModel):
    """Tier 1: High-resolution episodic event log."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = Field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
    session_id: str = "default_session"
    context: str = ""
    action: str
    result: str = ""
    status: ActionStatus = ActionStatus.SUCCESS
    latency_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    consolidated_at: float | None = None

class InhibitionResult(BaseModel):
    """Preflight check evaluation result."""
    passed: bool
    matched_signature: str | None = None
    reason: str | None = None
    suggested_fix: str | None = None
    severity: str = "warning"
    confidence: float = 0.0
    matched_sources: list[str] = Field(default_factory=list)

class AntiMemory(BaseModel):
    """Tier 3B: Inhibitory anti-memory representing a known dead end."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    signature: str               # Canonical failure label (e.g., 'port_8000_collision')
    pattern: str                 # Substring or regex to intercept
    reason: str                  # Explanation of the failure
    suggested_fix: str | None = None
    context_tags: list[str] = Field(default_factory=list)
    times_triggered: int = 0
    severity: str = "high"       # 'critical', 'high', 'warning'
    created_at: float = Field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
    last_seen: float = Field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
    valence: ValenceType = ValenceType.INHIBITORY
    project_root: str | None = None
    is_active: bool = True
    embedding: list[float] | None = None
    expires_at: float | None = None


class CorticalRule(BaseModel):
    """Tier 3A: Positive heuristic rule with biological decay and reinforcement."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    rule_text: str
    category: str = "general"
    confidence: float = 1.0
    access_count: int = 0
    created_at: float = Field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
    last_accessed_at: float = Field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
    base_strength: float = 1.0
    valence: ValenceType = ValenceType.POSITIVE
    embedding: list[float] | None = None
    source_episode_ids: list[str] = Field(default_factory=list)
    
    def calculate_effective_strength(self, decay_rate: float, current_time: float | None = None) -> float:
        """Computes exponential retention based on elapsed time since last access."""
        import math
        now = current_time or datetime.now(timezone.utc).timestamp()
        hours_elapsed = max(0.0, (now - self.last_accessed_at) / 3600.0)
        return self.base_strength * math.exp(-decay_rate * hours_elapsed)

class RuleEdge(BaseModel):
    """Tier 3A: Associative synaptic connection between cortical rules."""
    rule_id_a: str
    rule_id_b: str
    weight: float
    created_at: float = Field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
