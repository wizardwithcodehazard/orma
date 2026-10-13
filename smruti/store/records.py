"""records.py - Typed rows returned by the store."""

import json
from dataclasses import dataclass, fields


def fact_text(subject: str, predicate: str, object_: str) -> str:
    return f"{subject} {predicate.replace('_', ' ')} {object_}"


def from_row(cls, row):
    if hasattr(cls, "from_row"):
        return cls.from_row(row)
    return cls(**{f.name: row[f.name] for f in fields(cls)})


@dataclass
class Episode:
    id: int
    space: str
    role: str
    text: str
    outcome: str | None
    novelty: float | None
    seen_count: int
    created_at: float
    consolidated_at: float | None


@dataclass
class Entity:
    id: int
    space: str
    name: str
    type: str
    mention_count: int
    last_seen: float


@dataclass
class Predicate:
    id: int
    space: str
    name: str
    cardinality: str


@dataclass
class Fact:
    id: int
    space: str
    subject_id: int
    predicate_id: int
    object_id: int | None
    object_value: str | None
    confidence: float
    support: int
    valid_from: float
    valid_to: float | None
    last_used: float
    source_episode_ids: list[int]
    valid_from_stated: bool
    subject: str
    predicate: str
    object: str

    @property
    def text(self) -> str:
        return fact_text(self.subject, self.predicate, self.object)

    @classmethod
    def from_row(cls, row) -> "Fact":
        values = {f.name: row[f.name] for f in fields(cls)}
        values["source_episode_ids"] = json.loads(values["source_episode_ids"])
        values["valid_from_stated"] = bool(values["valid_from_stated"])
        return cls(**values)


@dataclass
class Summary:
    id: int
    space: str
    topic: str
    text: str
    tokens: int
    updated_at: float


@dataclass
class Procedure:
    id: int
    space: str
    trigger: str
    action: str
    error_class: str | None
    successes: int
    failures: int
    created_at: float
    last_used: float
    source_episode_ids: list[int]

    @classmethod
    def from_row(cls, row) -> "Procedure":
        values = {f.name: row[f.name] for f in fields(cls)}
        values["source_episode_ids"] = json.loads(values["source_episode_ids"])
        return cls(**values)

    @property
    def works(self) -> bool:
        return self.successes >= self.failures

    @property
    def confidence(self) -> float:
        """Beta-posterior mean of the majority outcome."""
        return (max(self.successes, self.failures) + 1) / (self.successes + self.failures + 2)

    @property
    def text(self) -> str:
        return f"When {self.trigger}: {self.action} ({'works' if self.works else 'fails'})"
