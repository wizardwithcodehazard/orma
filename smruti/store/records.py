"""records.py - Typed rows returned by the store."""

import json
from dataclasses import dataclass, fields
from datetime import timezone, tzinfo

from smruti import clock


def from_row(cls, row):
    if hasattr(cls, "from_row"):
        return cls.from_row(row)
    return cls(**{f.name: row[f.name] for f in fields(cls)})


def state_text(subject: str, attribute: str, value: str) -> str:
    return f"{subject} · {attribute.replace('_', ' ')}: {value}"


@dataclass
class Episode:
    """A kept record: a chat turn, a note, part of a document, a tool result."""
    id: int
    space: str
    role: str  # user | assistant | tool | note | document
    text: str
    outcome: str | None
    novelty: float | None
    seen_count: int
    created_at: float
    consolidated_at: float | None
    source: str  # where it came from: chat, api, file:<name>, ...
    speaker: str  # who wrote it; for user and note records, the user


@dataclass
class Sentence:
    id: int
    space: str
    episode_id: int
    position: int
    role: str
    speaker: str
    text: str
    created_at: float

    def label(self, tz: tzinfo | None) -> str:
        """The sentence as recall shows it: day, speaker, words."""
        return f"{clock.iso_day(self.created_at, tz)} {self.speaker}: {self.text}"


@dataclass
class Entity:
    id: int
    space: str
    name: str
    type: str
    mention_count: int
    last_seen: float


@dataclass
class Attribute:
    id: int
    space: str
    name: str


@dataclass
class State:
    """What one attribute of one entity is, from valid_from until valid_to (None while still true)."""
    id: int
    space: str
    subject_id: int
    attribute_id: int
    value: str
    valid_from: float
    valid_from_stated: bool
    valid_to: float | None
    sentence_id: int  # the sentence that first stated it
    ended_by: int | None  # the sentence that said it stopped, if one did
    evidence: list[int]  # every sentence that stated it
    evidence_times: list[float]  # when each was said
    subject: str
    attribute: str

    @property
    def text(self) -> str:
        return state_text(self.subject, self.attribute, self.value)

    @classmethod
    def from_row(cls, row) -> "State":
        values = {f.name: row[f.name] for f in fields(cls)}
        values["valid_from_stated"] = bool(values["valid_from_stated"])
        values["evidence"] = json.loads(values["evidence"])
        values["evidence_times"] = json.loads(values["evidence_times"])
        return cls(**values)


@dataclass
class Event:
    id: int
    space: str
    gist: str
    occurred_at: float
    occurred_stated: bool
    created_at: float
    sentence_id: int

    @classmethod
    def from_row(cls, row) -> "Event":
        values = {f.name: row[f.name] for f in fields(cls)}
        values["occurred_stated"] = bool(values["occurred_stated"])
        return cls(**values)

    def label(self, tz: tzinfo | None) -> str:
        """The gist after the day it happened in tz."""
        return f"{clock.iso_day(self.occurred_at, tz)}: {self.gist}"

    @property
    def text(self) -> str:
        return self.label(timezone.utc)
