"""
config.py - Dynamic environment and runtime configuration for smruti.
Discovers local .smruti/ directory or falls back to user home directory.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class smrutiConfig:
    # Storage settings
    db_filename: str = "smruti.db"
    project_dir: Path = Path.cwd()
    smruti_dir_name: str = ".smruti"
    
    # Biological decay parameters
    # decay_rate: controls how fast unused rules decay over time (in hours^-1)
    default_decay_rate: float = 0.05
    # prune_threshold: rules with effective strength below this are archived
    prune_threshold: float = 0.15
    # reinforcement_boost: strength added per successful recall
    reinforcement_boost: float = 0.15
    
    # Semantic & Associative Graph parameters
    dedup_similarity_threshold: float = 0.85
    associative_edge_threshold: float = 0.60
    spreading_activation_factor: float = 0.25
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Consolidation thresholds
    consolidation_turn_interval: int = 5
    auto_consolidate: bool = True
    raw_retention_days: int = 7

    # Pluggable Storage Backend (sqlite / postgres)
    backend_type: str = "sqlite"
    postgres_url: str | None = None


    @property
    def smruti_dir(self) -> Path:
        """Finds .smruti in current working directory or ancestors, else creates in cwd or home directory."""
        import os
        env_dir = os.environ.get("SMRUTI_DIR")
        if env_dir:
            return Path(env_dir).resolve()
        current = self.project_dir.resolve()
        if current == Path("/"):
            return Path.home() / self.smruti_dir_name
        nearest = current
        while not nearest.exists() and nearest != nearest.parent:
            nearest = nearest.parent
        if nearest == Path("/") or not os.access(nearest, os.W_OK):
            return Path.home() / self.smruti_dir_name
        for parent in [current, *current.parents]:
            candidate = parent / self.smruti_dir_name
            if candidate.is_dir():
                return candidate
        return current / self.smruti_dir_name

    @property
    def db_path(self) -> Path:
        return self.smruti_dir / self.db_filename

def get_config() -> smrutiConfig:
    return smrutiConfig()

SmrutiConfig = smrutiConfig
