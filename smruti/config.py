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
        """
        Dynamically finds .smruti in current working directory, parent git repositories,
        active editor/IDE workspaces, or falls back to user home directory (~/.smruti).
        Prevents polluting application installation directories.
        """
        import os
        env_dir = os.environ.get("SMRUTI_DIR")
        if env_dir:
            return Path(env_dir).resolve()

        current = self.project_dir.resolve()

        # 1. Search upwards from current directory for an existing .smruti
        for parent in [current, *current.parents]:
            candidate = parent / self.smruti_dir_name
            if candidate.is_dir():
                return candidate

        # 2. Search upwards for a git repository or project anchor
        for parent in [current, *current.parents]:
            if (parent / ".git").exists() or (parent / "pyproject.toml").exists() or (parent / "package.json").exists():
                return parent / self.smruti_dir_name

        # 3. Detect if process was launched inside an editor/IDE binary folder
        # e.g., AppData\Local\Programs, Program Files, site-packages, /usr/bin
        current_str = str(current).lower()
        is_app_dir = any(
            bad in current_str for bad in [
                "appdata\\local\\programs",
                "program files",
                "site-packages",
                "/usr/bin",
                "/usr/lib",
                "/applications"
            ]
        )

        if is_app_dir:
            # Check for active workspace folders from common editor storage state
            try:
                import json
                from urllib.parse import unquote, urlparse
                for storage_path in [
                    Path.home() / "AppData" / "Roaming" / "Antigravity IDE" / "User" / "globalStorage" / "storage.json",
                    Path.home() / "AppData" / "Roaming" / "Code" / "User" / "globalStorage" / "storage.json",
                    Path.home() / "AppData" / "Roaming" / "Cursor" / "User" / "globalStorage" / "storage.json",
                ]:
                    if storage_path.exists():
                        data = json.loads(storage_path.read_text(encoding="utf-8"))
                        backup = data.get("backupWorkspaces", {})
                        folders = backup.get("folders", [])
                        for f in folders:
                            uri = f.get("folderUri", "")
                            parsed = urlparse(uri)
                            raw_path = unquote(parsed.path)
                            if raw_path.startswith("/") and len(raw_path) > 2 and raw_path[2] == ":":
                                raw_path = raw_path[1:]
                            ws_path = Path(raw_path)
                            if ws_path.exists():
                                if (ws_path / self.smruti_dir_name).is_dir():
                                    return ws_path / self.smruti_dir_name
                                for sub in ws_path.iterdir():
                                    if sub.is_dir() and (sub / self.smruti_dir_name).is_dir():
                                        return sub / self.smruti_dir_name
            except Exception:
                pass

            # Safe global fallback for agent daemons
            return Path.home() / self.smruti_dir_name

        # 4. Standard fallback in writable project directory
        if not os.access(current, os.W_OK) or current == Path("/"):
            return Path.home() / self.smruti_dir_name

        return current / self.smruti_dir_name

    @property
    def db_path(self) -> Path:
        return self.smruti_dir / self.db_filename

def get_config() -> smrutiConfig:
    return smrutiConfig()

SmrutiConfig = smrutiConfig
