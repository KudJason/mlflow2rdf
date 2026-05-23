"""
Config Loader — loads YAML config files from the config/ directory.

Provides a single entry point ``load_config()`` that returns a ConfigProxy
object. All internal consumers access configs via attribute-style access:

    cfg = load_config()
    cfg.property_routing['model']       # list of param keys
    cfg.property_uris['lora_rank']      # "hasLoraRank"
    cfg.paradigm_labels['supervised']   # "Supervised"
    cfg.metric_types['rmse']            # "root_mean_squared_error"
    cfg.task_types['task_types']        # dict
    cfg.task_types['openml_measures']   # dict
    cfg.task_types['known_datasets']    # dict
    cfg.sources['mlflow']               # MLflow connection config
    cfg.mappings['mappings']            # declarative mapping rules
    cfg.validation['validation']        # SHACL validation config

YAML files are loaded lazily on first access and cached.
"""

import os
import yaml
import logging
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Default config directory (relative to this file)
_DEFAULT_CONFIG_DIR = Path(__file__).parent / "config"

# Map attribute names to file names (without .yaml extension)
_CONFIG_FILES: Dict[str, str] = {
    "property_routing": "property_routing.yaml",
    "property_uris":    "property_uris.yaml",
    "paradigm_labels":  "paradigm_labels.yaml",
    "metric_types":     "metric_types.yaml",
    "task_types":       "task_types.yaml",
    "sources":          "sources.yaml",
    "mappings":         "mappings.yaml",
    "rml_mappings":     "rml_mappings.yaml",
    "validation":       "validation.yaml",
}


class ConfigProxy:
    """Lazy-loading proxy for all YAML configs.

    Each attribute access loads and caches the corresponding YAML file.
    Allows ``cfg.property_uris`` style access without dict nesting.
    """

    def __init__(self, config_dir: Path):
        self._config_dir = config_dir
        self._cache: Dict[str, Any] = {}

    def _load(self, name: str) -> Any:
        """Load a single config file, with caching."""
        if name in self._cache:
            return self._cache[name]

        filename = _CONFIG_FILES.get(name)
        if filename is None:
            raise AttributeError(f"Unknown config: {name!r}. Available: {list(_CONFIG_FILES)}")

        path = self._config_dir / filename
        if not path.exists():
            logger.warning(f"Config file not found: {path}")
            self._cache[name] = {}
            return self._cache[name]

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            self._cache[name] = data
            return data

    # --- Attribute shortcuts for commonly used configs ---

    @property
    def property_routing(self) -> Dict[str, list]:
        """Paradigm property routing: model/run/algorithm/data → [param keys]."""
        return self._load("property_routing")

    @property
    def property_uris(self) -> Dict[str, str]:
        """Param key → MLSO property name mapping."""
        return self._load("property_uris")

    @property
    def paradigm_labels(self) -> Dict[str, str]:
        """Tag value → human-readable paradigm label."""
        return self._load("paradigm_labels")

    @property
    def metric_types(self) -> Dict[str, str]:
        """Metric name → MLSO evaluation measure URI suffix."""
        return self._load("metric_types")

    @property
    def task_types(self) -> Dict[str, Any]:
        """Task type map + OpenML measures + known dataset metadata."""
        return self._load("task_types")

    @property
    def sources(self) -> Dict[str, Any]:
        """MLflow data source + output config."""
        return self._load("sources")

    @property
    def mappings(self) -> Dict[str, Any]:
        """Declarative KV→RDF mapping rules."""
        return self._load("mappings")

    @property
    def rml_mappings(self) -> Dict[str, Any]:
        """YARRRML RML standard mappings."""
        return self._load("rml_mappings")

    @property
    def validation(self) -> Dict[str, Any]:
        """SHACL validation config."""
        return self._load("validation")

    # --- Generic access ---

    def get(self, name: str, default: Any = None) -> Any:
        """Load any config by attribute name, with default fallback."""
        try:
            return self._load(name)
        except AttributeError:
            return default

    def __getattr__(self, name: str) -> Any:
        """Fallback: try to load any unknown name as a config file."""
        if name.startswith("_"):
            raise AttributeError(name)
        return self.get(name, {})

    def reload(self, name: Optional[str] = None):
        """Clear cache for one or all configs (for hot-reload)."""
        if name is None:
            self._cache.clear()
        else:
            self._cache.pop(name, None)

    def available_configs(self) -> list:
        """List all registered config names."""
        return list(_CONFIG_FILES.keys())

    def __repr__(self) -> str:
        return f"ConfigProxy(config_dir={self._config_dir}, cached={list(self._cache)})"


# ============================================================================
# Singleton loader
# ============================================================================

_global_config: Optional[ConfigProxy] = None


def load_config(config_dir: Optional[str] = None) -> ConfigProxy:
    """Load config directory and return a ConfigProxy singleton.

    Args:
        config_dir: Override the default config directory path.
                    If not provided, uses ``config/`` alongside this file.

    Returns:
        ConfigProxy — lazy-loading config accessor.
    """
    global _global_config
    if config_dir is not None:
        path = Path(config_dir).resolve()
        return ConfigProxy(path)

    if _global_config is None:
        _global_config = ConfigProxy(_DEFAULT_CONFIG_DIR)
    return _global_config


def reset_config():
    """Clear global singleton (useful for testing)."""
    global _global_config
    _global_config = None
