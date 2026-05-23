"""
Run Parser — extracts and normalizes MLflow run metadata from raw records.

Provides helpers for timestamp normalization, status mapping,
and nested tag/param extraction.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional


def normalize_timestamp(ts: Any) -> Optional[str]:
    """Convert a timestamp (int/float/str) to ISO 8601 string.

    Handles MLflow's millisecond timestamps.
    """
    if ts is None:
        return None
    try:
        ms = int(ts)
        dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
        return dt.isoformat()
    except (ValueError, TypeError):
        return str(ts)


def normalize_status(status: Optional[str]) -> str:
    """Normalize MLflow status to uppercase standard."""
    if not status:
        return 'UNKNOWN'
    return status.upper()


def extract_run_name(record: dict) -> Optional[str]:
    """Extract best-effort run name from tags or record fields."""
    tags = record.get('tags', {})
    name = tags.get('mlflow.runName') or record.get('run_name')
    return name


def extract_dataset_name(record: dict) -> Optional[str]:
    """Extract dataset name from tags."""
    return record.get('tags', {}).get('mlsea.dataset')


def extract_paradigm(record: dict) -> Optional[str]:
    """Extract paradigm label from tags."""
    return record.get('tags', {}).get('mlsea.paradigm')


def extract_algorithm_name(record: dict) -> Optional[str]:
    """Extract algorithm/model name from tags or params."""
    tags = record.get('tags', {})
    params = record.get('params', {})
    return tags.get('mlsea.algorithm') or params.get('model_name')


def extract_framework(record: dict) -> Optional[str]:
    """Extract ML framework from params or tags."""
    params = record.get('params', {})
    tags = record.get('tags', {})
    return params.get('sklearn_version') or tags.get('mlsea.framework')
