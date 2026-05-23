"""
URI Utilities — helpers for generating and resolving RDF URIs.

Provides consistent URI construction for MLflow entities,
prefix expansion, and slug generation.
"""

import hashlib
import re
from typing import Optional
from ..config import MLFLOW, slugify


def run_uri(run_id: str) -> str:
    """Generate a URI for an MLflow run."""
    return str(MLFLOW[f'run/{run_id}'])


def experiment_uri(experiment_id: str) -> str:
    """Generate a URI for an MLflow experiment."""
    return str(MLFLOW[f'experiment/{experiment_id}'])


def algorithm_uri(algo_name: str) -> str:
    """Generate a URI for a named algorithm."""
    return str(MLFLOW[f'algorithm/{slugify(algo_name)}'])


def dataset_uri(dataset_name: str) -> str:
    """Generate a URI for a dataset."""
    return str(MLFLOW[f'dataset/{slugify(dataset_name)}'])


def model_uri(run_id: str) -> str:
    """Generate a URI for the model of a run."""
    return str(MLFLOW[f'model/{run_id}'])


def user_uri(user_id: str) -> str:
    """Generate a URI for a user."""
    return str(MLFLOW[f'user/{slugify(user_id)}'])


def param_setting_uri(run_id: str, param_key: str) -> str:
    """Generate a URI for a hyperparameter setting."""
    return str(MLFLOW[f'run/{run_id}/param/{slugify(param_key)}'])


def metric_uri(run_id: str, metric_key: str) -> str:
    """Generate a URI for a metric evaluation."""
    return str(MLFLOW[f'run/{run_id}/metric/{slugify(metric_key)}'])


def artifact_uri(run_id: str, artifact_path: str) -> str:
    """Generate a URI for an artifact."""
    slug = slugify(artifact_path.replace('/', '-').replace('.', '-'))
    return str(MLFLOW[f'run/{run_id}/artifact/{slug}'])


def safe_uri_id(value: str) -> str:
    """Generate a safe URI fragment from any value, falling back to hash."""
    slug = slugify(value)
    if slug == 'value' and value:
        return hashlib.md5(value.encode()).hexdigest()[:8]
    return slug
