"""
Artifact Parser — extracts artifact metadata from MLflow run records.

Handles artifact path classification and metadata extraction for
different artifact types (model files, plots, configs, vectorizers, etc.).
"""

from pathlib import Path


def parse_artifact_path(artifact_path: str) -> dict:
    """Parse an artifact path into structured components.

    Returns:
        Dict with keys: filename, extension, dirname, stem, is_dir.
    """
    p = Path(artifact_path)
    return {
        'filename': p.name,
        'extension': p.suffix.lower(),
        'dirname': str(p.parent),
        'stem': p.stem,
        'is_dir': p.suffix == '',
    }


def classify_artifact(artifact_path: str) -> str:
    """Classify an artifact into a semantic category.

    Categories: model, plot, config, tabular, vectorizer, tokenizer, other.
    """
    info = parse_artifact_path(artifact_path)
    ext = info['extension']
    stem = info['stem'].lower()
    dirname = info['dirname'].lower()

    if ext in ('.pt', '.bin', '.pth', '.pkl', '.joblib'):
        return 'model'
    if ext in ('.png', '.jpg', '.jpeg', '.svg'):
        return 'plot'
    if ext == '.json' and ('config' in stem or 'anchor' in stem):
        return 'config'
    if ext in ('.json', '.csv'):
        return 'tabular'
    if 'vectorizer' in stem or 'tfidf' in stem:
        return 'vectorizer'
    if 'tokenizer' in stem:
        return 'tokenizer'
    if 'feature_importance' in stem:
        return 'feature_importance'
    return 'other'
