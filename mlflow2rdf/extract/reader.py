import os
from pathlib import Path

def parse_meta_yaml(path: Path) -> dict:
    """Parses a simple MLflow meta.yaml file into a dictionary."""
    result = {}
    if not path.exists():
        return result
    for line in path.read_text().splitlines():
        if ":" not in line or line.strip().startswith("#"):
            continue
        k, _, v = line.strip().partition(":")
        v_val = v.strip().strip("'").strip('"')
        result[k.strip()] = v_val
    return result

def load_all_runs(mlruns_dir_path: str) -> list[dict]:
    """Scans the MLflow mlruns directory and extracts metadata for all runs."""
    mlruns_dir = Path(mlruns_dir_path)
    if not mlruns_dir.exists():
        raise FileNotFoundError(f"Directory not found: {mlruns_dir_path}")

    records = []
    
    for exp_dir in mlruns_dir.iterdir():
        if not exp_dir.is_dir() or exp_dir.name.startswith("."):
            continue

        exp_meta_path = exp_dir / "meta.yaml"
        exp_meta = parse_meta_yaml(exp_meta_path)
        exp_id = exp_meta.get("experiment_id", exp_dir.name)
        exp_name = exp_meta.get("name", f"experiment-{exp_id}")

        for run_dir in exp_dir.iterdir():
            run_meta_path = run_dir / "meta.yaml"
            if not run_meta_path.exists():
                continue

            meta = parse_meta_yaml(run_meta_path)
            run_id = meta.get("run_id", run_dir.name)

            record = {
                "run_id": run_id,
                "experiment_id": exp_id,
                "experiment_name": exp_name,
                "status": meta.get("status"),
                "start_time": meta.get("start_time"),
                "end_time": meta.get("end_time"),
                "artifact_uri": meta.get("artifact_uri"),
                "lifecycle_stage": meta.get("lifecycle_stage"),
                "entry_point_name": meta.get("entry_point_name"),
                "user_id": meta.get("user_id"),
                "source_name": meta.get("source_name"),
                "source_type": meta.get("source_type"),
                "source_version": meta.get("source_version"),
                "run_name": meta.get("run_name"),
                "params": {},
                "metrics": {},
                "tags": {},
                "artifacts": [],
            }

            # Parse Params
            p_dir = run_dir / "params"
            if p_dir.is_dir():
                for f in p_dir.iterdir():
                    if f.is_file():
                        record["params"][f.name] = f.read_text().strip()

            # Parse Metrics
            m_dir = run_dir / "metrics"
            if m_dir.is_dir():
                for f in m_dir.iterdir():
                    if f.is_file():
                        # Handle MLflow metrics format which may have multiple parts (e.g timestamp, val, step)
                        parts = f.read_text().strip().split()
                        val = parts[1] if len(parts) >= 2 else parts[0]
                        record["metrics"][f.name] = val

            # Parse Tags
            t_dir = run_dir / "tags"
            if t_dir.is_dir():
                for f in t_dir.iterdir():
                    if f.is_file():
                        record["tags"][f.name] = f.read_text().strip()

            # Parse Artifacts
            a_dir = run_dir / "artifacts"
            if a_dir.is_dir():
                for f in a_dir.rglob("*"):
                    if f.is_file():
                        # Store relative path from artifacts directory
                        rel_path = str(f.relative_to(a_dir))
                        record["artifacts"].append(rel_path)

            # Default User
            if not record["user_id"]:
                record["user_id"] = record["tags"].get("mlflow.user")

            # Infer missing semantic tags from native MLflow params
            _infer_missing_tags(record)

            records.append(record)
            
    return records


def _infer_missing_tags(record: dict):
    """Infer mlsea.* semantic tags from native MLflow metadata when tags are missing.
    
    Inference priority: (1) existing tags, (2) native params, (3) experiment-name heuristics.
    """
    tags = record['tags']
    params = record['params']
    exp_name = record.get('experiment_name', '')
    all_keys = set(params.keys())
    
    # --- Algorithm ---
    if not tags.get('mlsea.algorithm'):
        algo = (params.get('algorithm') or params.get('model_name') or
                params.get('base_model') or params.get('clip_model') or'')
        if algo:
            tags['mlsea.algorithm'] = algo
        elif 'anchor_sizes' in params:
            tags['mlsea.algorithm'] = 'ObjectDetection'
        elif 'arima_order' in params:
            tags['mlsea.algorithm'] = 'ARIMA'
        elif 'hidden_size' in params and 'forecast_horizon' in params:
            tags['mlsea.algorithm'] = 'NBEATS'
        elif 'tfidf' in exp_name.lower():
            tags['mlsea.algorithm'] = 'TF-IDF'
        elif 'generative' in exp_name.lower() or 'ddpm' in exp_name or 'vae' in exp_name:
            tags['mlsea.algorithm'] = 'GenerativeModel'
    
    # --- Modality ---
    if not tags.get('mlsea.modality'):
        if 'image_width' in params or 'color_channels' in params:
            tags['mlsea.modality'] = 'image'
        elif 'clip_model' in params or ('image' in exp_name.lower() and 'text' in exp_name.lower()):
            tags['mlsea.modality'] = 'image+text'
        elif 'anchor_sizes' in params:
            tags['mlsea.modality'] = 'image'
        elif 'image' in exp_name.lower():
            tags['mlsea.modality'] = 'image'
        elif 'forecast_horizon' in params or 'time' in exp_name.lower():
            tags['mlsea.modality'] = 'time-series'
        elif 'nlp' in exp_name.lower() or 'lora' in exp_name.lower() or 'tfidf' in exp_name.lower():
            tags['mlsea.modality'] = 'text'
        elif 'fusion' in exp_name.lower() or 'genai' in exp_name.lower():
            tags['mlsea.modality'] = 'image+text'
        elif 'knowledge_distillation' in exp_name or 'distillation' in exp_name:
            tags['mlsea.modality'] = 'image'
    
    # --- Paradigm ---
    if not tags.get('mlsea.paradigm'):
        if 'distillation' in exp_name.lower() or 'distillation_temperature' in params:
            tags['mlsea.paradigm'] = 'knowledge-distillation'
        elif 'contrastive' in exp_name.lower() or 'simclr' in exp_name.lower() or 'contrastive_temperature' in params:
            tags['mlsea.paradigm'] = 'contrastive_embedding'
        elif 'detection' in exp_name.lower() or 'anchor_sizes' in params:
            tags['mlsea.paradigm'] = 'object-detection'
        elif 'lora' in exp_name.lower() or 'lora_rank' in params:
            tags['mlsea.paradigm'] = 'parameter-efficient-ft'
        elif 'forecasting' in exp_name.lower() or 'time' in exp_name.lower():
            tags['mlsea.paradigm'] = 'time_series_forecasting'
        elif 'fusion' in exp_name.lower() or 'multimodal' in exp_name.lower():
            tags['mlsea.paradigm'] = 'multimodal_fusion'
        elif 'genai' in exp_name.lower() or 'clip' in exp_name.lower():
            tags['mlsea.paradigm'] = 'vision-language-alignment'
        elif 'ssl' in exp_name.lower() or 'learning_stage' in params:
            tags['mlsea.paradigm'] = 'self-supervised'
        elif 'supervised' in exp_name.lower():
            tags['mlsea.paradigm'] = 'supervised-classification'
        elif 'generative' in exp_name.lower():
            tags['mlsea.paradigm'] = 'generative-diffusion'
        elif 'statistical' in exp_name.lower():
            tags['mlsea.paradigm'] = 'statistical'
        elif 'nbeats' in exp_name.lower():
            tags['mlsea.paradigm'] = 'neural-forecasting'
        elif 'lightgbm' in exp_name.lower() or 'gradient' in exp_name.lower():
            tags['mlsea.paradigm'] = 'gradient-boosting'
        elif 'vit' in exp_name.lower():
            tags['mlsea.paradigm'] = 'vision-transformer'
