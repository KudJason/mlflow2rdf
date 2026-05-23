from rdflib import Graph
from .extract.reader import load_all_runs
from .transform.base_mapper import build_dataset_graph, build_runs_graph_skeleton
from .transform.handler_image import handle_image_metadata
from .transform.handler_nlp import handle_nlp_metadata
from .transform.handler_timeseries import handle_timeseries_metadata

class SemanticTracker:
    def __init__(self, mlflow_uri: str, output_graph_uri: str = "http://w3id.org/mlsea/mlflow/kg", default_modality: str = None):
        if not mlflow_uri.startswith("file://") and not mlflow_uri.startswith("/"):
            # Assume local directory if not explicit file URI
            self.mlflow_dir = mlflow_uri
        else:
            self.mlflow_dir = mlflow_uri.replace("file://", "")
            
        self.output_graph_uri = output_graph_uri
        self.default_modality = default_modality

    def export(self, experiment_ids: list = None, exclude_failed_runs: bool = False) -> Graph:
        """
        Reads MLflow tracking data, applies MLSO ontology mapping,
        and returns the combined RDF Graph.
        """
        # 1. Extraction
        records = load_all_runs(self.mlflow_dir)
        
        # Filter if needed
        if experiment_ids:
            records = [r for r in records if r.get('experiment_id') in experiment_ids]
        if exclude_failed_runs:
            records = [r for r in records if r.get('status') != 'FAILED']

        # 2. Base Transformation (Dataset & Runs Skeleton)
        dataset_graph, dataset_uris, processing_uris = build_dataset_graph(records)
        runs_graph = build_runs_graph_skeleton(records, dataset_uris, processing_uris)

        # 3. Modality-Specific Overlays
        for record in records:
            run_id = record['run_id']
            # Determine modality (simple heuristic based on default or tags)
            modality = self.default_modality or record.get('tags', {}).get('mlsea.modality', 'tabular')
            
            run_uri_str = f"http://w3id.org/mlsea/mlflow/run/{run_id}"
            
            # Find associated dataset URI for this run
            dataset_name = record.get('tags', {}).get('mlsea.dataset')
            ds_uri_obj = None
            if dataset_name and dataset_name in dataset_uris:
                ds_uri_obj = dataset_uris[dataset_name]

            # Apply specific handler logic directly to the dataset/runs graphs based on modality
            from rdflib import URIRef
            run_uri_obj = URIRef(run_uri_str)

            if modality == "image":
                handle_image_metadata(dataset_graph, record, run_uri_obj, ds_uri_obj)
            elif modality == "nlp":
                handle_nlp_metadata(dataset_graph, record, run_uri_obj, ds_uri_obj)
            elif modality == "timeseries":
                handle_timeseries_metadata(dataset_graph, record, run_uri_obj, ds_uri_obj)

        # 4. Combine Graphs
        final_graph = dataset_graph + runs_graph
        return final_graph


def export_and_validate(mlflow_uri: str, out_path: str = None, *, shapes_paths=None, ont_paths=None, inference: str = 'owlrl', advanced: bool = False, report_dest: str = None, **export_kwargs):
    """Helper to export RDF and run SHACL validation.

    Returns tuple (graph, validation_report_dict)
    """
    tracker = SemanticTracker(mlflow_uri=mlflow_uri, default_modality=export_kwargs.get('default_modality'))
    g = tracker.export(**{k: export_kwargs.get(k) for k in ('experiment_ids','exclude_failed_runs') if k in export_kwargs})

    # Lazy import to avoid adding pyshacl as mandatory import at module import time
    try:
        from .validate.validator import validate_graph
    except Exception:
        validate_graph = None

    report = None
    if validate_graph:
        report = validate_graph(g, shapes_paths=shapes_paths, ont_paths=ont_paths, inference=inference, advanced=advanced, report_dest=report_dest)

    if out_path:
        g.serialize(destination=out_path, format='turtle')

    return g, report
