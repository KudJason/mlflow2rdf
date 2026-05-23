from rdflib import Graph, URIRef
from rdflib.namespace import RDF, XSD

from ..config import MLSO, MLSEA, MLFLOW

def handle_image_metadata(graph: Graph, record: dict, run_uri: URIRef, dataset_uri: URIRef = None):
    """
    Extracts image-specific metadata from an MLflow record and maps it to MLSO RDF.
    Specifically handles dimensions, channels, and training curves.
    """
    params = record.get('params', {})
    tags = record.get('tags', {})

    # 1. Image Dataset Characteristics
    if dataset_uri:
        width = params.get('image_width') or tags.get('image_width')
        if width:
            graph.add((dataset_uri, MLSO.hasImageWidth, literal_from_value(width)))
            
        height = params.get('image_height') or tags.get('image_height')
        if height:
            graph.add((dataset_uri, MLSO.hasImageHeight, literal_from_value(height)))
            
        channels = params.get('color_channels') or tags.get('color_channels')
        if channels:
            graph.add((dataset_uri, MLSO.hasColorChannels, literal_from_value(channels)))

    # 2. Confusion Matrix Artifact Support
    # Artifacts will be thoroughly handled in a later module, but we can tag the run explicitly
    pass

from .base_mapper import literal_from_value
