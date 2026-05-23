from rdflib import Graph, URIRef
from rdflib.namespace import RDF, XSD

from ..config import MLSO, MLSEA, MLFLOW

def handle_nlp_metadata(graph: Graph, record: dict, run_uri: URIRef, dataset_uri: URIRef = None):
    """
    Extracts NLP-specific metadata from an MLflow record and maps it to MLSO RDF.
    Specifically handles vocabulary size, maximum sequence lengths.
    """
    params = record.get('params', {})
    tags = record.get('tags', {})

    # 1. Text Dataset Characteristics
    if dataset_uri:
        vocab_size = params.get('vocab_size') or tags.get('vocab_size')
        if vocab_size:
            graph.add((dataset_uri, MLSO.hasVocabularySize, literal_from_value(vocab_size)))
            
        max_seq_len = params.get('max_sequence_length') or tags.get('max_sequence_length')
        if max_seq_len:
            graph.add((dataset_uri, MLSO.hasMaxSequenceLength, literal_from_value(max_seq_len)))

from .base_mapper import literal_from_value
