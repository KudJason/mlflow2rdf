from rdflib import Graph, URIRef
from rdflib.namespace import RDF, XSD

from ..config import MLSO, MLSEA, MLFLOW

def handle_timeseries_metadata(graph: Graph, record: dict, run_uri: URIRef, dataset_uri: URIRef = None):
    """
    Extracts Time-Series specific metadata from an MLflow record and maps it to MLSO RDF.
    Specifically handles forecast horizon, frequency, and seasonality.
    """
    params = record.get('params', {})
    tags = record.get('tags', {})

    # 1. Time-Series Dataset Characteristics
    if dataset_uri:
        horizon = params.get('forecast_horizon') or tags.get('forecast_horizon')
        if horizon:
            graph.add((dataset_uri, MLSO.hasForecastingHorizon, literal_from_value(horizon)))

        frequency = params.get('frequency') or tags.get('frequency')
        if frequency:
            graph.add((dataset_uri, MLSO.hasFrequency, literal_from_value(frequency)))

        seasonality = params.get('seasonality_period') or tags.get('seasonality_period')
        if seasonality:
            graph.add((dataset_uri, MLSO.hasSeasonality, literal_from_value(seasonality)))

from .base_mapper import literal_from_value
