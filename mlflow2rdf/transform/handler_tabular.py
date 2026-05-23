"""
Tabular Modality Handler — adds tabular dataset characteristics.

Adds hasRowCount, hasFeatureCount, and related dataset properties
for tabular ML experiments.
"""

from rdflib import Graph, URIRef
from rdflib.namespace import RDF
from ..config import MLSO, MLSO_DC, MLS
from ..transform.base_mapper import literal_from_value


def handle_tabular_metadata(graph: Graph, record: dict, dataset_uri: URIRef = None):
    """Add tabular-specific dataset characteristics.

    Attaches hasRowCount, hasFeatureCount, hasMissingValues
    properties from params/tags to the dataset node.
    """
    params = record.get('params', {})
    tags = record.get('tags', {})
    if dataset_uri is None:
        return

    rows = params.get('rows') or tags.get('rows')
    if rows:
        lit = literal_from_value(rows)
        if lit is not None:
            graph.add((dataset_uri, MLSO.hasRowCount, lit))
            char_uri = URIRef(f'{str(dataset_uri)}/characteristic/number_of_instances')
            graph.add((char_uri, RDF.type, MLS.DatasetCharacteristic))
            graph.add((char_uri, MLSO.hasDataCharacteristicType,
                       MLSO_DC.number_of_instances))
            graph.add((char_uri, MLS.hasValue, lit))

    columns = params.get('columns') or tags.get('columns') or tags.get('feature_count')
    if columns:
        lit = literal_from_value(columns)
        if lit is not None:
            graph.add((dataset_uri, MLSO.hasFeatureCount, lit))
