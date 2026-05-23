"""
Taxonomy Enrichment — adds SKOS concept scheme structure to the RDF graph.

Creates concept scheme nodes for paradigm types, task types, and
evaluation measures, enabling semantic querying and cross-system alignment.
"""

from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF, RDFS, SKOS
from ..config import MLS, MLSO, MLFLOW, slugify


def enrich_concept_schemes(graph: Graph):
    """Add top-level ConceptScheme nodes for ML paradigms and task types.

    Creates:
    - mlsea:concept/paradigm → skos:ConceptScheme
    - mlsea:concept/task-type → skos:ConceptScheme
    """
    paradigm_scheme = URIRef('http://w3id.org/mlsea/concept/paradigm')
    graph.add((paradigm_scheme, RDF.type, SKOS.ConceptScheme))
    graph.add((paradigm_scheme, RDFS.label, Literal('ML Paradigm Types', lang='en')))

    task_scheme = URIRef('http://w3id.org/mlsea/concept/task-type')
    graph.add((task_scheme, RDF.type, SKOS.ConceptScheme))
    graph.add((task_scheme, RDFS.label, Literal('ML Task Types', lang='en')))
