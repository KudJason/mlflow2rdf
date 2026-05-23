from rdflib import Namespace
import re

# W3C and Domain Namespaces
MLS     = Namespace('http://www.w3.org/ns/mls#')
MLSO    = Namespace('http://w3id.org/mlso/')
MLSEA   = Namespace('http://w3id.org/mlsea/')

# Standard Namespaces
DCTERMS = Namespace('http://purl.org/dc/terms/')
PROV    = Namespace('http://www.w3.org/ns/prov#')
FOAF    = Namespace('http://xmlns.com/foaf/0.1/')
ADMS    = Namespace('http://www.w3.org/ns/adms#')
SKOS    = Namespace('http://www.w3.org/2004/02/skos/core#')
RDF     = Namespace('http://www.w3.org/1999/02/22-rdf-syntax-ns#')
RDFS    = Namespace('http://www.w3.org/2000/01/rdf-schema#')
XSD     = Namespace('http://www.w3.org/2001/XMLSchema#')

# MLSO Taxonomy Namespaces
MLST_TASK = Namespace('http://w3id.org/mlso/vocab/ml_task_type/')
MLST_ALGO = Namespace('http://w3id.org/mlso/vocab/ml_algorithm/')
MLSO_EVAL = Namespace('http://w3id.org/mlso/vocab/evaluation_measure/')
MLSO_DC = Namespace('http://w3id.org/mlso/vocab/dataset_characteristic/')

# Local Generated Namespaces
MLFLOW  = Namespace('http://w3id.org/mlsea/mlflow/')
NS1     = Namespace('http://w3id.org/mlsea/tag/')

# Graph URIs
DATASET_GRAPH_URI = 'http://w3id.org/mlsea/mlflow/dataset'
RUNS_GRAPH_URI    = 'http://w3id.org/mlsea/mlflow/runs'

def slugify(value: str, fallback: str = 'value') -> str:
    """Takes a string and converts it to a clean URI-safe slug."""
    if not value: return fallback
    value = str(value).lower()
    value = re.sub(r'[^a-z0-9]+', '-', value)
    return value.strip('-') or fallback
