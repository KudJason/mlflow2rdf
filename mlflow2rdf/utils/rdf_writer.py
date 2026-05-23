"""
RDF Writer — serializes RDF graphs to file with configurable format.

Wraps rdflib serialization with validation stats and format detection.
"""

from pathlib import Path
from rdflib import Graph
from typing import Optional


def write_graph(graph: Graph, path: str, fmt: Optional[str] = None) -> dict:
    """Serialize an RDF graph to a file.

    Args:
        graph: RDF graph to serialize.
        path: Output file path.
        fmt: Serialization format (turtle/n3/xml/json-ld).
              If None, inferred from file extension.

    Returns:
        Dict with keys: 'path', 'format', 'triples', 'bytes'.
    """
    if fmt is None:
        ext = Path(path).suffix.lower()
        fmt = {
            '.ttl': 'turtle',
            '.n3': 'n3',
            '.rdf': 'xml',
            '.xml': 'xml',
            '.jsonld': 'json-ld',
            '.json': 'json-ld',
        }.get(ext, 'turtle')

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = graph.serialize(format=fmt)
    out.write_text(payload, encoding='utf-8')

    return {
        'path': str(out),
        'format': fmt,
        'triples': len(graph),
        'bytes': len(payload.encode('utf-8')),
    }


def write_stats(graph: Graph) -> dict:
    """Return quick statistics about an RDF graph."""
    namespaces = list(graph.namespaces())
    subjects = len(set(graph.subjects()))
    predicates = len(set(graph.predicates()))
    objects = len(set(graph.objects()))

    return {
        'triples': len(graph),
        'subjects': subjects,
        'predicates': predicates,
        'objects': objects,
        'namespaces': [str(p) for p, _ in namespaces],
    }
