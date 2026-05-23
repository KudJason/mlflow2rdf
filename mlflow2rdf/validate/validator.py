from pathlib import Path
import json
from rdflib import Graph
from rdflib import Namespace
from pyshacl import validate

def _guess_format(p: Path):
    return 'turtle' if p.suffix in {'.ttl', '.turtle'} else 'turtle'

def _load_graph(paths):
    g = Graph()
    for p in paths:
        p = Path(p)
        if p.exists():
            g.parse(p.as_posix(), format=_guess_format(p))
        else:
            # skip missing
            continue
    return g

def validate_graph(graph: Graph, shapes_paths=None, ont_paths=None, *, inference='owlrl', advanced=False, debug=False, abort_on_violation=False, report_dest: str = None):
    """Validate an rdflib.Graph with SHACL shapes.

    Args:
        graph: rdflib.Graph to validate.
        shapes_paths: path or list of paths to shapes (ttl). If None, caller should provide.
        ont_paths: optional ontology files to pass as ont_graph.
        inference: 'none'|'rdfs'|'owlrl'
        advanced: whether to enable advanced checks (sh:sparql)
        debug: pyshacl debug
        abort_on_violation: if True, raise RuntimeError on non-conformance
        report_dest: directory to write report files (report.json, results.ttl, results.txt)

    Returns:
        dict with keys: conforms (bool), results_graph (rdflib.Graph), results_text (str), report_json (dict)
    """
    # Prepare shapes graph
    if shapes_paths is None:
        shapes_g = None
    else:
        if isinstance(shapes_paths, (list, tuple)):
            shapes_g = _load_graph(shapes_paths)
        else:
            shapes_g = _load_graph([shapes_paths])

    # Prepare ontology graph
    ont_g = None
    if ont_paths:
        if isinstance(ont_paths, (list, tuple)):
            ont_g = _load_graph(ont_paths)
        else:
            ont_g = _load_graph([ont_paths])

    if shapes_g is not None and not advanced:
        # Drop SHACL SPARQL constraints for faster/default-safe mode.
        sh = Namespace("http://www.w3.org/ns/shacl#")
        for s, _, sparql_node in list(shapes_g.triples((None, sh.sparql, None))):
            for t in list(shapes_g.triples((sparql_node, None, None))):
                shapes_g.remove(t)
            shapes_g.remove((s, sh.sparql, sparql_node))

    conforms, results_graph, results_text = validate(
        graph,
        shacl_graph=shapes_g,
        ont_graph=ont_g,
        inference=inference if inference in ('none','rdfs','owlrl') else 'owlrl',
        advanced=advanced,
        debug=debug
    )

    sh = Namespace("http://www.w3.org/ns/shacl#")
    violations = list(results_graph.triples((None, sh.resultSeverity, sh.Violation)))
    conforms_strict = bool(conforms) and len(violations) == 0

    report = {"conforms": bool(conforms), "conforms_strict": conforms_strict, "violation_count": len(violations)}

    if report_dest:
        d = Path(report_dest)
        d.mkdir(parents=True, exist_ok=True)
        (d / 'results.txt').write_text(results_text, encoding='utf-8')
        results_graph.serialize((d / 'results.ttl').as_posix(), format='turtle')
        (d / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

    if abort_on_violation and not conforms:
        raise RuntimeError('SHACL validation failed')

    return {"conforms": bool(conforms), "results_graph": results_graph, "results_text": results_text, "report": report}
