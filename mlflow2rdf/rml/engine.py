"""
RML Engine — MLflow → RDF using Morph-KGC with YARRRML mapping files.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import yaml

logger = logging.getLogger(__name__)


class RMLEngine:
    """RML-compliant transformation engine using Morph-KGC.

    Pipeline:
    1. Collect MLflow data → CSV files
    2. Load YARRRML mapping → RML via Morph-KGC
    3. Materialize RDF graph

    Usage:
        engine = RMLEngine(yarrrml_file='config/rml_mappings.yaml')
        graph = engine.transform()
        engine.save('output.ttl')
    """

    def __init__(self, yarrrml_file: str, config_dir: str = 'mlflow2rdf/config'):
        self.yarrrml_file = Path(yarrrml_file)
        self.config_dir = Path(config_dir)
        self.graph = None

    def transform(self) -> 'Graph':
        """Execute RML transformation via Morph-KGC.

        Returns:
            RDFlib Graph with materialized triples.
        """
        from morph_kgc import materialize

        morph_kgc_ini = self.config_dir / 'morph_kgc.ini'
        if not morph_kgc_ini.exists():
            raise FileNotFoundError(
                f"Morph-KGC config not found: {morph_kgc_ini}. "
                f"Create it with [CONFIGURATION] and a DataSource pointing to your YARRRML file."
            )

        logger.info(f"Starting RML transformation with Morph-KGC...")
        logger.info(f"  YARRRML: {self.yarrrml_file}")
        logger.info(f"  Config:  {morph_kgc_ini}")

        # morph-kgc materialize accepts the INI config file path
        self.graph = materialize(str(morph_kgc_ini))

        logger.info(f"RML transformation completed: {len(self.graph)} triples")
        return self.graph

    def save(self, output_path: str, fmt: str = 'turtle'):
        """Save the RDF graph to a file."""
        if self.graph is None:
            raise RuntimeError("No graph to save. Call transform() first.")

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.graph.serialize(destination=str(out), format=fmt)
        logger.info(f"RDF graph saved: {out} ({len(self.graph)} triples)")

    def get_stats(self) -> Dict:
        """Return transformation statistics."""
        if self.graph is None:
            return {'total_triples': 0}
        return {
            'total_triples': len(self.graph),
            'namespaces': list(self.graph.namespaces()),
            'yarrrml_file': str(self.yarrrml_file),
        }
