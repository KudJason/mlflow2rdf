"""
MLflow → RDF Converter (RML Standard)
Main orchestration: collect CSV data → run Morph-KGC YARRRML mapping → save RDF.
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class MLflowToRDFConverterRML:
    """MLflow to RDF converter using RML standard via Morph-KGC.

    Pipeline:
        1. Collect MLflow data from local mlruns/ directory → CSV files
        2. Execute YARRRML → RML mapping via Morph-KGC
        3. Save output RDF graph

    Args:
        mlflow_uri: Path to mlruns directory.
        config_dir: Directory with morph_kgc.ini and YARRRML files.
        output_dir: Directory for CSV data and RDF output.
    """

    def __init__(self, mlflow_uri: str, config_dir: str = 'mlflow2rdf/config',
                 output_dir: str = 'data'):
        self.mlflow_uri = mlflow_uri
        self.config_dir = Path(config_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def convert(self) -> 'Graph':
        """Run the full conversion pipeline.

        Returns:
            RDFlib Graph with materialized triples.
        """
        logger.info("=" * 60)
        logger.info("MLflow → RDF Conversion (RML Standard / Morph-KGC)")
        logger.info("=" * 60)

        start_time = datetime.now()

        # Step 1: Collect MLflow data to CSV
        logger.info("\n[Step 1/2] Collecting MLflow data...")
        self._collect_data()

        # Step 2: Execute RML transformation
        logger.info("\n[Step 2/2] Executing RML transformation via Morph-KGC...")
        from .engine import RMLEngine
        engine = RMLEngine(
            yarrrml_file=str(self.config_dir / 'rml_mappings.yaml'),
            config_dir=str(self.config_dir),
        )
        graph = engine.transform()

        # Save output
        output_file = self.output_dir / 'output.ttl'
        engine.save(str(output_file))

        duration = (datetime.now() - start_time).total_seconds()
        stats = engine.get_stats()

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Conversion Completed!")
        logger.info(f"{'=' * 60}")
        logger.info(f"Total triples: {stats['total_triples']}")
        logger.info(f"Output file:   {output_file}")
        logger.info(f"Duration:      {duration:.2f}s")
        logger.info(f"{'=' * 60}")

        return graph

    def _collect_data(self):
        """Collect MLflow data and export to CSV files."""
        from ..rml_data_collector import MLflowDataCollector
        collector = MLflowDataCollector(
            mlflow_uri=self.mlflow_uri,
            output_dir=str(self.output_dir),
        )
        collector.collect_all()


def main():
    """CLI entry point for RML-based conversion."""
    parser = argparse.ArgumentParser(
        description='MLflow to RDF Converter (RML Standard via Morph-KGC)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion from local mlruns directory
  python -m mlflow2rdf.rml.converter /path/to/mlruns

  # Use custom config directory
  python -m mlflow2rdf.rml.converter /path/to/mlruns --config-dir config
        """,
    )
    parser.add_argument('mlflow_uri', type=str, help='Path to mlruns directory')
    parser.add_argument('--config-dir', type=str, default='mlflow2rdf/config',
                        help='Configuration directory')
    parser.add_argument('--output-dir', type=str, default='data',
                        help='Output directory')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    converter = MLflowToRDFConverterRML(
        mlflow_uri=args.mlflow_uri,
        config_dir=args.config_dir,
        output_dir=args.output_dir,
    )
    converter.convert()


if __name__ == '__main__':
    main()
