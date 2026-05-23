import argparse
from pathlib import Path
from . import SemanticTracker
from .validate.validator import validate_graph

def main():
    parser = argparse.ArgumentParser(description="Export MLflow tracking data to MLSO-aligned RDF.")
    parser.add_argument("--uri", type=str, required=True, help="Path to the mlruns directory.")
    parser.add_argument("--out", type=str, required=True, help="Path to save the output Turtle file.")
    parser.add_argument("--modality", type=str, default="tabular", choices=["tabular", "image", "nlp", "timeseries"], help="Default data modality hint.")
    parser.add_argument("--validate", action="store_true", help="Run SHACL validation after export.")
    parser.add_argument("--shapes", type=str, default=None, help="Optional comma-separated SHACL shape files or directories.")
    parser.add_argument("--validation-output", type=str, default=None, help="Directory for SHACL reports. Defaults to output file directory.")
    parser.add_argument("--validation-strict", action="store_true", help="Exit with non-zero code if SHACL does not conform.")
    parser.add_argument("--enable-sparql", action="store_true", help="Enable SHACL advanced/SPARQL constraints.")
    parser.add_argument("--inference", type=str, default="owlrl", choices=["none", "rdfs", "owlrl"], help="SHACL inference mode.")
    
    args = parser.parse_args()
    
    tracker = SemanticTracker(mlflow_uri=args.uri, default_modality=args.modality)
    print(f"Extraction started on MLflow tracking URI: {args.uri} [Modality: {args.modality}]")
    
    kg = tracker.export()
    
    print(f"Export successful. Serializing {len(kg)} triples...")
    kg.serialize(destination=args.out, format="turtle")
    print(f"Knowledge Graph saved to: {args.out}")

    if args.validate:
        out_path = Path(args.out).resolve()
        report_dir = Path(args.validation_output).resolve() if args.validation_output else out_path.parent

        if args.shapes:
            shapes_inputs = [s.strip() for s in args.shapes.split(",") if s.strip()]
        else:
            # Default to packaged canonical shapes directory.
            shapes_inputs = [Path(__file__).resolve().parent / "shapes"]

        shape_files = []
        for item in shapes_inputs:
            p = Path(item)
            if p.is_dir():
                shape_files.extend(sorted(str(x) for x in p.glob("*.ttl")))
            elif p.exists():
                shape_files.append(str(p))

        if not shape_files:
            raise SystemExit("No SHACL shape files found. Use --shapes to provide a valid file/dir.")

        print(f"Running SHACL validation using {len(shape_files)} shape file(s)...")
        report = validate_graph(
            kg,
            shapes_paths=shape_files,
            inference=args.inference,
            advanced=args.enable_sparql,
            report_dest=str(report_dir),
            abort_on_violation=args.validation_strict,
        )
        print(f"SHACL conforms: {report['conforms']}")
        print(f"SHACL reports saved to: {report_dir}")

        if args.validation_strict and not report["conforms"]:
            raise SystemExit(2)

if __name__ == "__main__":
    main()
