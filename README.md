# mlflow2rdf

A Python package to convert MLflow tracking data into MLSO-aligned RDF Knowledge Graphs.

## Overview

`mlflow2rdf` transforms your MLflow experiment metadata (parameters, metrics, tags) into semantic RDF triples that conform to the [MLSO (Machine Learning Sailor Ontology)](https://github.com/dtai-kg/MLSO) standard. This enables:

- **Semantic querying** via SPARQL instead of imperative MLflow API loops
- **Cross-platform interoperability** with public ML knowledge graphs like MLSea
- **FAIR principles** for ML experiments: Findable, Accessible, Interoperable, Reusable

## Installation

```bash
pip install mlflow2rdf
```

## Quick Start

### Command Line Interface

```bash
# Convert MLflow runs to RDF
mlflow2rdf --mlruns /path/to/mlruns --output output.ttl

# With SHACL validation
mlflow2rdf --mlruns /path/to/mlruns --output output.ttl --validate
```

### Python API

```python
from mlflow2rdf import MLflow2RDFConverter

# Initialize converter
converter = MLflow2RDFConverter(mlruns_path="/path/to/mlruns")

# Convert to RDF
graph = converter.convert()

# Serialize to Turtle format
graph.serialize("output.ttl", format="turtle")

# Validate with SHACL
results = converter.validate(graph)
print(f"SHACL violations: {results}")
```

## Features

- **Multi-modal support**: Tabular, image, text, time-series, and multi-modal data
- **Paradigm-aware routing**: Automatic parameter routing based on learning paradigm
- **Pipeline relationship inference**: Detects knowledge distillation, LoRA adapters, self-supervised learning chains
- **SHACL validation**: Comprehensive shape validation against MLSO constraints
- **Blind spot analysis**: Completeness verification of metadata extraction

## Supported Learning Paradigms

| Paradigm | Key Properties |
|----------|---------------|
| Supervised Classification | Standard hyperparameters, accuracy metrics |
| Self-Supervised Learning | Pre-text/downstream run partitioning |
| Contrastive Learning | Temperature, distance metrics |
| Knowledge Distillation | Teacher-student relationships, distillation temperature |
| Parameter-Efficient Fine-tuning (LoRA) | Adapter rank, alpha, target modules |
| Multi-Modal Fusion | Fusion strategy, image/text encoders |
| Time-Series Forecasting | Forecasting horizon, lookback window |

## Output Format

The package generates RDF triples in Turtle format, using MLSO/MLST vocabulary:

```turtle
@prefix mls: <http://www.w3.org/ns/mls#> .
@prefix mlso: <http://w3id.org/mlso/> .

<run/abc123> a mls:Run ;
    mls:hasInput <dataset/cifar10> ;
    mls:hasOutput <evaluation/acc_0.95> ;
    mlso:hasParadigm "Supervised Classification" .
```

## Requirements

- Python >= 3.8
- MLflow >= 2.0.0
- RDFLib >= 6.0.0
- pySHACL >= 0.25.0

## License

MIT License

## Citation

If you use this package in your research, please cite:

```bibtex
@mastersthesis{jia2026mlflow2rdf,
  author = {Jia, Sijie},
  title = {Bridging ML Tracking and Semantic Interoperability: Transforming MLflow Experiment Metadata to MLSO-Aligned RDF Knowledge Graphs},
  school = {KU Leuven},
  year = {2026}
}
```

## Links

- [MLSO Ontology](https://github.com/dtai-kg/MLSO)
- [MLSea Knowledge Graph](https://mlsea.ai)
- [Issue Tracker](https://github.com/KudJason/mlflow2rdf/issues)
