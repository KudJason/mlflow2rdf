from setuptools import setup, find_packages

setup(
    name="mlflow2rdf",
    version="0.2.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "mlflow2rdf": ["shapes/*.ttl"],
    },
    install_requires=[
        "mlflow>=2.0.0",
        "rdflib>=6.0.0",
        "pyshacl>=0.25.0",
    ],
    entry_points={
        "console_scripts": [
            "mlflow2rdf = mlflow2rdf.cli:main",
        ]
    },
    author="Master Thesis Project",
    description="A Python package to convert MLflow tracking data into MLSO-aligned RDF Knowledge Graphs.",
    python_requires=">=3.8",
)
