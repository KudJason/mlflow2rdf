"""
MLflow Data Collector
Collects MLflow data and exports to CSV files for RML processing
"""

import csv
import mlflow
from mlflow.tracking import MlflowClient
from typing import List, Dict, Any
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLflowDataCollector:
    """Collects MLflow data and exports to CSV"""
    
    def __init__(self, mlflow_uri: str, output_dir: str = "data"):
        """
        Initialize data collector
        
        Args:
            mlflow_uri: MLflow tracking server URI
            output_dir: Output directory for CSV files
        """
        self.mlflow_client = MlflowClient(mlflow_uri)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def collect_all(self):
        """Collect all MLflow data"""
        logger.info("Starting MLflow data collection...")
        
        # Collect experiments
        self.collect_experiments()
        
        # Collect runs
        self.collect_runs()
        
        # Collect parameters
        self.collect_params()
        
        # Collect metrics
        self.collect_metrics()
        
        # Collect tags
        self.collect_tags()
        
        logger.info("Data collection completed!")
        
    def collect_experiments(self):
        """Collect experiment data"""
        logger.info("Collecting experiments...")
        
        experiments = self.mlflow_client.search_experiments()
        
        csv_file = self.output_dir / "mlflow_experiments.csv"
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'experiment_id',
                'name',
                'artifact_location',
                'lifecycle_stage',
                'creation_time',
                'last_update_time'
            ])
            
            # Write data
            for exp in experiments:
                writer.writerow([
                    exp.experiment_id,
                    exp.name,
                    exp.artifact_location,
                    exp.lifecycle_stage,
                    exp.creation_time,
                    exp.last_update_time
                ])
                
        logger.info(f"Collected {len(experiments)} experiments to {csv_file}")
        
    def collect_runs(self):
        """Collect run data"""
        logger.info("Collecting runs...")
        
        # Get all experiments
        experiments = self.mlflow_client.search_experiments()
        
        all_runs = []
        
        for exp in experiments:
            runs = self.mlflow_client.search_runs(
                experiment_ids=[exp.experiment_id],
                max_results=10000
            )
            
            for run in runs:
                # Extract run name from tags
                run_name = run.data.tags.get('mlflow.runName', '')
                
                all_runs.append({
                    'run_id': run.info.run_id,
                    'experiment_id': run.info.experiment_id,
                    'status': run.info.status,
                    'start_time': run.info.start_time,
                    'end_time': run.info.end_time if run.info.end_time else '',
                    'user_id': run.info.user_id if run.info.user_id else '',
                    'run_name': run_name
                })
                
        csv_file = self.output_dir / "mlflow_runs.csv"
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'run_id',
                'experiment_id',
                'status',
                'start_time',
                'end_time',
                'user_id',
                'run_name'
            ])
            
            # Write data
            for run in all_runs:
                writer.writerow([
                    run['run_id'],
                    run['experiment_id'],
                    run['status'],
                    run['start_time'],
                    run['end_time'],
                    run['user_id'],
                    run['run_name']
                ])
                
        logger.info(f"Collected {len(all_runs)} runs to {csv_file}")
        
    def collect_params(self):
        """Collect parameter data"""
        logger.info("Collecting parameters...")
        
        # Get all experiments
        experiments = self.mlflow_client.search_experiments()
        
        all_params = []
        
        for exp in experiments:
            runs = self.mlflow_client.search_runs(
                experiment_ids=[exp.experiment_id],
                max_results=10000
            )
            
            for run in runs:
                for key, value in run.data.params.items():
                    all_params.append({
                        'run_id': run.info.run_id,
                        'param_key': key,
                        'param_value': value
                    })
                    
        csv_file = self.output_dir / "mlflow_params.csv"
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'run_id',
                'param_key',
                'param_value'
            ])
            
            # Write data
            for param in all_params:
                writer.writerow([
                    param['run_id'],
                    param['param_key'],
                    param['param_value']
                ])
                
        logger.info(f"Collected {len(all_params)} parameters to {csv_file}")
        
    def collect_metrics(self):
        """Collect metric data"""
        logger.info("Collecting metrics...")
        
        # Get all experiments
        experiments = self.mlflow_client.search_experiments()
        
        all_metrics = []
        
        for exp in experiments:
            runs = self.mlflow_client.search_runs(
                experiment_ids=[exp.experiment_id],
                max_results=10000
            )
            
            for run in runs:
                for key, value in run.data.metrics.items():
                    all_metrics.append({
                        'run_id': run.info.run_id,
                        'metric_key': key,
                        'metric_value': value,
                        'timestamp': run.info.start_time,
                        'step': 0  # MLflow doesn't store step in run summary
                    })
                    
        csv_file = self.output_dir / "mlflow_metrics.csv"
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'run_id',
                'metric_key',
                'metric_value',
                'timestamp',
                'step'
            ])
            
            # Write data
            for metric in all_metrics:
                writer.writerow([
                    metric['run_id'],
                    metric['metric_key'],
                    metric['metric_value'],
                    metric['timestamp'],
                    metric['step']
                ])
                
        logger.info(f"Collected {len(all_metrics)} metrics to {csv_file}")
        
    def collect_tags(self):
        """Collect tag data"""
        logger.info("Collecting tags...")
        
        # Get all experiments
        experiments = self.mlflow_client.search_experiments()
        
        all_tags = []
        
        for exp in experiments:
            runs = self.mlflow_client.search_runs(
                experiment_ids=[exp.experiment_id],
                max_results=10000
            )
            
            for run in runs:
                for key, value in run.data.tags.items():
                    # Skip mlflow internal tags
                    if not key.startswith('mlflow.'):
                        all_tags.append({
                            'run_id': run.info.run_id,
                            'tag_key': key,
                            'tag_value': value
                        })
                    
        csv_file = self.output_dir / "mlflow_tags.csv"
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'run_id',
                'tag_key',
                'tag_value'
            ])
            
            # Write data
            for tag in all_tags:
                writer.writerow([
                    tag['run_id'],
                    tag['tag_key'],
                    tag['tag_value']
                ])
                
        logger.info(f"Collected {len(all_tags)} tags to {csv_file}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Collect MLflow data and export to CSV'
    )
    
    parser.add_argument(
        '--mlflow-uri',
        type=str,
        default='http://localhost:5000',
        help='MLflow tracking server URI'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data',
        help='Output directory for CSV files'
    )
    
    args = parser.parse_args()
    
    # Create collector
    collector = MLflowDataCollector(
        mlflow_uri=args.mlflow_uri,
        output_dir=args.output_dir
    )
    
    # Collect data
    collector.collect_all()


if __name__ == '__main__':
    main()
