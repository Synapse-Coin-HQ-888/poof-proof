import json
import time
from pathlib import Path
from typing import Dict, Optional

from utils.logging import setup_run_directory

class SynapseTrainingManager:
    """Coordinates and tracks Synapse model training runs, checkpoints, and metrics."""

    def __init__(self):
        self.metrics: Dict[str, list] = {
            'loss':   [],
            'elapsed': [],
            'steps':  [],
            'tokens': []
        }
        self.active_run_dir: Optional[Path] = None
        self.global_steps = 0
        self.global_tokens = 0
        self.start_timestamp = time.time()

    def log_metrics(self, metrics: dict):
        """Log a single set of metrics during training."""
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)

        self.global_steps += 1
        self.global_tokens += metrics.get('batch_size', 0)

        # Update timestamps and counters
        self.metrics['elapsed'].append(time.time() - self.start_timestamp)
        self.metrics['steps'].append(self.global_steps)
        self.metrics['tokens'].append(self.global_tokens)

        # Persist metric log to disk
        if self.active_run_dir:
            with open(self.active_run_dir / 'metrics.json', 'w') as f:
                json.dump(self.metrics, f, indent=2)

    def begin_new_run(self, model_label: str, configuration: dict):
        """Initialize a new Synapse training session."""
        self.active_run_dir = setup_run_directory(model_label)
        self.metrics.clear()
        self.global_steps = 0
        self.global_tokens = 0
        self.start_timestamp = time.time()

        # Save configuration snapshot
        with open(self.active_run_dir / 'config.json', 'w') as f:
            json.dump(configuration, f, indent=2)

    def resume_run(self, run_path: Path):
        """Resume or reload an existing Synapse training session."""
        self.active_run_dir = run_path

        try:
            with open(run_path / 'metrics.json', 'r') as f:
                self.metrics = json.load(f)

            # Restore counters
            self.global_steps = len(self.metrics.get('steps', []))
            self.global_tokens = self.metrics['tokens'][-1] if self.metrics.get('tokens') else 0

        except FileNotFoundError:
            print(f"No previous metrics found in {run_path}, initializing a new record.")
            self.metrics.clear()
            self.global_steps = 0
            self.global_tokens = 0

        self.start_timestamp = time.time() - (self.metrics['elapsed'][-1] if self.metrics.get('elapsed') else 0)
