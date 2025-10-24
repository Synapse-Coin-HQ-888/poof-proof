import os
from pathlib import Path
from datetime import datetime
import torch
import json


def setup_experiment_directory(model_name: str) -> Path:
    """Create and return a new experiment directory for Synapse"""
    base_dir = Path("experiments") / model_name
    base_dir.mkdir(parents=True, exist_ok=True)

    # Determine next available experiment index
    existing_experiments = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("session_")]
    next_index = len(existing_experiments)

    # Timestamped session directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = base_dir / f"session_{next_index}_{timestamp}"
    session_dir.mkdir()

    return session_dir


def save_model_state(
    session_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int
):
    """Save model and optimizer states to Synapse session"""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step
    }

    checkpoint_path = session_dir / f"state_{step}.pt"
    torch.save(checkpoint, checkpoint_path)

    # Write reference to latest state
    with open(session_dir / "latest.txt", "w") as f:
        f.write(f"state_{step}.pt")


def load_model_state(
    session_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
) -> int:
    """Load latest Synapse checkpoint and return the current training step"""
    try:
        with open(session_dir / "latest.txt", "r") as f:
            latest = f.read().strip()

        checkpoint_path = session_dir / latest
        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return checkpoint["step"]

    except FileNotFoundError:
        return 0


def load_training_log(session_dir: Path) -> dict:
    """Load experiment history from a Synapse session directory"""
    try:
        with open(session_dir / "history.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
