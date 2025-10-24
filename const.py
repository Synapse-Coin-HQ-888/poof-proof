import importlib
import pkgutil
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cpu':
    print("Running on CPU — performance may be limited.")

# Global registry for discovered modules and their Synapse mappings
synapse_modules = []
synapse_mappings = []

def discover_synapse_modules():
    """Scan and import all modules within the models/ directory and gather their __synapse__ mappings."""
    global synapse_modules, synapse_mappings

    # Import every module located under models/
    for _, name, _ in pkgutil.iter_modules(['models']):
        module = importlib.import_module(f'models.{name}')
        synapse_modules.append(module)

    # Extract all __synapse__ mappings from the loaded modules
    for module in synapse_modules:
        if hasattr(module, '__synapse__'):
            synapse_mappings.append(module.__synapse__)
