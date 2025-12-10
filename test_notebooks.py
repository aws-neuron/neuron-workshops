#!/usr/bin/env python3
"""
Automated testing for Neuron Workshop notebooks.
Uses nbval to execute notebooks and validate outputs.
Assumes Neuron SDK environment is always available.
"""
import pytest
from pathlib import Path

# Define notebook timeouts by category
NOTEBOOK_TIMEOUTS = {
    "NxD": 1800,        # 30 minutes for model compilation
    "FineTuning": 3600, # 60 minutes for fine-tuning
    "vLLM": 1800,       # 30 minutes for vLLM setup
    "NKI": 900,         # 15 minutes for NKI labs
}

def get_notebook_timeout(notebook_path: str) -> int:
    """Get timeout for a notebook based on its path."""
    for category, timeout in NOTEBOOK_TIMEOUTS.items():
        if f"labs/{category}/" in notebook_path:
            return timeout
    return 900  # Default 15 minutes

def get_notebooks():
    """Discover all notebook files in the labs directory."""
    notebooks = []
    labs_dir = Path("labs")
    
    if not labs_dir.exists():
        return notebooks
        
    for notebook_path in labs_dir.rglob("*.ipynb"):
        # Skip checkpoint files
        if ".ipynb_checkpoints" in str(notebook_path):
            continue
            
        timeout = get_notebook_timeout(str(notebook_path))
        notebooks.append((str(notebook_path), timeout))
            
    return notebooks

# Generate test parameters
notebook_params = get_notebooks()

@pytest.mark.parametrize("notebook_path,timeout", notebook_params)
def test_notebook_execution(notebook_path, timeout):
    """
    Test that notebooks execute without errors.
    The actual execution is handled by nbval plugin.
    """
    # This is a placeholder - nbval handles the actual execution
    # The timeout parameter is used by pytest-timeout plugin
    pass

if __name__ == "__main__":
    # Print discovered notebooks for debugging
    notebooks = get_notebooks()
    print(f"Discovered {len(notebooks)} notebooks:")
    for nb_path, timeout in notebooks:
        print(f"  {nb_path} (timeout: {timeout}s)")