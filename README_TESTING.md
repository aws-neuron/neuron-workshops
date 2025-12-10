# Notebook Testing Setup

This repository includes comprehensive testing for all Jupyter notebooks to ensure they work correctly after SDK updates and changes.

## Quick Start

```bash
# Clone repository
git clone https://github.com/aws-neuron/build-on-trainium-workshop.git neuron-workshops
cd neuron-workshops

# Run all tests
./run_tests.sh

# Run specific notebook
./run_tests.sh --notebook "vLLM/Chess/Chess-Deployment.ipynb"

# Run with HTML report
./run_tests.sh --html-report
```

## What Gets Tested

- **All notebooks execute without errors**
- **Cell outputs match expected results** (with nbval)
- **Import statements work correctly**
- **File paths resolve properly**
- **Model downloads and compilation succeed**
- **Hardware resources are available**

## Test Categories

| Category | Timeout | Description |
|----------|---------|-------------|
| NxD | 30 min | Model compilation and inference |
| FineTuning | 60 min | Model training and fine-tuning |
| vLLM | 30 min | Server setup and deployment |
| NKI | 15 min | Kernel development |

## Environment

- **Assumes**: Neuron SDK always available at `/opt/aws_neuronx_venv_pytorch_2_8_nxd_inference/`
- **Requires**: Trainium instance (trn1.2xlarge+)
- **Uses**: nbval + pytest for execution and validation

## Files

- `run_tests.sh` - Main test runner script
- `test_notebooks.py` - Test discovery and configuration  
- `pytest.ini` - Pytest configuration
- `requirements-test.txt` - Test dependencies
- `TESTING.md` - Detailed testing guide

## CI/CD

GitHub Actions automatically:
- Validates notebook syntax on PRs
- Runs full execution tests on pushes
- Runs weekly regression tests

See `TESTING.md` for complete documentation.