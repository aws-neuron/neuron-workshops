# Notebook Testing Guide

This repository includes automated testing for all Jupyter notebooks to ensure they work correctly after SDK updates and changes.

## Overview

The testing system uses:
- **nbval**: A pytest plugin that executes notebooks and validates outputs
- **pytest**: Test framework with parallel execution support
- **papermill**: Alternative execution engine for notebooks
- **GitHub Actions**: Automated testing on code changes

## Quick Start

### Prerequisites

1. **AWS Trainium Instance**: trn1.2xlarge or larger
2. **Neuron SDK**: Pre-installed with virtual environment at `/opt/aws_neuronx_venv_pytorch_2_8_nxd_inference/`

```bash
git clone https://github.com/aws-neuron/build-on-trainium-workshop.git
cd build-on-trainium-workshop
```

### Running Tests

#### Run All Notebooks
```bash
./run_tests.sh
```

#### Run Specific Notebook
```bash
./run_tests.sh --notebook "vLLM/Chess/Chess-Deployment.ipynb"
```

#### Run with Options
```bash
# Fast mode (stop on first failure)
./run_tests.sh --fast

# Generate HTML report
./run_tests.sh --html-report

# Verbose output
./run_tests.sh --verbose
```

#### Run Specific Lab Category
```bash
# Test only NxD notebooks
pytest labs/NxD/ --nbval

# Test only Fine-tuning notebooks  
pytest labs/FineTuning/ --nbval

# Test only vLLM notebooks
pytest labs/vLLM/ --nbval
```

## Test Configuration

### Timeouts

Different notebook categories have different timeout settings:

- **NxD Labs**: 30 minutes (model compilation)
- **Fine-tuning Labs**: 60 minutes (training time)
- **vLLM Labs**: 30 minutes (server setup)
- **NKI Labs**: 15 minutes (kernel development)

### Environment Requirements

Each notebook category requires:
- Neuron SDK virtual environment
- Specific working directory
- Hardware resources (Neuron cores)

## Understanding Test Results

### Success ✅
```
labs/vLLM/Chess/Chess-Deployment.ipynb::test_notebook PASSED
```

### Failure ❌
```
labs/NxD/Lab_One_NxDI.ipynb::test_notebook FAILED
```

### Skipped ⏭️
```
labs/FineTuning/Finetune-Qwen3-1.7B.ipynb::test_notebook SKIPPED
```
*Usually means notebook has skip conditions or hardware constraints*

### Common Failure Types

1. **Import Errors**: Missing dependencies
2. **Path Errors**: Incorrect file paths
3. **Hardware Errors**: Neuron cores not available
4. **Timeout Errors**: Notebook took too long
5. **Output Mismatch**: Cell output changed (may be expected)

## Handling Test Updates

### When Notebooks Change

If you update a notebook and the output legitimately changes:

1. **Run the notebook manually** to verify it works
2. **Update the stored outputs** by running with `--nbval-lax` flag
3. **Commit the updated notebook** with new outputs

```bash
# Run with relaxed validation to update outputs
pytest labs/your-notebook.ipynb --nbval-lax
```

### When SDK Updates

After Neuron SDK updates:

1. **Run all tests** to identify issues
2. **Update notebooks** as needed for new SDK
3. **Update requirements** if dependencies changed
4. **Update timeouts** if performance characteristics changed

## Advanced Usage

### Custom Test Markers

```bash
# Run only fast tests
pytest -m "not slow"

# Run only GPU/Neuron tests  
pytest -m gpu

# Run integration tests
pytest -m integration
```

### Debugging Failed Tests

```bash
# Stop on first failure with detailed output
pytest --maxfail=1 -vv --tb=long

# Run specific cell ranges
pytest --nbval-cell-timeout=300 notebook.ipynb

# Skip output comparison (execution only)
pytest --nbval-lax notebook.ipynb
```

### Sequential Execution

```bash
# All notebooks run sequentially to avoid Neuron device conflicts
pytest labs/ --nbval

# Control execution order if needed
pytest labs/NxD/ labs/FineTuning/ labs/vLLM/ labs/NKI/ --nbval
```

## CI/CD Integration

### GitHub Actions

The repository includes automated testing via GitHub Actions:

- **Syntax Validation**: Runs on every PR
- **Execution Tests**: Runs on Trainium instances (when available)
- **Scheduled Tests**: Weekly runs to catch SDK issues

### Local Pre-commit

Set up pre-commit hooks to validate notebooks before pushing:

```bash
# Install pre-commit
pip install pre-commit

# Set up hooks (create .pre-commit-config.yaml)
pre-commit install

# Run manually
pre-commit run --all-files
```

## Troubleshooting

### Virtual Environment Issues

```bash
# Check if Neuron SDK is available
ls -la /opt/aws_neuronx_venv_pytorch_2_8_nxd_inference/

# Manually activate and test
source /opt/aws_neuronx_venv_pytorch_2_8_nxd_inference/bin/activate
python -c "import neuronx_distributed_inference"
```

### Path Issues

```bash
# Verify you're in the right directory
pwd  # Should end with /neuron-workshops

# Check labs directory exists
ls -la labs/

# Verify paths in notebooks match
grep -r "/home/ubuntu/build-on-trainium-workshop" labs/
```

### Hardware Issues

```bash
# Check Neuron cores
neuron-ls

# Check system resources
neuron-top

# Verify instance type
curl -s http://169.254.169.254/latest/meta-data/instance-type
```

### Test Dependencies

```bash
# Install/update test dependencies
pip install -r requirements-test.txt

# Check nbval installation
python -c "import nbval; print(nbval.__version__)"

# Verify pytest plugins
pytest --version
```

## Best Practices

### For Notebook Authors

1. **Keep outputs clean**: Clear unnecessary outputs before committing
2. **Use relative paths**: Avoid hardcoded absolute paths where possible
3. **Add error handling**: Include try/catch for expected failures
4. **Document requirements**: Note any special setup in markdown cells
5. **Test locally**: Run notebooks manually before committing

### For Maintainers

1. **Run tests regularly**: Especially after SDK updates
2. **Monitor timeouts**: Adjust as hardware/software performance changes
3. **Update dependencies**: Keep test requirements current
4. **Review failures**: Distinguish between real issues and expected changes
5. **Document changes**: Update this guide when test setup changes

## Configuration Files

- `pytest.ini`: Main pytest configuration
- `requirements-test.txt`: Test dependencies
- `test_notebooks.py`: Test discovery and configuration
- `run_tests.sh`: Convenience script for running tests
- `.github/workflows/test-notebooks.yml`: CI/CD configuration

## Getting Help

1. **Check logs**: Look at detailed pytest output
2. **Run manually**: Execute problematic notebooks by hand
3. **Check environment**: Verify Neuron SDK and dependencies
4. **Review changes**: Compare with working versions
5. **Ask for help**: Include full error messages and environment details
