#!/bin/bash
# Test runner script for Neuron Workshop notebooks
# This script sets up the environment and runs notebook tests

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Neuron Workshop Notebook Testing${NC}"
echo "=================================="

# Check if we're in the right directory and navigate if needed
if [ ! -d "labs" ]; then
    # Try to find the neuron-workshops directory
    if [ -d "neuron-workshops/labs" ]; then
        echo -e "${YELLOW}Changing to neuron-workshops directory...${NC}"
        cd neuron-workshops
    elif [ -d "../labs" ]; then
        echo -e "${YELLOW}Changing to parent directory...${NC}"
        cd ..
    else
        echo -e "${RED}Error: labs directory not found. Please run from the repository root or parent directory.${NC}"
        echo -e "${RED}Current directory: $(pwd)${NC}"
        echo -e "${RED}Looking for: labs/ directory${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}Working directory: $(pwd)${NC}"

# Activate Neuron SDK virtual environment (required)
NEURON_VENV="/opt/aws_neuronx_venv_pytorch_2_8_nxd_inference/bin/activate"
if [ -f "$NEURON_VENV" ]; then
    echo -e "${GREEN}Activating Neuron SDK virtual environment...${NC}"
    source "$NEURON_VENV"
else
    echo -e "${RED}Error: Neuron SDK virtual environment not found at $NEURON_VENV${NC}"
    echo -e "${RED}Please ensure you're running on a Trainium instance with Neuron SDK installed.${NC}"
    exit 1
fi

# Install test dependencies
echo -e "${GREEN}Installing test dependencies...${NC}"
pip install -r requirements-test.txt

# Set environment variables for testing
export LABS_DIR="$(pwd)/labs"
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Parse command line arguments
PYTEST_ARGS=""
SPECIFIC_NOTEBOOK=""
REPORT_HTML=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --notebook)
            SPECIFIC_NOTEBOOK="$2"
            shift 2
            ;;
        --html-report)
            REPORT_HTML="--html=test_report.html --self-contained-html"
            shift
            ;;
        --fast)
            PYTEST_ARGS="$PYTEST_ARGS -x --tb=line"
            shift
            ;;
        --verbose)
            PYTEST_ARGS="$PYTEST_ARGS -vv"
            shift
            ;;
        *)
            PYTEST_ARGS="$PYTEST_ARGS $1"
            shift
            ;;
    esac
done

# Run tests
echo -e "${GREEN}Running notebook tests...${NC}"
echo "Working directory: $(pwd)"
echo "Labs directory: $LABS_DIR"

if [ -n "$SPECIFIC_NOTEBOOK" ]; then
    echo -e "${YELLOW}Testing specific notebook: $SPECIFIC_NOTEBOOK${NC}"
    pytest $PYTEST_ARGS $REPORT_HTML --nbval "labs/$SPECIFIC_NOTEBOOK"
else
    echo -e "${YELLOW}Testing all notebooks in labs/${NC}"
    echo "Discovering notebooks..."
    find labs/ -name "*.ipynb" -not -path "*/.ipynb_checkpoints/*" | head -5
    echo ""
    pytest $PYTEST_ARGS $REPORT_HTML --nbval labs/
fi

# Check exit code
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    if [ -n "$REPORT_HTML" ]; then
        echo -e "${GREEN}HTML report generated: test_report.html${NC}"
    fi
else
    echo -e "${RED}✗ Some tests failed.${NC}"
    exit 1
fi