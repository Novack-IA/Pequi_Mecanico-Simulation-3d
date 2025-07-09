#!/bin/bash

# Get the absolute path of the script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Set the PYTHONPATH to include the project's root directory
export PYTHONPATH="${PYTHONPATH}:${SCRIPT_DIR}"

# Now run your training script
python3 scripts/gyms/11v11.py