#!/bin/bash
#SBATCH -Jfinetune --account=paceship-dsgt_clef2026
#SBATCH --nodes=1 --partition=gpu-rtx6000 --gres=gpu:1
#SBATCH -t1:00:00 -qembers -ologs/finetune/%j.out
set -eux

function activate_sync_venv() {
    # activate a venv in the submission directory
    # this will fail if a pyproject.toml is not present
    local venv_path="${1?Please provide the path to the virtual environment}"
    export UV_LINK_MODE=copy
    export UV_PROJECT_ENVIRONMENT="${venv_path}"
    time uv sync --all-extras
    set +x
    source "${UV_PROJECT_ENVIRONMENT}/bin/activate"
    set -x
}

function main() {
    hostname
    nvidia-smi

    activate_sync_venv $HOME/scratch/dsgt-arc/llm-lora/.venv

    # running all experiemnts
    for ((i = 2; i < 8; i++)); do
        finetune --mode quick-$i
    done
}

main