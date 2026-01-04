# LLM LoRA fine-tuning
This project showcases how to fine-tune an LLM using **LoRA (Low-Rank Adaptation)** on PACE. Follow the steps below to setup your virtual environment, install dependencies, and run the fine-tuning notebook or scripts.

## 1. Prerequisites
Before starting, make sure you've completed the [**LLM Inference tutorial**](https://github.com/dsgt-arc/fall-2025-interest-group-projects/tree/lora/project/00-llm-inference). That setup ensures you have the required caches and environment variables configured.

If not already configured, add the following lines to the end of your `~/.bashrc` (or similar startup file):

```bash
# make pip-installed tools available
export PATH=$HOME/.local/bin:$PATH
# redirect uv and huggingface cache to scratch
export XDG_CACHE_HOME="$HOME/scratch/.cache"
# redirect ollama models to scratch
export OLLAMA_MODELS=$HOME/scratch/ollama_models
```

Then reload the shell configuration:
```bash
source ~/.bashrc
```

## 2. Setting up the virtual environment
To stay within your home directory quota, create your virtual environment in `$HOME/scratch` or `$TMPDIR`.
There are two supported methods — choose one depending on your workflow.

### Option A. Manual virtual environment setup (recommended for Jupyter users)
This approach gives you a reusable `.venv` you can activate manually and connect to your Jupyter notebook.

```bash
# create a directory to store the environment
mkdir -p ~/scratch/dsgt-arc/llm-lora

# create a virtual environment using uv
uv venv ~/scratch/dsgt-arc/llm-lora/.venv

# activate it
source ~/scratch/dsgt-arc/llm-lora/.venv/bin/activate
```

From the `project/01-llm-lora/` directory (where this README and notebook are located), link and install the dependencies using the `pyproject.toml`. 

For example, if my project is under the directory `~/dsgt-arc/fall-2025-interest-group-projects/project/01-llm-lora/`, I'll run the following:

```bash
# navigate to project directory
cd ~/dsgt-arc/fall-2025-interest-group-projects/project/01-llm-lora/

# create a symbolic link to your environment in this project folder
ln -s ~/scratch/dsgt-arc/llm-lora/.venv $(pwd)/.venv

# install project dependencies defined in pyproject.toml
uv pip install -e .
```

Once done, open the Jupyter notebook and select the `.venv` kernel.


### Option B. Using `UV_PROJECT_ENVIRONMENT` (recommended for batch or sbatch runs)

You can also configure the environment path once via an environment variable:

```bash
export UV_PROJECT_ENVIRONMENT=$HOME/scratch/dsgt-arc/llm-lora/.venv

# install all dependencies, including optional [dev] extras
uv sync --all-extras

# link it to your project for easier access
ln -s $UV_PROJECT_ENVIRONMENT
```
This method is ideal if you plan to reuse the same environment across multiple runs or scripts.


## 3. Running fine-tuning jobs
You can fine-tune the model interactively through the notebook or submit jobs with SLURM. 

By default it will run the quick dataset with baseline and lora experiments.
You may pass in options directly through to the entrypoint:

```bash
sbatch finetune.sbatch

# or run a full dataset fine-tuning
sbatch finetune.sbatch --mode full
```

## 4. Exercises

Try experimenting with:
- LoRA rank and scaling matrices
- Learning rate and batch size
- Adapter merge strategies or freezing parameters

These experiments will help you explore efficiency–performance tradeoffs in LoRA fine-tuning.

## 5. Notes
- Always prefer `$HOME/scratch` or `$TMPDIR` for environments to stay under storage limits.
- To reinstall or update dependencies later:
    ```bash
    source ~/scratch/dsgt-arc/llm-lora/.venv/bin/activate
    uv pip install -e .
    ```
- If using UV_PROJECT_ENVIRONMENT, you can recreate dependencies with:
    ```bash
    uv sync --all-extras
    ```