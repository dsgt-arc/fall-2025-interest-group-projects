# LLM LoRA fine-tuning

## configuring the venv

If you need to work in an interactive session, it's highly recommended to create the virtual environment either in `$HOME/scratch` or in `$TMPDIR` to avoid filling up your limited home directory quota.
You should already have the appropriate `$XDG_CACHE_HOME` variables set.
If not, please refer to PACE reference material or the `00-llm-inference` project.

We can set and create a new environment by configuring the `UV_PROJECT_ENVIRONMENT` variable.
This can be done once to configure the environment, and then you are free to symlink it (e.g. `ln -s $UV_PROJECT_ENVIRONMENT`) for easier activation later.

```bash
export UV_PROJECT_ENVIRONMENT=$HOME/scratch/dsgt-arc/llm-lora/.venv

# alternatively, `uv pip install -e .[dev]`
uv sync --all-extras

# link it to your project directory so you don't forget about it
ln -s $UV_PROJECT_ENVIRONMENT
```

## fine-tuning script

Run the lora fine-tuning script.
By default it will run the quick dataset with baseline and lora experiments.
You may pass in options directly through to the entrypoint

```bash
sbatch finetune.sbatch

# with custom args
sbatch finetune.sbatch --mode full
```

## exercises

- Experiment with hyperparameters such as rank, matrices, learning rate, etc.
