# 00-llm-inference


## What is the largest model in the Gemma3 family that you can run with ollama on a RTX6000 GPU?

Nvidia Quadro RTX 6000 has 24.6 GB avaialble. From the gemma3 model list https://ollama.com/library/gemma3/tags, gemma3:27b-it-qat is the largest model (18GB) you can run  on this GPU.

```
    +-----------------------------------------------------------------------------------------+
    | NVIDIA-SMI 575.57.08              Driver Version: 575.57.08      CUDA Version: 12.9     |
    |-----------------------------------------+------------------------+----------------------+
    | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
    |                                         |                        |               MIG M. |
    |=========================================+========================+======================|
    |   0  Quadro RTX 6000                On  |   00000000:AF:00.0 Off |                  Off |
    | 33%   33C    P2             62W /  260W |    4976MiB /  24576MiB |      0%      Default |
    |                                         |                        |                  N/A |
    +-----------------------------------------+------------------------+----------------------+
                                                                                             
    +-----------------------------------------------------------------------------------------+
    | Processes:                                                                              |
    |  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
    |        ID   ID                                                               Usage      |
    |=========================================================================================|
    |    0   N/A  N/A         1125743      C   ...kages/ollama/0.9.0/bin/ollama       4972MiB |
+-----------------------------------------------------------------------------------------+
```


## Compare the performance metrics of a model family of your choice (e.g. gemma3 or phi4) across parameter sizes and quantization levels

When benchmarking gemma3 using speed (tokens/sec), the speed decreases with higher parameter sizes. For the same parameter size, inference speed is similar across quantization levels.

```
Model: gemma3:1b
Performance Metrics:
    Prompt Processing:  1884.80 tokens/sec
    Generation Speed:   157.97 tokens/sec
    Combined Speed:     161.77 tokens/sec


Model: gemma3:1b-it-qat
Performance Metrics:
    Prompt Processing:  1463.93 tokens/sec
    Generation Speed:   144.38 tokens/sec
    Combined Speed:     146.82 tokens/sec


Model: gemma3:27b
Performance Metrics:
    Prompt Processing:  431.83 tokens/sec
    Generation Speed:   25.03 tokens/sec
    Combined Speed:     25.49 tokens/sec

Model: gemma3:27b-it-qat
Performance Metrics:
    Prompt Processing:  871.33 tokens/sec
    Generation Speed:   25.66 tokens/sec
    Combined Speed:     27.63 tokens/sec
```

## Implement file-based configuration for the sbatch script to allow switching of the model and prompt without modifying the script directly. One way to do this is to use yq (uv tool install yq) to read from a YAML configuration file in bash.


- Install Python package `yq` to process YAML files
```bash
uv tool install yq
```

- Create `config.yaml`

```yaml
model: gemma3:1b
prompt: "write a poem about georgia tech"
```

- Export the YAML variables

```bash
export MODEL=$(yq -r '.model' config.yaml)
export PROMPT=$(yq -r '.prompt' config.yaml)
```

- Verify variables
```bash
echo "$MODEL"
echo "$PROMPT"
```

- Run script
```bash
sbatch ollama.sbatch
```

## Implement a sbatch script to run vLLM serve using the ollama.sbatch as a template
- See `vllm.sbatch`
