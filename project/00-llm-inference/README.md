# LLM inference

This mini-project will walk you through the steps of performing inference with ollama and vllm on PACE.
At the end of this, you should have validated your environment for running LLM inference workloads on PACE.

Before starting, read through the [SSH and GIT environment setup instructions](https://notes.dsgt-arc.org/applied-methods/setup/ssh-git/).
We will assume that you have a valid `pace` and `pace-interactive` SSH configuration setup alongside the GATech VPN access.

## ollama

Ollama is a popular open-source tool based on `llama.cpp` for distributing model weights with docker-esque semantics.
It includes an OpenAI-compatible API for inference.

First, setup an interactive PACE session with GPU support.
Consider adding the `salloc-gpu` script to a folder in your `$PATH` for easier access.
You can run the following command directly to start a session with a RTX6000 GPU for 1 hour:

```bash
salloc \
    --account=paceship-dsgt_clef2026 \
    --gres=gpu:1 \
    --constraint=RTX6000 \
    --qos=embers \
    --time=1:00:00
```

Make note of the `hostname` and configure `pace-interactive` appropriately.
Alternatively, use a terminal multiplexer like `tmux` to manage multiple terminal windows and processes within a single SSH session.
We will be running the ollama server in one terminal window and the client in another.

Ollama is available as a module on PACE, so we don't need to install it ourselves.

```bash
module spider ollama
...
     Versions:
        ollama/0.5.1
        ollama/0.6.6
        ollama/0.9.0
```

It will download models to a directory under your home directory by default.
You will not have enough space in home to store several large models, so we will set the `OLLAMA_MODELS` environment variable to point to a directory in `scratch`.
The best way to do this is to add a new line to the bash startup script (e.g. `~/.bashrc`):

```bash
# in ~/.bashrc or similar
export OLLAMA_MODELS=$HOME/scratch/ollama_models
```

Then we can load the module and start the server:

```bash
# in the serving terminal window
module load ollama
ollama serve
```

In a new terminal window, we can run the client to download and run a model.
Create a new session via `ssh pace-interactive`.
If you are using tmux, create a new window with `ctrl-b c` and switch panes with `ctrl-b n`.

```bash
# in the client terminal window     
module load ollama
ollama run gemma3:4b
```

This should get you into a terminal where you can type messages to the model.
Feel free to try out some prompts:

```bash
pulling manifest
pulling aeda25e63ebd: 100% ▕██████████████████████████████████████████████████████████▏ 3.3 GB
pulling e0a42594d802: 100% ▕██████████████████████████████████████████████████████████▏  358 B
pulling dd084c7d92a3: 100% ▕██████████████████████████████████████████████████████████▏ 8.4 KB
pulling 3116c5225075: 100% ▕██████████████████████████████████████████████████████████▏   77 B
pulling b6ae5839783f: 100% ▕██████████████████████████████████████████████████████████▏  489 B
verifying sha256 digest
writing manifest
success
>>> define the normal distribution for me in 100 words or less
The normal distribution, or “bell curve,” is a fundamental concept in statistics. It describes how data tends to
cluster around a central average (the mean).  It’s symmetrical, meaning the left and right sides mirror each
other, and the shape resembles a bell. Most data points fall within one or two standard deviations of the mean,
while fewer points are further away. This distribution is incredibly common and used to model many natural
phenomena and human measurements, like height or test scores.
```

We can check that the server is running via curl.

```bash
$ curl http://localhost:11434/api/tags | jq
{
  "models": [
    {
      "name": "gemma3:4b",
      "model": "gemma3:4b",
      "modified_at": "2025-11-02T17:26:02-05:00",
      "size": 3338801804,
      "digest": "a2af6cc3eb7fa8be8504abaf9b04e88f17a119ec3f04a3addf55f92841195f5a",
      "details": {
        "parent_model": "",
        "format": "gguf",
        "family": "gemma3",
        "families": [
          "gemma3"
        ],
        "parameter_size": "4.3B",
        "quantization_level": "Q4_K_M"
      }
    }
  ]
}
```

We can verify that the model files are in the `OLLAMA_MODELS` directory:

```bash
$ tree ~/scratch/ollama_models/
/storage/home/hcoda1/8/amiyaguchi3/scratch/ollama_models/
├── blobs
│   ├── sha256-3116c52250752e00dd06b16382e952bd33c34fd79fc4fe3a5d2c77cf7de1b14b
│   ├── sha256-aeda25e63ebd698fab8638ffb778e68bed908b960d39d0becc650fa981609d25
│   ├── sha256-b6ae5839783f2ba248e65e4b960ab15f9c4b7118db285827dba6cba9754759e2
│   ├── sha256-dd084c7d92a3c1c14cc09ae77153b903fd2024b64a100a0cc8ec9316063d2dbc
│   └── sha256-e0a42594d802e5d31cdc786deb4823edb8adff66094d49de8fffe976d753e348
└── manifests
    └── registry.ollama.ai
        └── library
            └── gemma3
                └── 4b
```

### benchmarking ollama

We can benchmark ollama using a pre-built script: https://github.com/larryhopecode/ollama-benchmark.
We've forked this repository to make it easier to run with `uv`.

```bash
uv tool install https://github.com/dsgt-arc/ollama-benchmark.git
```

This will install the `ollama-benchmark` tool that should be accessible from your PATH.


```bash
$  nvidia-smi
Sun Nov  2 17:45:28 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 575.57.08              Driver Version: 575.57.08      CUDA Version: 12.9     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Quadro RTX 6000                On  |   00000000:3B:00.0 Off |                    0 |
| 33%   29C    P8             17W /  260W |       4MiB /  23040MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+

$ ollama-benchmark --verbose --models gemma3:4b --prompts "who is george p burdell?"

----------------------------------------------------
        Model: gemma3:4b
        Performance Metrics:
            Prompt Processing:  1543.26 tokens/sec
            Generation Speed:   87.11 tokens/sec
            Combined Speed:     88.72 tokens/sec

        Workload Stats:
            Input Tokens:       16
            Generated Tokens:   814
            Model Load Time:    0.06s
            Processing Time:    0.01s
            Generation Time:    9.35s
            Total Time:         9.42s
----------------------------------------------------

Average stats:

----------------------------------------------------
        Model: gemma3:4b
        Performance Metrics:
            Prompt Processing:  1543.26 tokens/sec
            Generation Speed:   87.11 tokens/sec
            Combined Speed:     88.72 tokens/sec

        Workload Stats:
            Input Tokens:       16
            Generated Tokens:   814
            Model Load Time:    0.06s
            Processing Time:    0.01s
            Generation Time:    9.35s
            Total Time:         9.42s
----------------------------------------------------
```

### exercises

* What is the largest model you can run with ollama on a RTX6000 GPU?
* Compare the performance metrics of a model family of your choice (e.g. gemma3 or phi4) across parameter sizes and quantization levels.

