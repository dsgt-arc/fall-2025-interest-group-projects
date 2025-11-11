### 1. What is the largest model in the Gemma3 family that you can run with ollama on a RTX6000 GPU? 
Looking at 'nvidia-smi' -> GPU: Quadro RTX6000 (24GB RAM)
  - but only 23040MiB available (~22GB)
Largest Gemma3 model we can run -> 'gemma3:27b' (17GB) or 'gemma3:27b-it-qat' (18GB)

**Output**:  
Ollama:
```
ollama run gemma3:27b write a haiku about embedded systems
Small code, big impact,
Hidden brains in every thing,
Worlds within a chip.  
```
OpenAI:  
```
python_openai_query 49503 gemma3:27b write a haiku about embedded systems
Okay, I will! But... "write" is a very broad instruction! To give you the *best* possible response, I need a little more direction. 

Here are a few options, and I'll provide examples of each. **Please tell me which one you'd like, or give me a more specific request!**

**1. Short Story:** I can write a.....(it wrote a lot, nothing abt embedded systems which is interesting)
```
---
### 2. Compare performance metrics of gemma3 family across parameter sizes and quantization levels
**Comparing across parameter sizes**  
Larger model sizes clearly show slower prompt processing, generation speed, and combined speed.  
- gemma3:270m (292MB) 
```
        Model: gemma3:270m
        Performance Metrics:
            Prompt Processing:  1896.06 tokens/sec
            Generation Speed:   215.87 tokens/sec
            Combined Speed:     349.98 tokens/sec

        Workload Stats:
            Input Tokens:       16
            Generated Tokens:   21
            Model Load Time:    0.04s
            Processing Time:    0.01s
            Generation Time:    0.10s
            Total Time:         0.15s
```
  
- gemma3:4b (3.3GB)  
```
        Model: gemma3:4b
        Performance Metrics:
            Prompt Processing:  1635.97 tokens/sec
            Generation Speed:   101.37 tokens/sec
            Combined Speed:     170.56 tokens/sec

        Workload Stats:
            Input Tokens:       16
            Generated Tokens:   21
            Model Load Time:    0.06s
            Processing Time:    0.01s
            Generation Time:    0.21s
            Total Time:         0.28s
```

- gemma3:27b (17GB)  
```
        Model: gemma3:27b
        Performance Metrics:
            Prompt Processing:  429.68 tokens/sec
            Generation Speed:   27.64 tokens/sec
            Combined Speed:     46.42 tokens/sec

        Workload Stats:
            Input Tokens:       16
            Generated Tokens:   21
            Model Load Time:    0.06s
            Processing Time:    0.04s
            Generation Time:    0.76s
            Total Time:         0.86s
```
  
**Comparing across quantization levels** (for some reason ollama didn't recognize any quantized model tags so i didn't do this :) )
- gemma3:4b-it-qat (4GB)
- gemma3:4b-it-q4_K_M (3.3GB)
- gemma3:4b-it-q8_0 (5GB)

### 3. File based configuration for sbatch script
See *config.yaml* and *ollama.sbatch*  

### 4. vLLM sbatch script
See *vmm_serve.sbatch*