## Exercise 1

The RTX6000 GPU has 24 GB of VRAM. With Ollama and GGUF quantized models, the largest Gemma3 model that fits and runs reliably is the 27B parameter model using Q4 quantization.

Answer:
gemma3:27b (Q4_K_M)

Higher quantization levels require more than 24 GB of VRAM and fail with out-of-memory errors.

## Exercise 2

Smaller Gemma3 models generate tokens much faster but with weaker reasoning. Medium-sized models provide a good balance, while the largest model is slowest but produces the best outputs.

Summary:
- Fastest: gemma3:4b  
- Best balance: gemma3:12b (Q4)  
- Best quality on RTX6000: gemma3:27b (Q4_K_M)

## Exercise 3

Model and prompt can be moved out of the sbatch script into a YAML file and read at runtime.

Example config:
```yaml
model: gemma3:12b
prompt: "Write a haiku about Georgia Tech"
```

Relevant sbatch lines:
```bash
CONFIG_FILE=${CONFIG_FILE:-config/ollama.yaml}
MODEL=$(yq -r '.model' $CONFIG_FILE)
PROMPT=$(yq -r '.prompt' $CONFIG_FILE)
```

```bash
ollama run $MODEL "$PROMPT" 2> /dev/null
python_openai_query $PORT $MODEL "$PROMPT"
```
