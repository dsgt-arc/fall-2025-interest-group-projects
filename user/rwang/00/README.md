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
- gemma3:270m (292MB)
- gemma3:4b (3.3GB)
- gemma3:27b (17GB)  
  
**Comparing across quantization levels**
- gemma3:4b-it-qat (4GB)
- gemma3:4b-it-q4_K_M (3.3GB)
- gemma3:4b-it-q8_0 (5GB)