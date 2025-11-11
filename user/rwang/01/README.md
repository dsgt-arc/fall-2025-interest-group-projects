### Experimenting with LoRA rank, Learning Rate, and Batch Size
**didn't have time to try more complex stuff**  
All experiments used these common parameters:
```
{
    "dataset_name": "rotten_tomatoes",
    "model_name": "roberta-base",
    "max_length": 128,
    "num_labels": 2,
    "num_epochs": 3,
    "learning_rate_full": 2e-05,
    "lora_alpha_ratio": 2,
  }
```
**Experiments**  
| Experiment | Batch Size | LoRA Rank | LoRA LR |   
|---|---|---|---| 
| Baseline (quick) | 8 | 8 | 1e-4 |
| Reduced batch size (quick-2) | **4** | 8 | 1e-4 |
| Increased batch size (quick-3) | **16** | 8 | 1e-4 |
| Reduced LoRA Rank (quick-4) | 8 | **4** | 1e-4 |
| Increased LoRA Rank (quick-5) | 8 | **16** | 1e-4 |
| Increased LoRA LR (quick-6) | 8 | 8 | 1e-2 |
| Reduced LoRA LR (quick-7) | 8 | 8 | 2e-5 |

**Results**
| Experiment | Speedup | Param Reduction |Accuracy Diff | Accuracy Retention % |  
|---|---|---|---|---|
| Baseline (quick) | 1.473 | 77.751 | -0.008 | 99.155 |
| Reduced batch size (quick-2) | 1.379 | 77.751 | 0.009 | 101.078 |
| Increased batch size (quick-3) | 1.307 | 77.751 | 0.004 | 100.429 |
| Reduced LoRA Rank (quick-4) | 1.369 | 87.851 | -0.005 | 99.471 |
| Increased LoRA Rank (quick-5) | 1.377 | 63.215 | -0.011 | 98.727 |
| Increased LoRA LR (quick-6) | 1.377 | 77.751 | -0.384 | 56.582 |
| Reduced LoRA LR (quick-7) | 1.379 | 77.751 | -0.030 | 96.628 |
