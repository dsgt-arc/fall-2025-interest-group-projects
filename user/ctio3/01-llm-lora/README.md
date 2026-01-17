# 01-llm-lora

## Try experimenting with:
    • LoRA rank and scaling matrices
    • Learning rate and batch size
    • Adapter merge strategies or freezing parameters
These experiments will help you explore efficiency–performance tradeoffs in LoRA fine-tuning.



| Hyperparameter                | Accuracy   | Training Time | Trainable Parameters | Notes                                  |
|-------------------------------|-----------|---------------|--------------------|----------------------------------------|
| batch_size: 8                 | 0.886492  | 75.70241332   | 1,603,163          | baseline                               |
| batch_size: 16                | 0.880863  | 81.71039081   | 1,603,163          | slight accuracy drop, slower           |
| batch_size: 32                | 0.884615  | 55.5227406    | 1,603,163          | faster, almost same accuracy           |
| lora_rank: 8                  | 0.886492  | 75.70241332   | 1,603,163          | baseline                               |
| lora_rank: 16                 | 0.881801  | 81.16608405   | 1,971,803          | slight accuracy drop, more params      |
| lora_rank: 32                 | 0.878987  | 75.56088829   | 2,709,083          | accuracy drops, params increase a lot  |
| learning_rate_lora: 1e-4      | 0.886492  | 75.70241332   | 1,603,163          | baseline                               |
| learning_rate_lora: 5e-5      | 0.874296  | 181.8380032   | 1,603,163          | accuracy drops, slower convergence     |
| learning_rate_lora: 2e-4      | 0.890244  | 179.3662257   | 1,603,163          | slight accuracy gain, slower convergence |

The hyperparameter sweep results suggest keeping the default LoRA rank (8) and learning rate (1e-4) values for best accuracy, while increasing the batch size from 8 to 32 increases training speed while retaining almost the same accuracy.