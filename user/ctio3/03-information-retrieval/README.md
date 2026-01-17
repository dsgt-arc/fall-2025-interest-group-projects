# 03-information-retrieval


For FAISS, the larger [`all-mpnet-base-v2`](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) model achieved marginally higher average top-1 similarity scores (0.7800 vs 0. 7764) but was ~4x as slow as the base model

For the final comparision, the larger model improved Dense pipelines in all 3 evaluation metrics:

- Base model:
```
MRR@10         : 4. Dense + Rerank         (0.5824)
Recall@10      : 4. Dense + Rerank         (0.9388)
Precision@10   : 4. Dense + Rerank         (0.1071)
```

- Larger model:
```
MRR@10         : 4. Dense + Rerank         (0.5925)
Recall@10      : 4. Dense + Rerank         (0.9490)
Precision@10   : 4. Dense + Rerank         (0.1082)
```
