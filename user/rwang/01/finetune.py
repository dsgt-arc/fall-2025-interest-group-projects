"""
Utility library for running full vs. LoRA fine-tuning experiments.
"""

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

from sklearn.metrics import accuracy_score
import numpy as np
import torch
from adapters import AdapterTrainer, AutoAdapterModel, LoRAConfig
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    EvalPrediction,
    IntervalStrategy,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
)

CONFIGS = {
    "quick": {
        "dataset_name": "rotten_tomatoes",
        "model_name": "roberta-base",
        "max_length": 128,
        "num_labels": 2,
        "num_epochs": 3,
        "batch_size": 16,
        "learning_rate_full": 2e-5,
        "learning_rate_lora": 1e-4,
        "lora_rank": 8,
        "lora_alpha_ratio": 2,
        "description": "Quick experiment: Rotten Tomatoes + RoBERTa-base",
    },
        "quick-2": {
        "dataset_name": "rotten_tomatoes",
        "model_name": "roberta-base",
        "max_length": 128,
        "num_labels": 2,
        "num_epochs": 3,
        "batch_size": 8,
        "learning_rate_full": 2e-5,
        "learning_rate_lora": 1e-4,
        "lora_rank": 8,
        "lora_alpha_ratio": 2,
        "description": "Quick experiment-2 batch_size = 8: Rotten Tomatoes + RoBERTa-base",
    },
        "quick-3": {
        "dataset_name": "rotten_tomatoes",
        "model_name": "roberta-base",
        "max_length": 128,
        "num_labels": 2,
        "num_epochs": 3,
        "batch_size": 32,
        "learning_rate_full": 2e-5,
        "learning_rate_lora": 1e-4,
        "lora_rank": 8,
        "lora_alpha_ratio": 2,
        "description": "Quick experiment-3 batch_size = 32: Rotten Tomatoes + RoBERTa-base",
    },
        "quick-4": {
        "dataset_name": "rotten_tomatoes",
        "model_name": "roberta-base",
        "max_length": 128,
        "num_labels": 2,
        "num_epochs": 3,
        "batch_size": 16,
        "learning_rate_full": 2e-5,
        "learning_rate_lora": 1e-4,
        "lora_rank": 4,
        "lora_alpha_ratio": 2,
        "description": "Quick experiment-4 lora_rank = 4: Rotten Tomatoes + RoBERTa-base",
    },
        "quick-5": {
        "dataset_name": "rotten_tomatoes",
        "model_name": "roberta-base",
        "max_length": 128,
        "num_labels": 2,
        "num_epochs": 3,
        "batch_size": 16,
        "learning_rate_full": 2e-5,
        "learning_rate_lora": 1e-4,
        "lora_rank": 16,
        "lora_alpha_ratio": 2,
        "description": "Quick experiment-5 lora_rank = 16: Rotten Tomatoes + RoBERTa-base",
    },
        "quick-6": {
        "dataset_name": "rotten_tomatoes",
        "model_name": "roberta-base",
        "max_length": 128,
        "num_labels": 2,
        "num_epochs": 3,
        "batch_size": 16,
        "learning_rate_full": 2e-5,
        "learning_rate_lora": 1e-2,
        "lora_rank": 8,
        "lora_alpha_ratio": 2,
        "description": "Quick experiment-6 lr_lora = 1e-2: Rotten Tomatoes + RoBERTa-base",
    },
        "quick-7": {
        "dataset_name": "rotten_tomatoes",
        "model_name": "roberta-base",
        "max_length": 128,
        "num_labels": 2,
        "num_epochs": 3,
        "batch_size": 16,
        "learning_rate_full": 2e-5,
        "learning_rate_lora": 2e-5,
        "lora_rank": 8,
        "lora_alpha_ratio": 2,
        "description": "Quick experiment-7 lr_lora = 2e-5: Rotten Tomatoes + RoBERTa-base",
    },
    "full": {
        "dataset_name": "imdb",
        "model_name": "roberta-large",
        "max_length": 512,
        "num_labels": 2,
        "num_epochs": 3,
        "batch_size": 8,
        "learning_rate_full": 2e-5,
        "learning_rate_lora": 3e-4,
        "lora_rank": 16,
        "lora_alpha_ratio": 2,
        "description": "Full experiment: IMDB + RoBERTa-large (GPU recommended)",
    },
}


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Gets the available torch device (CUDA or CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_and_prepare_dataset(dataset_name):
    """Loads and prepares a specified dataset."""
    if dataset_name == "rotten_tomatoes":
        dataset = load_dataset("rotten_tomatoes")
        text_field = "text"
    elif dataset_name == "imdb":
        dataset = load_dataset("imdb")
        text_field = "text"
    elif dataset_name == "sst2":
        dataset = load_dataset("glue", "sst2")
        text_field = "sentence"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return dataset, text_field


def tokenize_dataset(dataset, text_field, model_name, max_length):
    """Tokenizes the dataset using the specified model's tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(
            examples[text_field],
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True, desc="Tokenizing")

    columns_to_remove = [
        col
        for col in tokenized_dataset["train"].column_names
        if col not in ["input_ids", "attention_mask", "label"]
    ]
    tokenized_dataset = tokenized_dataset.remove_columns(columns_to_remove)
    tokenized_dataset.set_format("torch")
    return tokenized_dataset, tokenizer


def compute_metrics(eval_pred: EvalPrediction):
    """Compute metrics for evaluation."""
    labels = eval_pred.label_ids
    preds = np.argmax(eval_pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}


def create_full_finetuning_model(model_name, num_labels):
    """Create a standard model for full fine-tuning."""
    model = RobertaForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    return model


def create_lora_model(model_name, num_labels, rank=8, alpha=16):
    """Create a model with LoRA adapters for PEFT."""
    config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    model = AutoAdapterModel.from_pretrained(model_name, config=config)

    lora_config = LoRAConfig(
        r=rank,
        alpha=alpha,
        dropout=0.1,
        selfattn_lora=False,
        intermediate_lora=True,
        output_lora=False,
        attn_matrices=["q", "v"],
        composition_mode="add",
        init_weights="lora",
    )

    model.add_adapter("lora_adapter", config=lora_config)
    model.add_classification_head(
        "lora_adapter",
        num_labels=num_labels,
        id2label={0: "negative", 1: "positive"},
    )
    model.train_adapter("lora_adapter")
    return model


def run_experiment(
    model,
    device,
    tokenized_dataset,
    config,
    experiment_name,
    learning_rate,
    is_full_finetuning=True,
):
    """Runs a single training and evaluation experiment."""
    model.to(device)
    output_dir = tempfile.mkdtemp()

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy=IntervalStrategy.EPOCH,
        save_strategy=IntervalStrategy.NO,
        learning_rate=learning_rate,
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"] * 2,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        load_best_model_at_end=False,
        fp16=torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=False,
    )

    TrainerClass = Trainer if is_full_finetuning else AdapterTrainer
    eval_split = "test" if "test" in tokenized_dataset else "validation"

    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset[eval_split],
        compute_metrics=compute_metrics,
    )

    start_time = time.time()
    train_result = trainer.train()
    training_time = time.time() - start_time

    eval_result = trainer.evaluate()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    results = {
        "experiment_name": experiment_name,
        "training_time": training_time,
        "train_loss": train_result.training_loss,
        "eval_loss": eval_result["eval_loss"],
        "eval_accuracy": eval_result["eval_accuracy"],
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_percentage": 100 * trainable_params / total_params,
    }

    shutil.rmtree(output_dir, ignore_errors=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results, model


def log_comparison(results_full, results_lora, config):
    """Creates a comparison dictionary from two experiment results."""

    def _safe_divide(a, b):
        return (a / b) if b != 0 else 0.0

    comparison_metrics = {}
    for metric in ["accuracy"]:
        eval_metric = f"eval_{metric}"
        full_val = results_full.get(eval_metric, 0.0)
        lora_val = results_lora.get(eval_metric, 0.0)

        comparison_metrics[f"{metric}_diff"] = lora_val - full_val
        comparison_metrics[f"{metric}_retention_pct"] = (
            _safe_divide(lora_val, full_val) * 100
        )

    results_dict = {
        "config": config,
        "full_finetuning": results_full,
        "lora": results_lora,
        "comparison": {
            "speedup": float(
                results_full["training_time"] / results_lora["training_time"]
            ),
            "param_reduction": float(
                results_full["trainable_params"] / results_lora["trainable_params"]
            ),
            **comparison_metrics,
        },
    }
    return results_dict


def predict_sentiment(text, model, tokenizer, device, max_length=128):
    """Predict sentiment for a given text string. Returns (sentiment, confidence)."""
    model.eval()
    inputs = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()

    sentiment = "Positive" if pred_class == 1 else "Negative"
    confidence = probs[0][pred_class].item()
    return sentiment, confidence


def main():
    parser = argparse.ArgumentParser(
        description="Run LoRA vs. Full Fine-Tuning Experiments."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="quick",
        help="Experiment mode (name of config).",
    )
    parser.add_argument(
        "--run",
        type=str,
        default="all",
        choices=["all", "baseline", "lora"],
        help="Which experiment(s) to run.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="results",
        help="Directory to save results.json and plots.",
    )
    args = parser.parse_args()

    config = CONFIGS[args.mode]
    device = get_device()
    print(f"Running {args.mode} mode on device: {device}")

    dataset, text_field = load_and_prepare_dataset(config["dataset_name"])
    tokenized_dataset, tokenizer = tokenize_dataset(
        dataset, text_field, config["model_name"], config["max_length"]
    )

    results_full = None
    results_lora = None

    if args.run in ["all", "baseline"]:
        model_full = create_full_finetuning_model(
            config["model_name"], config["num_labels"]
        )
        results_full, _ = run_experiment(
            model=model_full,
            device=device,
            tokenized_dataset=tokenized_dataset,
            config=config,
            experiment_name="Full Fine-Tuning",
            learning_rate=config["learning_rate_full"],
            is_full_finetuning=True,
        )

    if args.run in ["all", "lora"]:
        lora_alpha = config["lora_rank"] * config["lora_alpha_ratio"]
        model_lora = create_lora_model(
            config["model_name"],
            config["num_labels"],
            rank=config["lora_rank"],
            alpha=lora_alpha,
        )
        results_lora, _ = run_experiment(
            model=model_lora,
            device=device,
            tokenized_dataset=tokenized_dataset,
            config=config,
            experiment_name="LoRA Fine-Tuning",
            learning_rate=config["learning_rate_lora"],
            is_full_finetuning=False,
        )

    if args.run == "all":
        results_dict = log_comparison(results_full, results_lora, config)
        output_path = Path(args.output_path) / f"experiment_results_{args.mode}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(results_dict, f, indent=2)
        # dump the output to the logs too
        print(output_path.read_text())


if __name__ == "__main__":
    # Set seeds for reproducibility
    set_seed(42)
    main()
