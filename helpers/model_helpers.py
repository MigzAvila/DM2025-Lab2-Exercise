"""

Utilities for Hugging Face datasets, tokenization, evaluation, and model manipulation.

Features:
- Create Hugging Face datasets from text and optional labels
- Tokenize batches with truncation and padding
- Compute macro and micro F1 scores for evaluation
- Simple text augmentation via random word drop
- Map parameter names to encoder layer indices
- Reinitialize last N layers of a Transformer encoder
"""

import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from datasets import Dataset
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from typing import List, Optional, Dict, Any, List, Tuple
from transformers import PreTrainedTokenizer, Trainer, AutoTokenizer, \
    AutoConfig, AutoModelForSequenceClassification, TrainingArguments, \
    EarlyStoppingCallback, default_data_collator
from scipy.stats import rankdata
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# Create a Hugging Face Dataset from lists of texts and optional labels
# -------------------------------------------------------------------------
def create_dataset(text_list: List[str], label_list: Optional[List[int]] = None) -> Dataset:
    """
    Create a Hugging Face Dataset from text and optional labels.

    Parameters
    ----------
    text_list : list of str
        List of text samples.
    label_list : list of int, optional
        Optional list of labels corresponding to each text sample.

    Returns
    -------
    Dataset
        Hugging Face Dataset containing the text and optional labels.
    """
    data: Dict[str, Any] = {"post": text_list}
    if label_list is not None:
        data["label"] = label_list
    return Dataset.from_dict(data)


# -------------------------------------------------------------------------
# Tokenization helper for batch encoding with truncation/padding
# -------------------------------------------------------------------------
def tokenize_batch(batch: Dict[str, List[str]], tokenizer: PreTrainedTokenizer, max_length: int = 256) -> Dict[str, Any]:
    """
    Tokenize a batch of text samples using a Hugging Face tokenizer.

    Parameters
    ----------
    batch : dict
        Dictionary containing a 'post' key with a list of text samples.
    tokenizer : PreTrainedTokenizer
        Hugging Face tokenizer instance.
    max_length : int, default 256
        Maximum sequence length for truncation/padding.

    Returns
    -------
    dict
        Dictionary containing tokenized inputs ready for model ingestion.
    """
    return tokenizer(
        batch["post"],
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )


# -------------------------------------------------------------------------
# Compute macro and micro F1 scores for evaluation
# -------------------------------------------------------------------------
def calculate_f1(eval_prediction) -> Dict[str, float]:
    """
    Compute macro and micro F1 scores from model predictions.

    Parameters
    ----------
    eval_prediction : object
        Object containing model predictions and true labels.
        Must have attributes:
            - predictions: numpy array or torch tensor of logits
            - label_ids: true label array

    Returns
    -------
    dict
        Dictionary with 'f1_macro' and 'f1_micro' scores.
    """
    logits = np.asarray(eval_prediction.predictions)
    true_labels = eval_prediction.label_ids
    predicted_labels = logits.argmax(axis=-1)

    f1_macro = f1_score(true_labels, predicted_labels, average="macro")
    f1_micro = f1_score(true_labels, predicted_labels, average="micro")

    return {"f1_macro": f1_macro, "f1_micro": f1_micro}


# -------------------------------------------------------------------------
# Map a parameter name to its corresponding encoder layer index
# -------------------------------------------------------------------------
def get_encoder_layer_index(param_name: str, num_hidden_layers: int) -> int:
    """
    Determine which Transformer encoder layer a parameter belongs to.

    Parameters
    ----------
    param_name : str
        Name of the model parameter.
    num_hidden_layers : int
        Total number of hidden layers in the encoder.

    Returns
    -------
    int
        Layer index (0 for embeddings, 1-based for encoder layers, or num_hidden_layers + 1 if not found).
    """
    if "embeddings" in param_name:
        return 0
    for i in range(num_hidden_layers):
        if f"encoder.layer.{i}." in param_name:
            return i + 1
    return num_hidden_layers + 1

"""

Custom training utilities for PyTorch / Hugging Face Transformers:
- Focal loss for class imbalance
- WeightedTrainer with optional focal loss, R-Drop, and LLRD
- Multi-head classifier for ensemble-style predictions

Example Usage:

from transformers import AutoModelForSequenceClassification
from custom_training_utils import WeightedTrainer, MultiHeadClassifier

# Multi-head classifier
model = MultiHeadClassifier(input_dim=768, num_labels=3, n_heads=4, dropout_prob=0.2)
x = torch.randn(8, 768)
logits = model(x)  # [8, 3]

# Weighted Trainer
trainer = WeightedTrainer(
    model=model,
    class_weight=[1.0, 2.0, 0.5],
    rdrop_alpha=0.2,
    gamma=2.0,
    layerwise_lr_decay=0.9,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    args=training_args
)
"""

# --------------------------------------------------------------------------
# Focal Loss
# --------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """
    Focal Loss for class imbalance.

    Args:
        gamma (float): focusing parameter for modulating factor (1-pt)^gamma
        weight (torch.Tensor, optional): class weights
        reduction (str): 'mean' or 'sum' over the batch

    Example:
        focal_loss = FocalLoss(gamma=2.0)
        loss = focal_loss(logits, labels)
    """

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean() if self.reduction == "mean" else focal_loss.sum()


# --------------------------------------------------------------------------
# Weighted Trainer
# --------------------------------------------------------------------------
class WeightedTrainer(Trainer):
    """
    Custom Trainer supporting:
        - Layer-wise learning rate decay (LLRD)
        - Optional Focal Loss
        - R-Drop regularization
        - Label smoothing

    Args:
        class_weight (list[float], optional): class weighting for CE/focal loss
        rdrop_alpha (float): weight for R-Drop KL loss
        layerwise_lr_decay (float): decay factor per layer
        label_smoothing_factor (float): smoothing factor for CE loss
        gamma (float): focal loss gamma
        kwargs: all other Trainer args

    Example:
        trainer = WeightedTrainer(
            model=model,
            class_weight=[1.0, 2.0, 0.5],
            rdrop_alpha=0.2,
            gamma=2.0,
            layerwise_lr_decay=0.9,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=tokenizer,
            args=training_args
        )
    """

    def __init__(
        self,
        class_weight: Optional[List[float]] = None,
        rdrop_alpha: float = 0.0,
        layerwise_lr_decay: float = 1.0,
        label_smoothing_factor: float = 0.05,
        gamma: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.class_weight = torch.tensor(class_weight, dtype=torch.float) if class_weight is not None else None
        self.rdrop_alpha = rdrop_alpha
        self.layerwise_lr_decay = layerwise_lr_decay
        self.label_smoothing = label_smoothing_factor
        self.gamma = gamma

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        lr, weight_decay = self.args.learning_rate, self.args.weight_decay
        decay_rate = self.layerwise_lr_decay

        if decay_rate == 1.0:
            return super().create_optimizer()

        model = self.model
        num_layers = getattr(model.config, "num_hidden_layers", 12)
        layer_groups = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            layer_id = get_encoder_layer_index(name, num_layers)
            layer_groups.setdefault(layer_id, []).append((name, param))

        optimizer_groups = []
        max_layer_id = max(layer_groups.keys())
        no_decay_terms = ["bias", "LayerNorm.weight"]

        for layer_id in sorted(layer_groups.keys()):
            scaled_lr = lr * (decay_rate ** (max_layer_id - layer_id))
            params = layer_groups[layer_id]

            decay_params = [p for n, p in params if not any(nd in n for nd in no_decay_terms)]
            no_decay_params = [p for n, p in params if any(nd in n for nd in no_decay_terms)]

            if decay_params:
                optimizer_groups.append({"params": decay_params, "lr": scaled_lr, "weight_decay": weight_decay})
            if no_decay_params:
                optimizer_groups.append({"params": no_decay_params, "lr": scaled_lr, "weight_decay": 0.0})

        self.optimizer = AdamW(
            optimizer_groups,
            lr=lr,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )
        return self.optimizer

    def ce_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.view(-1).long() if labels.ndim > 1 else labels.long()
        logits = logits.view(logits.size(0), -1) if logits.ndim > 2 else logits
        assert logits.size(0) == labels.size(0), f"Logits {logits.size()} - labels {labels.size()}"

        device = logits.device
        weight = self.class_weight.to(device) if self.class_weight is not None else None

        if self.gamma > 0.0:
            return FocalLoss(gamma=self.gamma, weight=weight)(logits, labels)
        return F.cross_entropy(logits, labels, weight=weight, label_smoothing=self.label_smoothing)

    def compute_loss(self, model, inputs, return_outputs: bool = False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs1 = model(**inputs)
        logits1 = outputs1.logits

        if self.rdrop_alpha > 0.0:
            outputs2 = model(**inputs)
            logits2 = outputs2.logits

            loss_ce = 0.5 * (self.ce_loss(logits1, labels) + self.ce_loss(logits2, labels))
            p_log, q_log = F.log_softmax(logits1, dim=-1), F.log_softmax(logits2, dim=-1)
            kl_loss = 0.5 * (F.kl_div(p_log, q_log.exp(), reduction="batchmean") + F.kl_div(q_log, p_log.exp(), reduction="batchmean"))

            total_loss = loss_ce + self.rdrop_alpha * kl_loss
            if return_outputs:
                outputs1.logits = 0.5 * (logits1 + logits2)
                return total_loss, outputs1
            return total_loss
        else:
            loss = self.ce_loss(logits1, labels)
            return (loss, outputs1) if return_outputs else loss


# --------------------------------------------------------------------------
# Multi-head classifier
# --------------------------------------------------------------------------
class MultiHeadClassifier(nn.Module):
    """
    Multi-head classifier for ensemble-style predictions.

    Args:
        input_dim (int): input feature dimension
        num_labels (int): number of output classes
        n_heads (int): number of parallel heads
        dropout_prob (float): dropout probability per head

    Example:
        model = MultiHeadClassifier(input_dim=768, num_labels=3, n_heads=4)
        logits = model(torch.randn(8, 768))
    """

    def __init__(self, input_dim: int, num_labels: int, n_heads: int = 4, dropout_prob: float = 0.2):
        super().__init__()
        self.n_heads = n_heads
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_prob) for _ in range(n_heads)])
        self.classifiers = nn.ModuleList([nn.Linear(input_dim, num_labels) for _ in range(n_heads)])

        for clf in self.classifiers:
            nn.init.normal_(clf.weight, mean=0.0, std=0.02)
            nn.init.zeros_(clf.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x[:, 0, :]  # Take first token representation

        logits = sum(clf(drop(x)) for clf, drop in zip(self.classifiers, self.dropouts))
        return logits / self.n_heads


"""

Post-training ensemble methods for multi-class classification:
- Simple Average
- Weighted Average
- Rank Average
- Stacked Ensemble (Logistic Regression meta-model)
"""

# -----------------------------
# Simple Average Ensemble
# -----------------------------
def simple_average_ensemble(
    out_of_fold_list: List[np.ndarray],
    test_prediction_list: List[np.ndarray],
    ground_truth: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute simple average of model predictions.

    Args:
        out_of_fold_list: List of Out of Fold predictions (num_samples x num_classes) from each model.
        test_prediction_list: List of test predictions (num_samples x num_classes) from each model.
        y: Optional true labels for computing Out of Fold F1.

    Returns:
        test_result: Averaged test predictions (probabilities).
        x_out_of_fold: Predicted classes on Out of Fold data.
    """
    x_out_of_fold = np.mean(out_of_fold_list, axis=0)
    out_of_fold_pred = x_out_of_fold.argmax(axis=1)
    if ground_truth is not None:
        f1 = f1_score(ground_truth, out_of_fold_pred, average="macro")
        print(f"Simple Average Technique: {f1:.4f}")
    
    test_result = np.mean(test_prediction_list, axis=0)
    return test_result, out_of_fold_pred

# -----------------------------
# Weighted Average Ensemble
# -----------------------------
def weighted_average_ensemble(
    out_of_fold_list: List[np.ndarray],
    test_prediction_list: List[np.ndarray],
    ground_truth: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute weighted average of model predictions based on OOF F1 scores.

    Args:
        out_of_fold_list: List of OOF predictions.
        test_prediction_list: List of test predictions.
        ground_truth: True labels.

    Returns:
        test_result: Weighted averaged test predictions.
        out_of_fold_pred: Weighted averaged OOF predictions.
    """
    model_f1s = [f1_score(ground_truth, oof.argmax(axis=1), average="macro") for oof in out_of_fold_list]
    print("OOF F1 per model:", model_f1s)
    weights = np.array(model_f1s) / sum(model_f1s)

    out_of_fold = sum(w * oof for w, oof in zip(weights, out_of_fold_list))
    out_of_fold_pred = out_of_fold.argmax(axis=1)
    f1 = f1_score(ground_truth, out_of_fold_pred, average="macro")
    print(f"Weighted Average Score: {f1:.4f}")

    test_result = sum(w * test for w, test in zip(weights, test_prediction_list))
    return test_result, out_of_fold_pred

# -----------------------------
# Rank Average Ensemble
# -----------------------------
def rank_average_ensemble(
    out_of_fold_list: List[np.ndarray],
    test_prediction_list: List[np.ndarray],
    ground_truth: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute rank average of model predictions.

    Args:
        out_of_fold_list: List of OOF predictions.
        test_prediction_list: List of test predictions.
        ground_truth: True labels.

    Returns:
        test_result: Rank-averaged test predictions.
        out_of_fold_pred: Rank-averaged predictions.
    """
    ranked_oof = [np.apply_along_axis(rankdata, 1, out_of_fold_x) for out_of_fold_x in out_of_fold_list]
    out_of_fold = np.mean(ranked_oof, axis=0)
    out_of_fold_pred = out_of_fold.argmax(axis=1)
    f1 = f1_score(ground_truth, out_of_fold_pred, average="macro")
    print(f"Rank Average Technique: {f1:.4f}")

    ranked_test = [np.apply_along_axis(rankdata, 1, test) for test in test_prediction_list]
    test_result = np.mean(ranked_test, axis=0)
    return test_result, out_of_fold_pred

# -----------------------------
# Stacked Ensemble
# -----------------------------
def stacked_ensemble(
    out_of_fold_list: List[np.ndarray],
    test_prediction_list: List[np.ndarray],
    ground_truth: np.ndarray,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train a Logistic Regression meta-model on OOF predictions.

    Args:
        out_of_fold_list: List of OOF predictions.
        test_prediction_list: List of test predictions.
        ground_truth: True labels.
        random_seed: Random seed for reproducibility.

    Returns:
        test_proba: Predicted probabilities on test data.
        out_of_fold_pred: Predicted classes on out-of-fold data.
    """
    out_of_fold_x = np.hstack(out_of_fold_list)
    meta_model = LogisticRegression(
        multi_class='multinomial', max_iter=1000, C=0.1, random_state=random_seed
    )
    meta_model.fit(out_of_fold_x, ground_truth)

    out_of_fold_pred = meta_model.predict(out_of_fold_x)
    f1 = f1_score(ground_truth, out_of_fold_pred, average="macro")
    print(f"Stacked Ensemble Score: {f1:.4f}")

    x_test = np.hstack(test_prediction_list)
    test_proba = meta_model.predict_proba(x_test)
    return test_proba, out_of_fold_pred

# -----------------------------
# Run all ensembles and save submissions
# -----------------------------
def run_all_ensembles(
    out_of_fold_list: List[np.ndarray],
    test_prediction_list: List[np.ndarray],
    ground_truth: np.ndarray,
    test_df: pd.DataFrame,
    id_to_label: Dict[int, str],
    save_dir: str = "./submit"
):
    """
    Run all ensemble methods, save submission CSVs, and save confusion matrices.

    Args:
        out_of_fold_list: List of OOF predictions.
        test_prediction_list: List of test predictions.
        ground_truth: True labels.
        test_df: Test DataFrame with an "id" column.
        id_to_label: Mapping from class index to label.
        save_dir: Directory to save CSV submissions.
    """
    os.makedirs(save_dir, exist_ok=True)
    cm_dir = os.path.join(save_dir, "confusion_matrix")
    os.makedirs(cm_dir, exist_ok=True)  # Create folder for confusion matrices

    results = {}

    # Run ensembles
    test_simple, oof_simple = simple_average_ensemble(out_of_fold_list, test_prediction_list, ground_truth)
    results["simpleAvg"] = (f1_score(ground_truth, oof_simple, average="macro"), oof_simple, test_simple)

    test_weighted, oof_weighted = weighted_average_ensemble(out_of_fold_list, test_prediction_list, ground_truth)
    results["weighted"] = (f1_score(ground_truth, oof_weighted, average="macro"), oof_weighted, test_weighted)
    
    test_rank, oof_rank = rank_average_ensemble(out_of_fold_list, test_prediction_list, ground_truth)
    results["rank"] = (f1_score(ground_truth, oof_rank, average="macro"), oof_rank, test_rank)

    test_stacked, oof_stacked = stacked_ensemble(out_of_fold_list, test_prediction_list, ground_truth)
    results["stacked"] = (f1_score(ground_truth, oof_stacked, average="macro"), oof_stacked, test_stacked)

    # Identify best method
    best_method = max(results, key=lambda k: results[k][0])
    print(f"\nBest method based on F1 score: {best_method} ({results[best_method][0]:.4f})")

    # Save submissions and confusion matrices
    for name, (f1, oof_pred, test_proba) in results.items():
        # Save submission CSV
        if test_proba.shape[0] != len(test_df):
            raise ValueError(f"Prediction shape {test_proba.shape[0]} doesn't match test_df {len(test_df)}")

        final_pred = test_proba.argmax(axis=1)
        test_labels = [id_to_label[i] for i in final_pred]
        submission = pd.DataFrame({
            "id": test_df["id"].values,
            "emotion": test_labels
        })
        sub_path = os.path.join(save_dir, f"submission_{name}.csv")
        submission.to_csv(sub_path, index=False)
        print(f"Computed: {name}")

        # Compute confusion matrix for OOF predictions
        cm = confusion_matrix(ground_truth, oof_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(id_to_label.values()))
        disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
        plt.title(f"Confusion Matrix: {name}")
        plt.tight_layout()

        # Save confusion matrix figure
        cm_path = os.path.join(cm_dir, f"confusion_matrix_{name}.png")
        plt.savefig(cm_path)
        plt.close()  # Close figure to free memory
        print(f"Saved confusion matrix: {cm_path}")

def train_transformer_folds(
    model_checkpoint: str,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    labels_array: np.ndarray,
    label_to_id: Dict[str, int],
    id_to_label: Dict[int, str],
    n_splits: int = 5,
    learning_rate: float = 2e-5,
    num_epochs: int = 3,
    train_batch_size: int = 16,
    max_token_length: int = 192,
    weight_decay: float = 0.01,
    seed: int = 42,
    num_heads: int = 4,
    head_dropout_prob: Optional[float] = None,
    use_fp16: bool = True,
    use_bf16: bool = False,
    checkpoint_dir: str = "./new_trained_model"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train a transformer model with multi-head classification using stratified K-Fold CV.

    Parameters
    ----------
    model_checkpoint : str
        Hugging Face model checkpoint name.
    train_data : pd.DataFrame
        Training data with 'post' and 'emotion' columns.
    test_data : pd.DataFrame
        Test data with 'post' column.
    labels_array : np.ndarray
        Numeric labels array for training data.
    label_to_id : dict
        Mapping from label names to numeric IDs.
    id_to_label : dict
        Mapping from numeric IDs to label names.
    n_splits : int
        Number of CV folds.
    learning_rate : float
        Learning rate for optimizer.
    num_epochs : int
        Number of training epochs.
    train_batch_size : int
        Batch size for training.
    max_token_length : int
        Max token length for the tokenizer.
    weight_decay : float
        Weight decay for optimizer.
    seed : int
        Random seed for reproducibility.
    num_heads : int
        Number of heads for the multi-head classifier.
    head_dropout_prob : Optional[float]
        Dropout probability for classifier head; if None, uses model default.
    use_fp16 : bool
        Enable FP16 mixed precision training.
    use_bf16 : bool
        Enable BF16 mixed precision training.
    checkpoint_dir : str
        Directory to save model checkpoints.

    Returns
    -------
    out_of_fold_preds : np.ndarray
        Out-of-fold predictions for training data.
    test_preds_avg : np.ndarray
        Averaged predictions for test data across folds.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

    # Add special tokens if missing
    special_tokens = ["<USER>", "<URL>", "<NAME>"]
    missing_tokens = [tok for tok in special_tokens if tok not in tokenizer.get_vocab()]
    if missing_tokens:
        tokenizer.add_tokens(missing_tokens)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out_of_fold_preds = np.zeros((len(train_data), len(label_to_id)), dtype=np.float32)
    test_preds_avg = np.zeros((len(test_data), len(label_to_id)), dtype=np.float32)

    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(train_data, labels_array), 1):
        fold_dir = f"{checkpoint_dir}/{os.path.basename(model_checkpoint)}_train_{fold_idx}"
        print(f"\n[{model_checkpoint}] Trained: {fold_idx}/{n_splits}")

        # Split train/val
        train_subset = train_data.iloc[train_indices]
        val_subset = train_data.iloc[val_indices]

        train_texts = train_subset["post"].tolist()
        train_labels = train_subset["emotion"].map(label_to_id).tolist()

        val_texts = val_subset["post"].tolist()
        val_labels = val_subset["emotion"].map(label_to_id).tolist()

        # Create tokenized datasets
        tokenized_train = create_dataset(train_texts, train_labels).map(
            lambda ex: tokenize_batch(ex, tokenizer, max_token_length), batched=True
        )
        tokenized_val = create_dataset(val_texts, val_labels).map(
            lambda ex: tokenize_batch(ex, tokenizer, max_token_length), batched=True
        )
        tokenized_test = create_dataset(test_data["post"].tolist()).map(
            lambda ex: tokenize_batch(ex, tokenizer, max_token_length), batched=True
        )

        # Compute class weights
        class_weights = compute_class_weight("balanced", classes=np.unique(labels_array[train_indices]), y=labels_array[train_indices])
        class_weights = np.power(class_weights, 0.5)  # gamma=0.5

        # Set torch dataset format
        cols = ["input_ids", "attention_mask", "label"]
        tokenized_train.set_format(type="torch", columns=[c for c in cols if c in tokenized_train.column_names])
        tokenized_val.set_format(type="torch", columns=[c for c in cols if c in tokenized_val.column_names])
        tokenized_test.set_format(type="torch", columns=[c for c in ["input_ids", "attention_mask"] if c in tokenized_test.column_names])

        # -----------------------------
        # Initialize model
        # -----------------------------
        model_config = AutoConfig.from_pretrained(
            model_checkpoint,
            num_labels=len(label_to_id),
            label2id=label_to_id,
            id2label=id_to_label,
            hidden_dropout_prob=0.2,
            attention_probs_dropout_prob=0.2,
        )
        model_config.problem_type = "single_label_classification"

        model = AutoModelForSequenceClassification.from_pretrained(
            model_checkpoint, config=model_config, 
            ignore_mismatched_sizes=True, use_safetensors=True
        )
        model.resize_token_embeddings(len(tokenizer))
        if head_dropout_prob is None:
            head_dropout_prob = model_config.hidden_dropout_prob

        model.classifier = MultiHeadClassifier(
            input_dim=model_config.hidden_size,
            num_labels=len(label_to_id),
            n_heads=num_heads,
            dropout_prob=head_dropout_prob,
        )

        # -----------------------------
        # Training arguments
        # -----------------------------
        training_args = TrainingArguments(
            output_dir=fold_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=train_batch_size * 2,
            num_train_epochs=num_epochs,
            weight_decay=weight_decay,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            save_total_limit=1,
            logging_steps=50,
            greater_is_better=True,
            report_to="none",
            seed=seed + fold_idx,
            fp16=use_fp16 and not use_bf16,
            bf16=use_bf16,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            max_grad_norm=0.5,
            optim="adamw_torch",
        )

        trainer = WeightedTrainer(
            class_weight=class_weights,
            rdrop_alpha=0.5,
            layerwise_lr_decay=0.95,
            label_smoothing_factor=0.05,
            model=model,
            gamma=0.0,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            processing_class=tokenizer,
            data_collator=default_data_collator,
            compute_metrics=calculate_f1,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        # -----------------------------
        # Resume checkpoint if available
        # -----------------------------
        resume_ckpt = None
        print(f"Checking for existing checkpoints in {fold_dir}...")
        if os.path.isdir(fold_dir):
            checkpoints = sorted(glob.glob(os.path.join(fold_dir, "checkpoint-*")), key=lambda x: int(x.split("-")[-1]))
            print(f"Found {len(checkpoints)} checkpoints in {fold_dir}")
            if checkpoints:
                resume_ckpt = checkpoints[-1]

        if resume_ckpt:
            print(f"Resuming from checkpoint: {resume_ckpt}")
            trainer.train(resume_from_checkpoint=resume_ckpt)
        else:
            print("Training from scratch.")
            trainer.train()

        # -----------------------------
        # Make predictions
        # -----------------------------
        val_logits = trainer.predict(tokenized_val).predictions
        out_of_fold_preds[val_indices] = val_logits

        test_logits = trainer.predict(tokenized_test).predictions
        test_preds_avg += test_logits / n_splits

        val_pred_labels = np.argmax(val_logits, axis=1)
        fold_f1 = f1_score(val_labels, val_pred_labels, average="macro")
        print(f"Fold {fold_idx} Score: {fold_f1:.4f}")

        del trainer, model
        torch.cuda.empty_cache()

    # -----------------------------
    # Overall out-of-fold metrics
    # -----------------------------
    out_pred_labels = out_of_fold_preds.argmax(axis=1)
    print(f"\n[Model {model_checkpoint}] Out-of-Fold F1(macro): {f1_score(labels_array, out_pred_labels, average='macro'):.4f}")
    print(f"[Model {model_checkpoint}] Out-of-Fold F1(micro): {f1_score(labels_array, out_pred_labels, average='micro'):.4f}")

    return out_of_fold_preds, test_preds_avg

__all__ = [
    "simple_average_ensemble",
    "weighted_average_ensemble",
    "rank_average_ensemble",
    "stacked_ensemble",
    "run_all_ensembles",
    "create_dataset",
    "tokenize_batch",
    "calculate_f1",
    "WeightedTrainer",
    "MultiHeadClassifier",
    "FocalLoss",
    "get_encoder_layer_index",
    "train_transformer_folds",
]
