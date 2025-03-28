"""
Evaluation utilities for the AMR entailment model.
"""
from typing import Dict, List, Tuple, Optional, Any, Union
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    classification_report
)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from torch.utils.data import DataLoader
from transformers import PreTrainedModel


def evaluate_model(
    model: PreTrainedModel,
    dataloader: DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate a model on a dataset.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader for the evaluation dataset
        device: Device to run evaluation on
        threshold: Threshold for binary classification
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_scores = []
    total_loss = 0.0
    
    # Disable gradient computation for evaluation
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Get model outputs
            outputs = model(**batch)
            loss = outputs["loss"]
            logits = outputs["logits"]
            
            # Add batch loss to total
            total_loss += loss.item() * len(batch["input_ids"])
            
            # Convert logits to predictions
            scores = logits.view(-1).cpu().numpy()
            preds = (scores >= threshold).astype(int)
            
            # Save predictions and labels
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].view(-1).cpu().numpy())
            all_scores.extend(scores)
    
    # Calculate metrics
    average_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary"
    )
    
    # Calculate AUC if there are both positive and negative examples
    auc = 0.0
    if len(set(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_scores)
    
    # Return metrics
    metrics = {
        "loss": average_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }
    
    return metrics


def evaluate_by_perturbation_type(
    model: PreTrainedModel,
    dataloader: DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    threshold: float = 0.5
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model performance broken down by perturbation type.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader for the evaluation dataset
        device: Device to run evaluation on
        threshold: Threshold for binary classification
        
    Returns:
        Dictionary with metrics for each perturbation type
    """
    model.eval()
    
    # Create dictionaries to store predictions and labels for each perturbation type
    perturbation_preds = {}
    perturbation_labels = {}
    perturbation_scores = {}
    
    # Disable gradient computation for evaluation
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating by perturbation")):
            # Get the perturbation types for this batch
            # We need to access the original data since perturbation_type isn't part of the model inputs
            examples = [dataloader.dataset.data[idx] for idx in range(
                batch_idx * dataloader.batch_size,
                min((batch_idx + 1) * dataloader.batch_size, len(dataloader.dataset))
            )]
            
            pert_types = [ex.get("perturbation_type") for ex in examples]
            
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Get model outputs
            outputs = model(**batch)
            logits = outputs["logits"]
            
            # Convert logits to predictions
            scores = logits.view(-1).cpu().numpy()
            preds = (scores >= threshold).astype(int)
            labels = batch["labels"].view(-1).cpu().numpy()
            
            # Store predictions and labels by perturbation type
            for i, pert_type in enumerate(pert_types):
                if i >= len(preds):
                    continue  # Skip if index out of range (last batch might be smaller)
                    
                if pert_type not in perturbation_preds:
                    perturbation_preds[pert_type] = []
                    perturbation_labels[pert_type] = []
                    perturbation_scores[pert_type] = []
                
                perturbation_preds[pert_type].append(preds[i])
                perturbation_labels[pert_type].append(labels[i])
                perturbation_scores[pert_type].append(scores[i])
    
    # Calculate metrics for each perturbation type
    metrics_by_type = {}
    for pert_type in perturbation_preds:
        # Skip if there are not enough examples
        if len(perturbation_preds[pert_type]) < 2:
            continue
            
        # Calculate metrics
        accuracy = accuracy_score(perturbation_labels[pert_type], perturbation_preds[pert_type])
        precision, recall, f1, _ = precision_recall_fscore_support(
            perturbation_labels[pert_type], perturbation_preds[pert_type], average="binary", zero_division=0
        )
        
        # Calculate AUC if there are both positive and negative examples
        auc = 0.0
        label_set = set(perturbation_labels[pert_type])
        if len(label_set) > 1:
            auc = roc_auc_score(perturbation_labels[pert_type], perturbation_scores[pert_type])
        
        # Store metrics
        metrics_by_type[pert_type] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
            "count": len(perturbation_preds[pert_type])
        }
    
    return metrics_by_type


def plot_confusion_matrix(
    model: PreTrainedModel,
    dataloader: DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    threshold: float = 0.5,
    save_path: Optional[str] = None
) -> None:
    """
    Plot the confusion matrix for the model predictions.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader for the evaluation dataset
        device: Device to run evaluation on
        threshold: Threshold for binary classification
        save_path: Path to save the plot to
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    # Disable gradient computation for evaluation
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating predictions"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Get model outputs
            outputs = model(**batch)
            logits = outputs["logits"]
            
            # Convert logits to predictions
            preds = (logits.view(-1).cpu().numpy() >= threshold).astype(int)
            
            # Save predictions and labels
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].view(-1).cpu().numpy())
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Entails", "Entails"],
        yticklabels=["Not Entails", "Entails"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    
    # Save plot if requested
    if save_path:
        plt.savefig(save_path)
        
    plt.close()


def generate_classification_report(
    model: PreTrainedModel,
    dataloader: DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    threshold: float = 0.5,
) -> str:
    """
    Generate a detailed classification report.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader for the evaluation dataset
        device: Device to run evaluation on
        threshold: Threshold for binary classification
        
    Returns:
        Classification report as a string
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    # Disable gradient computation for evaluation
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating predictions"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Get model outputs
            outputs = model(**batch)
            logits = outputs["logits"]
            
            # Convert logits to predictions
            preds = (logits.view(-1).cpu().numpy() >= threshold).astype(int)
            
            # Save predictions and labels
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].view(-1).cpu().numpy())
    
    # Generate report
    report = classification_report(
        all_labels, 
        all_preds,
        target_names=["Not Entails", "Entails"],
        digits=4
    )
    
    return report
