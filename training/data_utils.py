"""
Data utilities for loading and processing AMR entailment datasets.
"""
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import os
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class AMREntailmentDataset(Dataset):
    """
    Dataset for AMR entailment task with source documents and summaries.
    
    This dataset is designed to handle examples where we need to determine if 
    summaries (generated from AMR) entail the source document.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data: List[Dict[str, Any]],
        max_length: int = 512,
        summary_field: str = "amr",  # Default is AMR, can be switched to "perturbed_summary"
        include_title: bool = True
    ):
        """
        Initialize the dataset.
        
        Args:
            tokenizer: Tokenizer for encoding text
            data: List of examples with source_text, title, and AMR/summary
            max_length: Maximum length for tokenization
            summary_field: Field containing the summary text (either "amr" or "perturbed_summary")
            include_title: Whether to prepend title to source text
        """
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length
        self.summary_field = summary_field
        self.include_title = include_title
        
        # Check which field to use for summaries
        if summary_field not in ["amr", "perturbed_summary"]:
            logger.warning(f"Unknown summary field: {summary_field}, falling back to 'amr'")
            self.summary_field = "amr"
            
        # Check if we need to handle AMR directly
        self.has_generated_summaries = "perturbed_summary" in data[0] if data else False
        
        # Log dataset information
        logger.info(f"Created dataset with {len(data)} examples")
        logger.info(f"Using {self.summary_field} field for summaries")
        if not self.has_generated_summaries and self.summary_field == "perturbed_summary":
            logger.warning("'perturbed_summary' requested but not found in data, will use AMR directly")
            
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        Get an encoded example from the dataset.
        
        Returns a dictionary with tensors ready for the model:
            - input_ids: token IDs for the combined text
            - attention_mask: attention mask for padding
            - labels: binary label (1=entails, 0=not entails)
        """
        item = self.data[idx]
        
        # Prepare source text (optionally with title)
        source_text = item["source_text"]
        if self.include_title and item.get("title"):
            source_text = f"{item['title']}\n\n{source_text}"
            
        # Get the summary text
        if self.summary_field == "perturbed_summary" and self.has_generated_summaries:
            summary_text = item["perturbed_summary"]
        else:
            # If using AMR directly or perturbed_summary not available
            summary_text = item["amr"]
            
        # Create entailment pair structure
        text_pair = (source_text, summary_text)
        
        # Tokenize the text pair
        encoding = self.tokenizer(
            text_pair[0],
            text_pair[1],
            max_length=self.max_length,
            padding="max_length",
            truncation="longest_first",
            return_tensors="pt"
        )
        
        # Convert dict of tensors (each with batch dim of 1) to dict of 1D tensors
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        
        # Add label
        encoding["labels"] = torch.tensor(item["score"], dtype=torch.float)
        
        return encoding


def load_and_process_data(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    test_size: float = 0.1,
    val_size: float = 0.1,
    max_length: int = 512,
    summary_field: str = "amr",
    include_title: bool = True,
    random_state: int = 42
) -> Tuple[AMREntailmentDataset, AMREntailmentDataset, AMREntailmentDataset]:
    """
    Load and split data for training, validation, and testing.
    
    Args:
        data_path: Path to the JSON data file
        tokenizer: Tokenizer for encoding text
        test_size: Fraction of data to use for testing
        val_size: Fraction of data to use for validation
        max_length: Maximum length for tokenization
        summary_field: Field to use for summaries
        include_title: Whether to include title in source text
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Load data
    logger.info(f"Loading data from {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Split data into train, val, test
    train_data, test_data = train_test_split(
        data, test_size=test_size, random_state=random_state
    )
    
    train_data, val_data = train_test_split(
        train_data, test_size=val_size/(1-test_size), random_state=random_state
    )
    
    logger.info(f"Split data into {len(train_data)} train, {len(val_data)} val, {len(test_data)} test examples")
    
    # Create datasets
    train_dataset = AMREntailmentDataset(
        tokenizer=tokenizer,
        data=train_data,
        max_length=max_length,
        summary_field=summary_field,
        include_title=include_title
    )
    
    val_dataset = AMREntailmentDataset(
        tokenizer=tokenizer,
        data=val_data,
        max_length=max_length,
        summary_field=summary_field,
        include_title=include_title
    )
    
    test_dataset = AMREntailmentDataset(
        tokenizer=tokenizer,
        data=test_data,
        max_length=max_length, 
        summary_field=summary_field,
        include_title=include_title
    )
    
    return train_dataset, val_dataset, test_dataset


def create_data_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = 8,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size for training and evaluation
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


def get_distribution_stats(data_path: str) -> pd.DataFrame:
    """
    Get statistics about the dataset distribution.
    
    Args:
        data_path: Path to the JSON data file
        
    Returns:
        DataFrame with distribution statistics
    """
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Create a DataFrame
    df = pd.DataFrame(data)
    
    # Get basic statistics
    stats = {
        "Total examples": len(df),
        "Entailment (score=1.0)": len(df[df["score"] == 1.0]),
        "Non-entailment (score=0.0)": len(df[df["score"] == 0.0]),
    }
    
    # Get perturbation type distribution
    if "perturbation_type" in df.columns:
        pert_counts = df["perturbation_type"].value_counts().to_dict()
        for pert_type, count in pert_counts.items():
            if pert_type:  # Skip None values
                stats[f"Perturbation: {pert_type}"] = count
    
    return pd.DataFrame(stats.items(), columns=["Metric", "Count"])
