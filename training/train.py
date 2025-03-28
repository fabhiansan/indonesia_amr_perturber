"""
Training script for AMR entailment model using Indo-RoBERTa.
"""
from typing import Dict, List, Tuple, Optional, Any, Union
import os
import json
import logging
import argparse
import random
import numpy as np
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    RobertaTokenizer,
    RobertaConfig,
    get_linear_schedule_with_warmup,
    set_seed
)
from tqdm import tqdm, trange

from data_utils import (
    load_and_process_data,
    create_data_loaders,
    get_distribution_stats
)
from model import load_pretrained_model
from evaluation import (
    evaluate_model,
    evaluate_by_perturbation_type,
    plot_confusion_matrix,
    generate_classification_report
)

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def train(args):
    """
    Train the model with the given arguments.
    
    Args:
        args: Training arguments
    """
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info(f"Training on {device}")
    
    # Set seed for reproducibility
    if args.seed is not None:
        set_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
    
    # Set up output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Set up tensorboard
    tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "runs"))
    
    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    
    # Load dataset
    logger.info(f"Loading data from {args.data_path}")
    logger.info(f"Using summary field: {args.summary_field}")
    
    # Print dataset distribution
    logger.info("Dataset distribution:")
    stats_df = get_distribution_stats(args.data_path)
    logger.info("\n" + stats_df.to_string())
    
    # Load and process data
    train_dataset, val_dataset, test_dataset = load_and_process_data(
        data_path=args.data_path,
        tokenizer=tokenizer,
        test_size=args.test_size,
        val_size=args.val_size,
        max_length=args.max_seq_length,
        summary_field=args.summary_field,
        include_title=args.include_title,
        random_state=args.seed
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Load model
    logger.info(f"Loading model from {args.model_name_or_path}")
    model, config = load_pretrained_model(
        model_name_or_path=args.model_name_or_path,
        device=device
    )
    
    # Set up optimizer and scheduler
    # No weight decay for bias and LayerNorm parameters
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=args.adam_epsilon
    )
    
    # Calculate total training steps
    total_steps = len(train_loader) * args.num_train_epochs
    warmup_steps = int(total_steps * args.warmup_proportion)
    
    # Create scheduler with linear warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size = {args.batch_size}")
    logger.info(f"  Total optimization steps = {total_steps}")
    logger.info(f"  Warmup steps = {warmup_steps}")
    
    global_step = 0
    best_val_metric = 0.0
    
    # Training progress
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_loader, desc="Iteration")
        total_loss = 0.0
        model.train()
        
        for step, batch in enumerate(epoch_iterator):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs["loss"]
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            
            # Update tracking
            total_loss += loss.item()
            global_step += 1
            epoch_iterator.set_description(f"Loss: {loss.item():.4f}")
            
            # Log to tensorboard
            if global_step % args.logging_steps == 0:
                tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                tb_writer.add_scalar("loss", loss.item(), global_step)
            
            # Evaluate during training
            if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                # Evaluate on validation set
                val_metrics = evaluate_model(
                    model=model,
                    dataloader=val_loader,
                    device=device,
                    threshold=args.threshold
                )
                
                # Log validation metrics
                for metric_name, metric_value in val_metrics.items():
                    tb_writer.add_scalar(f"eval_{metric_name}", metric_value, global_step)
                
                logger.info(f"Eval at step {global_step}:")
                for metric_name, metric_value in val_metrics.items():
                    logger.info(f"  {metric_name} = {metric_value:.4f}")
                
                # Save the best model
                current_metric = val_metrics[args.eval_metric]
                if current_metric > best_val_metric:
                    best_val_metric = current_metric
                    logger.info(f"New best {args.eval_metric}: {best_val_metric:.4f}")
                    
                    # Save model
                    model_to_save = model.module if hasattr(model, "module") else model
                    output_dir = os.path.join(args.output_dir, "best_model")
                    os.makedirs(output_dir, exist_ok=True)
                    
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    
                    # Save training arguments
                    with open(os.path.join(output_dir, "training_args.json"), "w") as f:
                        json.dump(vars(args), f, indent=4)
                
                # Back to training mode
                model.train()
        
        # Calculate average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{args.num_train_epochs} - Average Loss: {avg_loss:.4f}")
        
        # Evaluate at the end of each epoch
        logger.info(f"Evaluating at end of epoch {epoch+1}")
        val_metrics = evaluate_model(
            model=model,
            dataloader=val_loader,
            device=device,
            threshold=args.threshold
        )
        
        # Log validation metrics
        for metric_name, metric_value in val_metrics.items():
            tb_writer.add_scalar(f"epoch_eval_{metric_name}", metric_value, epoch)
        
        logger.info(f"Epoch {epoch+1} evaluation:")
        for metric_name, metric_value in val_metrics.items():
            logger.info(f"  {metric_name} = {metric_value:.4f}")
        
        # Save checkpoint
        if args.save_every_epoch:
            # Save model
            model_to_save = model.module if hasattr(model, "module") else model
            output_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}")
            os.makedirs(output_dir, exist_ok=True)
            
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            # Save training arguments
            with open(os.path.join(output_dir, "training_args.json"), "w") as f:
                json.dump(vars(args), f, indent=4)
    
    # Final evaluation on test set
    logger.info("***** Final evaluation on test set *****")
    
    # Load best model for final evaluation
    best_model_path = os.path.join(args.output_dir, "best_model")
    if os.path.exists(best_model_path):
        logger.info(f"Loading best model from {best_model_path}")
        model, _ = load_pretrained_model(
            model_name_or_path=best_model_path,
            device=device
        )
    
    # Evaluate on test set
    test_metrics = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
        threshold=args.threshold
    )
    
    logger.info("Test set evaluation:")
    for metric_name, metric_value in test_metrics.items():
        logger.info(f"  {metric_name} = {metric_value:.4f}")
    
    # Evaluate by perturbation type
    logger.info("Evaluating by perturbation type...")
    perturbation_metrics = evaluate_by_perturbation_type(
        model=model,
        dataloader=test_loader,
        device=device,
        threshold=args.threshold
    )
    
    logger.info("Metrics by perturbation type:")
    for pert_type, metrics in perturbation_metrics.items():
        pert_name = "Original" if pert_type is None else pert_type
        logger.info(f"  {pert_name} (count: {metrics['count']}):")
        for metric_name, metric_value in metrics.items():
            if metric_name != "count":
                logger.info(f"    {metric_name} = {metric_value:.4f}")
    
    # Generate confusion matrix
    logger.info("Generating confusion matrix...")
    confusion_matrix_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plot_confusion_matrix(
        model=model,
        dataloader=test_loader,
        device=device,
        threshold=args.threshold,
        save_path=confusion_matrix_path
    )
    
    # Generate classification report
    logger.info("Generating classification report...")
    report = generate_classification_report(
        model=model,
        dataloader=test_loader,
        device=device,
        threshold=args.threshold
    )
    
    logger.info(f"Classification report:\n{report}")
    
    # Save test results
    test_results = {
        "metrics": test_metrics,
        "perturbation_metrics": perturbation_metrics,
        "classification_report": report
    }
    
    with open(os.path.join(args.output_dir, "test_results.json"), "w") as f:
        json.dump(test_results, f, indent=4)
    
    logger.info(f"All results saved to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser()
    
    # Data parameters
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the JSON data file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save model checkpoints and results")
    parser.add_argument("--summary_field", type=str, default="amr",
                        help="Field containing summary text (amr or perturbed_summary)")
    parser.add_argument("--include_title", action="store_true",
                        help="Whether to include title in source text")
    parser.add_argument("--test_size", type=float, default=0.1,
                        help="Fraction of data to use for testing")
    parser.add_argument("--val_size", type=float, default=0.1,
                        help="Fraction of data to use for validation")
    
    # Model parameters
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Maximum sequence length for tokenization")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for binary classification")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training and evaluation")
    parser.add_argument("--num_train_epochs", type=float, default=3.0,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8,
                        help="Epsilon for Adam optimizer")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm for gradient clipping")
    parser.add_argument("--warmup_proportion", type=float, default=0.1,
                        help="Proportion of training steps for linear warmup")
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="Log training metrics every X steps")
    parser.add_argument("--eval_steps", type=int, default=1000,
                        help="Evaluate every X steps (0 to disable)")
    parser.add_argument("--eval_metric", type=str, default="f1",
                        help="Metric to use for selecting best model (accuracy, precision, recall, f1, auc)")
    parser.add_argument("--save_every_epoch", action="store_true",
                        help="Save checkpoint after every epoch")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Disable CUDA even when available")
    
    args = parser.parse_args()
    
    # Start training
    train(args)


if __name__ == "__main__":
    main()
