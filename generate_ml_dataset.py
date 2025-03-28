#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate a labeled dataset for machine learning from AMR data.
The script creates a dataset with original AMRs (labeled 1) and their perturbed versions (labeled 0).
"""

import json
import argparse
import random
import os
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import penman
from tqdm import tqdm
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('perturbation.log')
    ]
)
logger = logging.getLogger(__name__)

# Import perturbation functions
perturbation_modules_loaded = False

try:
    # First try direct imports from individual modules
    from data_perturber.predicates_perturber import insertWrongPredicates
    from data_perturber.circumstance_perturber import insertCircumstanceError
    from data_perturber.entity_perturber import EntityError
    from data_perturber.discourse_perturber import insertDiscourseError
    from data_perturber.out_of_article_perturber import insertOutOfArticleError
    perturbation_modules_loaded = True
    logger.info("Loaded perturbation modules from individual files")
except ImportError as e:
    logger.warning(f"Could not import from individual modules: {e}")
    try:
        # Try importing through the insertion wrapper module
        from data_perturber.insertion import (
            predicate_error_insertion,
            circumstance_error_insertion,
            entity_error_insertion,
            discourse_error_insertion,
            out_of_article_error_insertion
        )
        perturbation_modules_loaded = True
        logger.info("Loaded perturbation modules from data_perturber.insertion")
    except ImportError as e:
        logger.error(f"Failed to import perturbation modules: {e}")
        raise ImportError("Could not import perturbation modules. Please check your installation.")

if not perturbation_modules_loaded:
    raise ImportError("Could not load any perturbation modules")


def clean_amr_string(amr_string: str) -> str:
    """
    Clean AMR string by removing comments and metadata.
    
    Args:
        amr_string: The AMR string to clean
        
    Returns:
        Cleaned AMR string
    """
    clean_lines = []
    for line in amr_string.split('\n'):
        if not line.strip() or line.strip().startswith('#'):
            continue
        clean_lines.append(line)
    return '\n'.join(clean_lines)


def apply_perturbation(amr_graph: penman.Graph, perturbation_type: str) -> Tuple[Optional[penman.Graph], Dict[str, Any]]:
    """
    Apply a specific type of perturbation to an AMR graph.
    
    Args:
        amr_graph: The AMR graph to perturb
        perturbation_type: Type of perturbation to apply
        
    Returns:
        Tuple of (perturbed_graph, changelog)
    """
    # Map perturbation types to functions
    try:
        if 'predicate_error_insertion' in globals():
            # Using wrapper functions from insertion module
            perturber_map = {
                "predicate": predicate_error_insertion,
                "circumstance": circumstance_error_insertion, 
                "entity": entity_error_insertion,
                "discourse": discourse_error_insertion,
                "out_of_article": out_of_article_error_insertion
            }
        else:
            # Using direct functions from individual modules
            perturber_map = {
                "predicate": lambda g: insertWrongPredicates(g),
                "circumstance": lambda g: insertCircumstanceError(g, "both"),
                "entity": lambda g: (EntityError(g), {"perturber": "entity"}),
                "discourse": insertDiscourseError,
                "out_of_article": insertOutOfArticleError
            }
    except Exception as e:
        error_msg = f"Error setting up perturber map: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if perturbation_type not in perturber_map:
        raise ValueError(f"Unknown perturbation type: {perturbation_type}")
    
    perturber_func = perturber_map[perturbation_type]
    
    try:
        # Try to apply the perturbation
        logger.debug(f"Applying {perturbation_type} perturbation")
        
        # Special handling for entity errors since it has a different return structure
        if perturbation_type == "entity" and "EntityError" in globals():
            perturbed_graph = perturber_func(amr_graph)
            changelog = {"perturber": "entity"}
        else:
            perturbed_graph, changelog = perturber_func(amr_graph)
        
        # Convert list changelog to dict if necessary
        if isinstance(changelog, list):
            logger.debug(f"Converting list changelog to dictionary: {changelog}")
            changelog_dict = {
                "perturber": perturbation_type,
                "changes": changelog
            }
            changelog = changelog_dict
        
        # If there's an error in the changelog, consider it a failure
        if isinstance(changelog, dict) and "error" in changelog:
            error_msg = f"Error in perturbation: {changelog['error']}"
            logger.warning(error_msg)
            raise ValueError(error_msg)
            
        return perturbed_graph, changelog
    except Exception as e:
        error_msg = f"Exception in {perturbation_type} perturber: {str(e)}"
        logger.warning(error_msg)
        changelog = {
            "error": error_msg,
            "perturber": perturbation_type
        }
        return None, changelog


def generate_perturbed_amr(
    amr_string: str, 
    perturbation_weights: Dict[str, float],
    stats: Dict[str, Dict[str, int]]
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Generate a perturbed version of an AMR string using weighted random selection.
    
    Args:
        amr_string: The original AMR string
        perturbation_weights: Weights for each perturbation type
        stats: Dictionary to track perturbation statistics
        
    Returns:
        Tuple of (perturbed_amr_string, changelog)
    """
    # Parse the AMR
    try:
        # Clean the AMR string first
        clean_amr = clean_amr_string(amr_string)
        amr_graph = penman.decode(clean_amr)
    except Exception as e:
        # If cleaning fails, try with original string
        try:
            amr_graph = penman.decode(amr_string)
        except Exception as e2:
            error_msg = f"Failed to parse AMR: {str(e2)}"
            logger.error(error_msg)
            stats["parsing"]["failure"] += 1
            return None, {"error": error_msg}
    
    stats["parsing"]["success"] += 1
    
    # Normalize weights
    total_weight = sum(perturbation_weights.values())
    if abs(total_weight - 1.0) > 1e-10:
        perturbation_weights = {k: v / total_weight for k, v in perturbation_weights.items()}
    
    # Filter out perturbation types with zero weight
    available_perturbations = {k: v for k, v in perturbation_weights.items() if v > 0}
    
    if not available_perturbations:
        error_msg = "No perturbation types available (all weights are zero)"
        logger.error(error_msg)
        return None, {"error": error_msg}
    
    # Select a perturbation type randomly based on weights
    perturbation_type = random.choices(
        list(available_perturbations.keys()),
        weights=list(available_perturbations.values()),
        k=1
    )[0]
    
    logger.debug(f"Selected perturbation type: {perturbation_type}")
    stats["selection"][perturbation_type] += 1
    
    # Apply the selected perturbation
    perturbed_graph, changelog = apply_perturbation(amr_graph, perturbation_type)
    
    if perturbed_graph is None:
        stats["perturbation"][perturbation_type]["failure"] += 1
        logger.debug(f"Primary perturbation {perturbation_type} failed, trying alternatives")
        
        # If the perturbation failed, try a different type (in order of weights)
        for retry_type, _ in sorted(available_perturbations.items(), key=lambda x: x[1], reverse=True):
            if retry_type == perturbation_type:
                continue
                
            logger.debug(f"Trying alternative perturbation: {retry_type}")
            perturbed_graph, retry_changelog = apply_perturbation(amr_graph, retry_type)
            
            if perturbed_graph is not None:
                stats["perturbation"][retry_type]["success"] += 1
                stats["fallback"][retry_type] += 1
                changelog = retry_changelog
                if isinstance(changelog, dict):
                    changelog["perturber"] = retry_type  # Note that we changed perturber
                    changelog["fallback"] = True
                else:
                    # Handle list changelog
                    changelog = {
                        "perturber": retry_type,
                        "changes": changelog,
                        "fallback": True
                    }
                logger.debug(f"Alternative perturbation {retry_type} succeeded")
                break
            else:
                stats["perturbation"][retry_type]["failure"] += 1
    else:
        stats["perturbation"][perturbation_type]["success"] += 1
        # Ensure changelog has perturber field
        if isinstance(changelog, dict) and "perturber" not in changelog:
            changelog["perturber"] = perturbation_type
    
    # If we got a valid graph, encode it back to a string
    if perturbed_graph is not None:
        try:
            perturbed_amr_string = penman.encode(perturbed_graph)
            stats["encoding"]["success"] += 1
            return perturbed_amr_string, changelog
        except Exception as e:
            error_msg = f"Error encoding perturbed graph: {str(e)}"
            logger.error(error_msg)
            stats["encoding"]["failure"] += 1
            return None, {"error": error_msg}
    
    return None, changelog


def generate_dataset(
    input_file: str,
    output_file: str,
    perturbation_weights: Dict[str, float],
    perturbed_per_original: int = 1,
    amr_field: str = "summary_amr",
    seed: Optional[int] = None,
    max_examples: Optional[int] = None,
    debug_sample: Optional[int] = None
) -> None:
    """
    Generate a labeled dataset for machine learning.
    
    Args:
        input_file: Path to input JSON file with AMR data
        output_file: Path to output JSON file for the dataset
        perturbation_weights: Weights for each perturbation type
        perturbed_per_original: Number of perturbed examples to generate per original
        amr_field: Field in input data containing the AMR string
        seed: Random seed for reproducibility
        max_examples: Maximum number of examples to process (for testing)
        debug_sample: Number of samples to debug in detail
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Load input data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if max_examples is not None:
        data = data[:max_examples]
    
    # Prepare output dataset
    output_data = []
    
    # Initialize statistics
    stats = {
        "parsing": {"success": 0, "failure": 0},
        "encoding": {"success": 0, "failure": 0},
        "selection": {pert_type: 0 for pert_type in perturbation_weights.keys()},
        "perturbation": {
            pert_type: {"success": 0, "failure": 0} 
            for pert_type in perturbation_weights.keys()
        },
        "fallback": {pert_type: 0 for pert_type in perturbation_weights.keys()},
        "total_attempts": 0,
        "total_successful": 0,
        "total_failures": 0,
    }
    
    logger.info(f"Starting to process {len(data)} examples with {perturbed_per_original} perturbations each")
    logger.info(f"Perturbation weights: {perturbation_weights}")
    
    # Process each item
    for i, item in enumerate(tqdm(data, desc="Generating dataset")):
        # Skip if AMR field is missing
        if amr_field not in item or not item[amr_field]:
            continue
        
        amr_string = item[amr_field]
        
        # Add the original AMR example (labeled as correct = 1)
        original_example = {
            "id": f"{i}_original",
            "amr": amr_string,
            "score": 1.0,  # Original AMR is correct
            "perturbation_type": None,
            "source_id": item.get("id", str(i)),
            "source_text": item.get("source_text", ""),  # Include source text
            "title": item.get("title", ""),              # Include title
            "target_summary": item.get("target_summary", "")  # Include target summary
        }
        output_data.append(original_example)
        
        # Debug first few examples in detail if requested
        if debug_sample is not None and i < debug_sample:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        
        # Generate perturbed versions
        successful_perturbations = 0
        max_attempts = perturbed_per_original * 3  # Allow more attempts to hit target
        attempts = 0
        
        while successful_perturbations < perturbed_per_original and attempts < max_attempts:
            attempts += 1
            stats["total_attempts"] += 1
            
            perturbed_amr, changelog = generate_perturbed_amr(amr_string, perturbation_weights, stats)
            
            # If perturbation failed, skip this example
            if perturbed_amr is None:
                if isinstance(changelog, dict) and "error" in changelog:
                    logger.debug(f"Perturbation attempt {attempts} failed: {changelog['error']}")
                stats["total_failures"] += 1
                continue
            
            successful_perturbations += 1
            stats["total_successful"] += 1
            
            # Ensure changelog is a dictionary
            if not isinstance(changelog, dict):
                logger.debug(f"Converting non-dictionary changelog to dictionary: {type(changelog)}")
                changelog = {
                    "perturber": "unknown",
                    "changes": changelog
                }
            
            # Add the perturbed example (labeled as incorrect = 0)
            perturbed_example = {
                "id": f"{i}_perturbed_{successful_perturbations}",
                "amr": perturbed_amr,
                "score": 0.0,  # Perturbed AMR is incorrect
                "perturbation_type": changelog.get("perturber", "unknown"),
                "source_id": item.get("id", str(i)),
                "changelog": changelog,
                "source_text": item.get("source_text", ""),  # Include source text
                "title": item.get("title", ""),              # Include title 
                "target_summary": item.get("target_summary", "")  # Include target summary
            }
            output_data.append(perturbed_example)
            
            if debug_sample is not None and i < debug_sample:
                logger.debug(f"Successfully created perturbation {successful_perturbations}/{perturbed_per_original}")
        
        if successful_perturbations < perturbed_per_original:
            logger.warning(f"Could only generate {successful_perturbations}/{perturbed_per_original} perturbations for example {i}")
    
    # Save the output dataset
    random.shuffle(output_data)  # Shuffle to avoid bias in training
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Generated dataset with {len(output_data)} examples saved to {output_file}")
    
    # Report statistics
    originals = sum(1 for ex in output_data if ex["score"] == 1.0)
    perturbed = sum(1 for ex in output_data if ex["score"] == 0.0)
    
    print(f"Statistics:")
    print(f"  Original AMRs: {originals}")
    print(f"  Perturbed AMRs: {perturbed}")
    
    # Count by perturbation type
    perturbation_counts = {}
    for ex in output_data:
        if ex["score"] == 0.0:
            pert_type = ex.get("perturbation_type", "unknown")
            perturbation_counts[pert_type] = perturbation_counts.get(pert_type, 0) + 1
    
    if perturbed > 0:
        print("Perturbation types:")
        for pert_type, count in sorted(perturbation_counts.items()):
            print(f"  {pert_type}: {count} ({count/perturbed*100:.1f}%)")
    
    # Print debugging statistics
    success_rate = stats["total_successful"] / max(1, stats["total_attempts"]) * 100
    print(f"\nDetailed Perturbation Statistics:")
    print(f"  Success rate: {success_rate:.1f}% ({stats['total_successful']}/{stats['total_attempts']} attempts)")
    print(f"  Parsing: {stats['parsing']['success']} successful, {stats['parsing']['failure']} failed")
    print(f"  Encoding: {stats['encoding']['success']} successful, {stats['encoding']['failure']} failed")
    
    print("\nPerturbation type selection:")
    for pert_type, count in sorted(stats["selection"].items()):
        print(f"  {pert_type}: {count} times")
    
    print("\nPerturbation success rates by type:")
    for pert_type in perturbation_weights.keys():
        success = stats["perturbation"][pert_type]["success"]
        failure = stats["perturbation"][pert_type]["failure"]
        total = success + failure
        if total > 0:
            rate = success / total * 100
            print(f"  {pert_type}: {rate:.1f}% ({success}/{total})")
    
    print("\nFallback usage:")
    total_fallbacks = sum(stats["fallback"].values())
    if total_fallbacks > 0:
        for pert_type, count in sorted(stats["fallback"].items()):
            if count > 0:
                print(f"  {pert_type}: {count} times ({count/total_fallbacks*100:.1f}%)")


def main():
    """Command-line interface for generating the dataset."""
    parser = argparse.ArgumentParser(description="Generate a labeled dataset for machine learning from AMR data")
    
    parser.add_argument("input", help="Input JSON file with AMR data")
    parser.add_argument("output", help="Output JSON file for the dataset")
    parser.add_argument("--predicate", "-p", type=float, default=0.2, help="Weight for predicate errors")
    parser.add_argument("--circumstance", "-c", type=float, default=0.2, help="Weight for circumstance errors")
    parser.add_argument("--entity", "-e", type=float, default=0.2, help="Weight for entity errors")
    parser.add_argument("--discourse", "-d", type=float, default=0.2, help="Weight for discourse errors")
    parser.add_argument("--out-of-article", "-o", type=float, default=0.2, help="Weight for out-of-article errors")
    parser.add_argument("--perturbed-per-original", "-n", type=int, default=1, 
                        help="Number of perturbed examples to generate per original")
    parser.add_argument("--amr-field", type=str, default="summary_amr", 
                        help="Field in input data containing the AMR string")
    parser.add_argument("--seed", "-s", type=int, help="Random seed for reproducibility")
    parser.add_argument("--max-examples", "-m", type=int, help="Maximum number of examples to process (for testing)")
    parser.add_argument("--split", action="store_true", 
                        help="Split the output into train/dev/test sets (80/10/10 split)")
    parser.add_argument("--output-dir", help="Directory to save the split datasets (required if --split is used)")
    parser.add_argument("--debug-sample", type=int, default=0, 
                        help="Number of examples to debug in detail (0 for none)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Collect perturbation weights
    perturbation_weights = {
        "predicate": args.predicate,
        "circumstance": args.circumstance,
        "entity": args.entity,
        "discourse": args.discourse,
        "out_of_article": args.out_of_article
    }
    
    if args.split and not args.output_dir:
        parser.error("--output-dir is required when --split is specified")
    
    if not args.split:
        # Generate a single dataset
        generate_dataset(
            args.input,
            args.output,
            perturbation_weights,
            perturbed_per_original=args.perturbed_per_original,
            amr_field=args.amr_field,
            seed=args.seed,
            max_examples=args.max_examples,
            debug_sample=args.debug_sample
        )
    else:
        # Generate a full dataset first
        temp_output = os.path.join(os.path.dirname(args.output), "temp_full_dataset.json")
        
        generate_dataset(
            args.input,
            temp_output,
            perturbation_weights,
            perturbed_per_original=args.perturbed_per_original,
            amr_field=args.amr_field,
            seed=args.seed,
            max_examples=args.max_examples,
            debug_sample=args.debug_sample
        )
        
        # Load the full dataset
        with open(temp_output, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Shuffle the data
        if args.seed is not None:
            random.seed(args.seed)
        random.shuffle(full_data)
        
        # Split the data (80/10/10)
        n = len(full_data)
        train_size = int(0.8 * n)
        dev_size = int(0.1 * n)
        
        train_data = full_data[:train_size]
        dev_data = full_data[train_size:train_size+dev_size]
        test_data = full_data[train_size+dev_size:]
        
        # Save the splits
        with open(os.path.join(args.output_dir, "train.json"), 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
        
        with open(os.path.join(args.output_dir, "dev.json"), 'w', encoding='utf-8') as f:
            json.dump(dev_data, f, indent=2, ensure_ascii=False)
        
        with open(os.path.join(args.output_dir, "test.json"), 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)
        
        # Clean up the temporary file
        os.remove(temp_output)


if __name__ == "__main__":
    main()
