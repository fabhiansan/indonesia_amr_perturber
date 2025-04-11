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
import logging.handlers # Import handlers

# --- Custom Buffering Error Handler ---
class BufferingErrorHandler(logging.Handler):
    def __init__(self, filename, capacity=10, mode='a', encoding=None, delay=False):
        super().__init__()
        self.filename = filename
        self.capacity = capacity
        self.mode = mode
        self.encoding = encoding
        self.delay = delay
        self.buffer = []
        self.error_count = 0
        self.stream = None
        if not delay:
            self._open()

    def _open(self):
        if self.stream is None:
             self.stream = open(self.filename, self.mode, encoding=self.encoding)

    def close(self):
        self.acquire()
        try:
            self.flush() # Flush any remaining messages
            if self.stream and not self.stream.closed:
                self.stream.close()
            self.stream = None
            super().close()
        finally:
            self.release()

    def flush(self):
        self.acquire()
        try:
            if self.stream and self.buffer:
                # Ensure stream is open if delayed
                self._open()
                try:
                    for record in self.buffer:
                        msg = self.format(record)
                        self.stream.write(msg + self.terminator)
                    self.stream.flush()
                    self.buffer = []
                    self.error_count = 0 # Reset count after flushing
                except Exception:
                    self.handleError(record) # Use default error handling
        finally:
            self.release()

    def emit(self, record):
        # Ensure stream is open if delayed
        self._open()
        self.acquire()
        try:
            # Only buffer ERROR and CRITICAL messages
            if record.levelno >= logging.ERROR:
                self.buffer.append(record)
                self.error_count += 1
                if self.error_count >= self.capacity:
                    self.flush()
            else:
                # For non-error messages handled by this handler (if level allows), write directly
                # This part might not be strictly necessary if level is set to ERROR
                # but provides robustness if the handler level is changed later.
                if self.stream:
                     try:
                          msg = self.format(record)
                          self.stream.write(msg + self.terminator)
                          self.stream.flush()
                     except Exception:
                          self.handleError(record)
        finally:
            self.release()

    # Add terminator attribute for compatibility
    terminator = '\n'
# --- End Custom Handler ---


# Configure logging
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger() # Get root logger
logger.setLevel(logging.DEBUG) # Set root logger level to lowest level we want to capture (DEBUG or INFO)

# Console Handler (INFO and above)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

# General Log File Handler (INFO and above)
general_log_handler = logging.FileHandler('perturbation.log', mode='a') # Use append mode
general_log_handler.setFormatter(log_formatter)
general_log_handler.setLevel(logging.INFO)
logger.addHandler(general_log_handler)

# Error Log File Handler (ERROR and above) using Custom Handler
# Flush every 10 error messages
error_log_handler = BufferingErrorHandler('error.log', capacity=10, mode='a')
error_log_handler.setFormatter(log_formatter)
error_log_handler.setLevel(logging.ERROR) # Only process ERROR level and above
logger.addHandler(error_log_handler)

# Note: We get the root logger, so child loggers (like in perturber modules)
# using logging.getLogger(__name__) will inherit this configuration.

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
    graph_lines = []
    in_graph = False
    for line in amr_string.split('\n'):
        stripped_line = line.strip()
        # Skip empty lines and comment lines entirely
        if not stripped_line or stripped_line.startswith('#'):
            continue

        # Check if this line starts the graph or is part of it
        if stripped_line.startswith('('):
            in_graph = True

        # Only append lines once we are inside the graph structure
        if in_graph:
            graph_lines.append(line) # Append the original line to preserve indentation

    if not graph_lines:
         # Handle case where no graph lines were found at all
         logger.warning(f"No graph lines found in AMR string after cleaning:\n---\n{amr_string}\n---")
         return "" # Return empty string or raise error? Returning empty for now.

    return '\n'.join(graph_lines)


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
    logger.debug(f"Selected perturber function: {perturber_func}")

    try:
        # Try to apply the perturbation
        logger.debug(f"Applying {perturbation_type} perturbation")

        # Special handling for entity errors (intended for direct import case)
        if perturbation_type == "entity" and "EntityError" in globals():
            logger.debug("Entering special handling block for entity (direct import case)")
            # The lambda returns (graph, changelog), so unpack it
            # The lambda returns ((graph_or_none, inner_changelog), outer_changelog)
            raw_output = perturber_func(amr_graph)
            logger.debug(f"Raw output from entity perturber lambda (direct): {type(raw_output)} - {raw_output}")
            # Unpack the nested structure
            entity_result, outer_changelog = raw_output
            # entity_result is the tuple returned by EntityError: (graph_or_none, inner_changelog)
            perturbed_graph, inner_changelog = entity_result
            # Use the more detailed inner_changelog from EntityError
            changelog = inner_changelog
            # Ensure the perturber type is set correctly in the final changelog
            if isinstance(changelog, dict):
                changelog['perturber'] = 'entity' # Ensure perturber type is set
            else: # If inner_changelog wasn't a dict (e.g., None or error string), create one
                changelog = {'perturber': 'entity', 'details': changelog}

            logger.debug(f"After unpacking lambda result (direct): perturbed_graph type={type(perturbed_graph)}, final changelog type={type(changelog)}")
        else:
            # General case (including wrapper functions or other perturbation types)
            logger.debug(f"Calling general perturber function for {perturbation_type}")
            raw_output = perturber_func(amr_graph)
            logger.debug(f"Raw output from {perturbation_type} perturber (general): {type(raw_output)} - {raw_output}")
            # Assuming it returns a tuple (graph, changelog)
            perturbed_graph, changelog = raw_output
            logger.debug(f"After unpacking (general): perturbed_graph type={type(perturbed_graph)}, changelog type={type(changelog)}")

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

        logger.debug(f"Before returning from apply_perturbation: perturbed_graph type={type(perturbed_graph)}")
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
    stats: Dict[str, Dict[str, Union[int, Dict[str, int]]]],
    source_id: str # Add source_id parameter
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Generate a perturbed version of an AMR string using weighted random selection
    with random fallback on structural failures.

    Args:
        amr_string: The original AMR string
        perturbation_weights: Weights for each perturbation type
        stats: Dictionary to track perturbation statistics
        source_id: Identifier for the source item (for logging)

    Returns:
        Tuple of (perturbed_amr_string, changelog)
    """
    # --- Step 1: Parse the AMR ---
    amr_graph = None # Initialize amr_graph
    try:
        # Attempt 1: Clean and decode
        clean_amr = clean_amr_string(amr_string)
        logger.debug(f"Attempting to decode cleaned AMR for source_id '{source_id}':\n---\n{clean_amr}\n---")
        amr_graph = penman.decode(clean_amr)
        logger.debug(f"Successfully decoded cleaned AMR for source_id '{source_id}'")
    except Exception as e1:
        logger.warning(f"Failed to decode cleaned AMR for source_id '{source_id}': {e1}. Trying original string.")
        try:
            # Attempt 2: Decode original string
            logger.debug(f"Attempting to decode ORIGINAL AMR for source_id '{source_id}':\n---\n{amr_string}\n---")
            amr_graph = penman.decode(amr_string)
            logger.debug(f"Successfully decoded ORIGINAL AMR for source_id '{source_id}'")
        except Exception as e2:
            # Both attempts failed
            error_msg = f"Failed to parse AMR for source_id '{source_id}' (tried cleaned and original): {str(e2)}"
            logger.error(f"{error_msg}\nOriginal problematic string:\n---\n{amr_string}\n---")
            stats["parsing"]["failure"] += 1
            return None, {"error": error_msg}

    # Safeguard check after parsing attempts
    if amr_graph is None:
         error_msg = f"AMR graph is None after parsing attempts for source_id '{source_id}'"
         logger.error(error_msg)
         stats["parsing"]["failure"] += 1
         return None, {"error": error_msg}

    stats["parsing"]["success"] += 1

    # --- Step 2: Attempt Perturbations ---
    enabled_perturbations = [k for k, v in perturbation_weights.items() if v > 0]
    if not enabled_perturbations:
        error_msg = "No perturbation types enabled (all weights are zero or negative)"
        logger.error(error_msg)
        return None, {"error": error_msg}

    random.shuffle(enabled_perturbations)
    logger.debug(f"Available perturbations to try (shuffled): {enabled_perturbations}")

    for perturbation_type in enabled_perturbations:
        stats["selection"][perturbation_type] += 1
        logger.debug(f"Trying perturbation: {perturbation_type}")

        perturbed_graph, changelog = apply_perturbation(amr_graph, perturbation_type)

        # --- Handle Successful Perturbation ---
        if perturbed_graph is not None:
            stats["perturbation"][perturbation_type]["success"] += 1
            logger.debug(f"Perturbation {perturbation_type} succeeded.")

            # Ensure changelog format
            if isinstance(changelog, dict) and "perturber" not in changelog:
                 changelog["perturber"] = perturbation_type
            elif not isinstance(changelog, dict):
                 changelog = {"perturber": perturbation_type, "details": changelog}

            # Attempt encoding
            try:
                logger.debug(f"Attempting to encode perturbed_graph of type: {type(perturbed_graph)}")
                perturbed_amr_string = penman.encode(perturbed_graph)
                stats["encoding"]["success"] += 1
                return perturbed_amr_string, changelog # SUCCESS! Return result
            except Exception as e:
                error_msg = f"Error encoding perturbed graph after successful {perturbation_type} perturbation: {str(e)}"
                logger.error(error_msg)
                stats["encoding"]["failure"] += 1
                stats["perturbation"][perturbation_type]["failure"] += 1 # Count encoding failure as perturbation failure too
                stats["perturbation"][perturbation_type]["success"] -= 1
                # Encoding failed, continue loop to try next perturbation type
                continue

        # --- Handle Failed Perturbation ---
        else:
            stats["perturbation"][perturbation_type]["failure"] += 1
            logger.debug(f"Perturbation {perturbation_type} failed.")

            # Check for structural failure to trigger fallback
            is_structural_failure = (
                isinstance(changelog, dict) and
                (changelog.get('reason') == 'structural' or
                 changelog.get('error') == 'No discourse relations found')
            )

            if is_structural_failure:
                potential_fallbacks = [
                    ptype for ptype, weight in perturbation_weights.items()
                    if weight > 0 and ptype != perturbation_type
                ]

                if potential_fallbacks:
                    chosen_fallback_type = random.choice(potential_fallbacks)
                    logger.info(f"Structural failure for '{perturbation_type}'. Attempting random fallback: '{chosen_fallback_type}'")
                    stats["fallback"][perturbation_type] += 1

                    # Attempt fallback perturbation
                    fallback_graph, fallback_changelog = apply_perturbation(amr_graph, chosen_fallback_type)

                    if fallback_graph is not None:
                        # Fallback succeeded
                        stats["perturbation"][chosen_fallback_type]["success"] += 1
                        logger.debug(f"Fallback perturbation '{chosen_fallback_type}' succeeded.")

                        # Ensure fallback changelog format
                        if isinstance(fallback_changelog, dict) and "perturber" not in fallback_changelog:
                            fallback_changelog["perturber"] = chosen_fallback_type
                        elif not isinstance(fallback_changelog, dict):
                            fallback_changelog = {"perturber": chosen_fallback_type, "details": fallback_changelog}

                        # Attempt encoding fallback result
                        try:
                            perturbed_amr_string = penman.encode(fallback_graph)
                            stats["encoding"]["success"] += 1
                            return perturbed_amr_string, fallback_changelog # SUCCESS! Return fallback result
                        except Exception as e:
                            error_msg = f"Error encoding perturbed graph after successful FALLBACK {chosen_fallback_type} perturbation: {str(e)}"
                            logger.error(error_msg)
                            stats["encoding"]["failure"] += 1
                            stats["perturbation"][chosen_fallback_type]["failure"] += 1
                            stats["perturbation"][chosen_fallback_type]["success"] -= 1
                            # Fallback encoding failed, continue main loop
                    else:
                        # Fallback attempt also failed
                        stats["perturbation"][chosen_fallback_type]["failure"] += 1
                        logger.warning(f"Fallback perturbation '{chosen_fallback_type}' also failed.")
                        # Fallback failed, continue main loop
                else:
                    # No other types available for fallback
                    logger.debug(f"Structural failure for '{perturbation_type}', but no other types enabled for fallback. Trying next.")
                    # No fallbacks possible, continue main loop
            else:
                # Not a structural failure
                logger.debug(f"Perturbation {perturbation_type} failed (non-structural). Trying next.")
                # Non-structural failure, continue main loop

            # If we reach here, it means either the original perturbation failed non-structurally,
            # or it failed structurally and the fallback attempt (if any) also failed or its encoding failed.
            # In all these cases, we continue the main loop to try the next original perturbation type.
            continue

    # If loop finishes without success
    logger.warning(f"All enabled perturbations (and fallbacks) failed for source_id '{source_id}'.")
    final_changelog = {"error": "All enabled perturbations failed"}
    return None, final_changelog


def generate_dataset(
    input_file: str,
    output_file: str,
    perturbation_weights: Dict[str, float],
    perturbed_per_original: int = 1,
    amr_field: str = "summary_amr",
    seed: Optional[int] = None,
    max_examples: Optional[int] = None,
    debug_sample: Optional[int] = None,
    save_interval: int = 10000  # Add parameter for interval
) -> None:
    """
    Generate a labeled dataset for machine learning with periodic saving.

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

    # Prepare output dataset and check for existing data to resume
    output_data = []
    processed_source_ids = set()
    last_save_count = 0

    if os.path.exists(output_file):
        logger.info(f"Output file {output_file} exists. Attempting to load and resume.")
        try:
            with open(output_file, 'r', encoding='utf-8') as f_existing:
                output_data = json.load(f_existing)
            for example in output_data:
                if "source_id" in example:
                    processed_source_ids.add(example["source_id"])
            last_save_count = len(output_data)
            logger.info(f"Successfully loaded {len(output_data)} existing examples. Found {len(processed_source_ids)} unique processed source IDs.")
        except json.JSONDecodeError:
            logger.warning(f"Could not parse existing output file {output_file}. Starting fresh.")
            output_data = []
            processed_source_ids = set()
            last_save_count = 0
        except Exception as e:
            logger.error(f"Error loading existing output file {output_file}: {e}. Starting fresh.")
            output_data = []
            processed_source_ids = set()
            last_save_count = 0
    else:
        logger.info("No existing output file found. Starting fresh.")


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
        "total_duplicates": 0, # Add counter for duplicates
    }

    logger.info(f"Starting to process {len(data)} examples with {perturbed_per_original} perturbations each")
    logger.info(f"Perturbation weights: {perturbation_weights}")

    # Process each item
    processed_count = 0
    skipped_count = 0
    for i, item in enumerate(tqdm(data, desc="Generating dataset")):
        source_id = item.get("id") # Try to get the ID
        if not source_id: # If ID is null, None, empty, or key missing
            source_id = f"item_{i}" # Use index to guarantee uniqueness

        # Check if this source ID has already been processed
        if source_id in processed_source_ids:
            skipped_count += 1
            continue # Skip this item

        # Skip if AMR field is missing
        if amr_field not in item or not item[amr_field]:
            logger.warning(f"Skipping item {source_id} due to missing or empty AMR field '{amr_field}'.")
            continue

        amr_string = item[amr_field]

        # Attempt to parse original AMR to extract metadata (e.g., snt)
        target_summary_from_snt = "" # Default value
        try:
            # Use the original amr_string, not the cleaned one for metadata
            original_graph = penman.decode(amr_string)
            if 'snt' in original_graph.metadata:
                target_summary_from_snt = original_graph.metadata['snt']
            else:
                 logger.debug(f"No 'snt' metadata found for source_id '{source_id}'")
        except Exception as parse_error:
            logger.warning(f"Could not parse original AMR for metadata extraction (source_id '{source_id}'): {parse_error}")
            # Keep default empty string for target_summary

        # Add the original AMR example (labeled as correct = 1)
        original_example = {
            "id": f"{i}_original",
            "amr": amr_string, # Store the original, potentially uncleaned AMR string
            "score": 1.0,  # Original AMR is correct
            "perturbation_type": None,
            "source_id": source_id, # Use consistent source_id
            "source_text": item.get("source_text", ""),  # Include source text from input item
            "title": item.get("title", ""),              # Include title from input item
            "target_summary": target_summary_from_snt  # Use extracted snt metadata
        }
        output_data.append(original_example)

        # Check for intermediate save after adding original
        if len(output_data) // save_interval > last_save_count // save_interval:
             logger.info(f"Saving intermediate dataset with {len(output_data)} examples...")
             # Save without shuffling for intermediate saves
             with open(output_file, 'w', encoding='utf-8') as f_intermediate:
                 json.dump(output_data, f_intermediate, indent=2, ensure_ascii=False)
             logger.info(f"Intermediate dataset saved to {output_file}")
             last_save_count = len(output_data)

        # Debug first few examples in detail if requested
        if debug_sample is not None and i < debug_sample:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        # Generate perturbed versions
        successful_perturbations = 0
        generated_perturbations_for_item = set() # Track unique perturbations for this item
        max_attempts = perturbed_per_original * 5  # Increase attempts slightly more to account for potential duplicates
        attempts = 0

        while successful_perturbations < perturbed_per_original and attempts < max_attempts:
            attempts += 1
            stats["total_attempts"] += 1

            # Call generate_perturbed_amr, passing source_id
            perturbed_amr, changelog = generate_perturbed_amr(amr_string, perturbation_weights, stats, source_id=source_id)

            # If perturbation failed, skip this example
            if perturbed_amr is None:
                if isinstance(changelog, dict) and "error" in changelog:
                    logger.debug(f"Perturbation attempt {attempts} failed: {changelog['error']}")
                stats["total_failures"] += 1
                continue

            # Check if this perturbation is a duplicate for the current item
            if perturbed_amr in generated_perturbations_for_item:
                logger.debug(f"Duplicate perturbation generated for source_id {source_id}. Skipping.")
                stats["total_duplicates"] += 1
                # Don't increment successful_perturbations, just continue to next attempt
                continue

            # If not a duplicate, add it and count it
            generated_perturbations_for_item.add(perturbed_amr)
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
                "source_id": source_id, # Use consistent source_id
                "changelog": changelog,
                "source_text": item.get("source_text", ""),  # Include source text
                "title": item.get("title", ""),              # Include title
                "target_summary": item.get("target_summary", "")  # Include target summary
            }
            output_data.append(perturbed_example)

            # Check for intermediate save after adding perturbed
            if len(output_data) // save_interval > last_save_count // save_interval:
                logger.info(f"Saving intermediate dataset with {len(output_data)} examples...")
                # Save without shuffling for intermediate saves
                with open(output_file, 'w', encoding='utf-8') as f_intermediate:
                    json.dump(output_data, f_intermediate, indent=2, ensure_ascii=False)
                logger.info(f"Intermediate dataset saved to {output_file}")
                last_save_count = len(output_data)

            if debug_sample is not None and i < debug_sample:
                logger.debug(f"Successfully created perturbation {successful_perturbations}/{perturbed_per_original}")
        processed_count += 1
        processed_source_ids.add(source_id) # Mark as processed

        if successful_perturbations < perturbed_per_original:
            logger.warning(f"Could only generate {successful_perturbations}/{perturbed_per_original} perturbations for source_id {source_id}")

    logger.info(f"Finished processing. Processed {processed_count} new items, skipped {skipped_count} already existing items.")

    # Save the final output dataset
    logger.info("Shuffling final dataset...")
    random.shuffle(output_data)  # Shuffle to avoid bias in training
    logger.info(f"Saving final dataset with {len(output_data)} examples to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Generated dataset with {len(output_data)} examples saved to {output_file}")

    # --- Save Statistics ---
    stats_output_file = "output_statistic.json"
    logger.info(f"Saving final statistics to {stats_output_file}")
    try:
        # Add summary counts to the stats dictionary before saving
        stats["summary"] = {
            "total_examples_generated": len(output_data),
            "original_amrs": sum(1 for ex in output_data if ex["score"] == 1.0),
            "perturbed_amrs": sum(1 for ex in output_data if ex["score"] == 0.0)
        }
        # Add perturbation counts
        perturbation_counts = {}
        for ex in output_data:
            if ex["score"] == 0.0:
                pert_type = ex.get("perturbation_type", "unknown")
                perturbation_counts[pert_type] = perturbation_counts.get(pert_type, 0) + 1
        stats["summary"]["perturbed_counts_by_type"] = perturbation_counts

        with open(stats_output_file, 'w', encoding='utf-8') as f_stats:
            json.dump(stats, f_stats, indent=2, ensure_ascii=False)
        print(f"Statistics saved to {stats_output_file}")
    except Exception as e:
        logger.error(f"Failed to save statistics to {stats_output_file}: {e}")
        print(f"Error: Failed to save statistics to {stats_output_file}")

    # Optional: Keep printing a brief summary to console
    print("\nBrief Summary:")
    print(f"  Total Examples: {stats['summary']['total_examples_generated']}")
    print(f"  Original AMRs: {stats['summary']['original_amrs']}")
    print(f"  Perturbed AMRs: {stats['summary']['perturbed_amrs']}")
    if stats['summary']['perturbed_amrs'] > 0:
         print("  Perturbed Breakdown:")
         for pert_type, count in sorted(stats["summary"]["perturbed_counts_by_type"].items()):
              print(f"    {pert_type}: {count}")


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
    parser.add_argument("--perturbed-per-original", "-n", type=int, default=5,
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
    parser.add_argument("--save-interval", type=int, default=10000,
                        help="Save intermediate dataset every N examples")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        # If not verbose, ensure console doesn't show DEBUG
        console_handler.setLevel(logging.INFO)
        # Keep general log at INFO, error log at ERROR regardless of verbosity
        # Root logger remains DEBUG to allow handlers to filter

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
            debug_sample=args.debug_sample,
            save_interval=args.save_interval # Pass the interval
        )
    else:
        # Generate a full dataset first
        # Ensure temp file is in a writable location, e.g., same dir as output_dir
        temp_output_dir = os.path.dirname(args.output_dir) if args.output_dir else '.'
        temp_output = os.path.join(temp_output_dir, "temp_full_dataset.json")

        generate_dataset(
            args.input,
            temp_output,
            perturbation_weights,
            perturbed_per_original=args.perturbed_per_original,
            amr_field=args.amr_field,
            seed=args.seed,
            max_examples=args.max_examples,
            debug_sample=args.debug_sample,
            save_interval=args.save_interval # Pass the interval
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
