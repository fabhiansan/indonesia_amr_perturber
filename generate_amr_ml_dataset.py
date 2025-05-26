# -*- coding: utf-8 -*-
import json
import random
import os
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import penman
from tqdm.notebook import tqdm # Use tqdm.notebook for Jupyter
import numpy as np

# --- Perturbation Module Imports ---
# Configure logging initially (can be reconfigured by the main function)
logging.basicConfig(
    level=logging.WARNING, # Default to WARNING, can be overridden
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()] # Log to console/notebook output by default
)
logger = logging.getLogger(__name__)

perturbation_modules_loaded = False
perturbation_functions = {}

try:
    # First try direct imports from individual modules
    from data_perturber.predicates_perturber import insertWrongPredicates
    from data_perturber.circumstance_perturber import insertCircumstanceError
    from data_perturber.entity_perturber import insertEntityError
    from data_perturber.discourse_perturber import insertDiscourseError
    from data_perturber.out_of_article_perturber import insertOutOfArticleError

    # Map using direct functions
    perturbation_functions = {
        "predicate": lambda g: insertWrongPredicates(g),
        "circumstance": lambda g: insertCircumstanceError(g, "both"),
        # EntityError might be a class or function, adjust lambda if needed
        # Assuming EntityError(g) returns the graph, and changelog needs construction
        "entity": insertEntityError,
        "discourse": insertDiscourseError,
        "out_of_article": insertOutOfArticleError
    }
    perturbation_modules_loaded = True
    logger.info("Loaded perturbation modules from individual files")

except ImportError as e_individual:
    logger.warning(f"Could not import from individual modules: {e_individual}")
    try:
        # Try importing through the insertion wrapper module
        from data_perturber.insertion import (
            predicate_error_insertion,
            circumstance_error_insertion,
            entity_error_insertion,
            discourse_error_insertion,
            out_of_article_error_insertion
        )
        # Map using wrapper functions
        perturbation_functions = {
            "predicate": predicate_error_insertion,
            "circumstance": circumstance_error_insertion,
            "entity": entity_error_insertion,
            "discourse": discourse_error_insertion,
            "out_of_article": out_of_article_error_insertion
        }
        perturbation_modules_loaded = True
        logger.info("Loaded perturbation modules from data_perturber.insertion")
    except ImportError as e_wrapper:
        logger.error(f"Failed to import perturbation modules from both individual files and wrapper: {e_wrapper}")
        # Raise error only if needed when the function is called
        # raise ImportError("Could not import perturbation modules. Please check your installation.")

# --- Helper Functions (Identical to the original script) ---

def clean_amr_string(amr_string: str) -> str:
    """
    Clean AMR string by removing comments and metadata.
    """
    clean_lines = []
    for line in amr_string.split('\n'):
        if not line.strip() or line.strip().startswith('#'):
            continue
        clean_lines.append(line)
    return '\n'.join(clean_lines)

def apply_perturbation(
    amr_graph: penman.Graph,
    perturbation_type: str,
    available_perturbation_funcs: Dict[str, Any]
) -> Tuple[Optional[penman.Graph], Dict[str, Any]]:
    """
    Apply a specific type of perturbation to an AMR graph.
    Uses the globally loaded perturbation functions.
    """
    if not perturbation_modules_loaded:
         raise ImportError("Perturbation modules could not be loaded.")

    if perturbation_type not in available_perturbation_funcs:
        raise ValueError(f"Unknown or unavailable perturbation type: {perturbation_type}")

    perturber_func = available_perturbation_funcs[perturbation_type]

    try:
        logger.debug(f"Applying {perturbation_type} perturbation")

        # Execute the perturbation function. Assume it returns (graph, changelog)
        # The original script had special handling for EntityError if imported directly.
        # We assume the unified interface returns (graph, changelog) for simplicity here.
        # If EntityError has a different signature, this needs adjustment.
        result = perturber_func(amr_graph)

        # Check if the result is the expected tuple
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError(f"Perturber '{perturbation_type}' did not return the expected (graph, changelog) tuple. Got: {type(result)}")
        else:
            perturbed_graph, changelog = result

        # Ensure changelog is a dict
        if not isinstance(changelog, dict):
            logger.warning(f"Changelog for {perturbation_type} was not a dict ({type(changelog)}). Converting.")
            changelog = {"perturber": perturbation_type, "original_changelog": changelog}

        # Check for explicit errors reported in changelog
        if "error" in changelog:
            error_msg = f"Error explicitly reported by {perturbation_type} perturber: {changelog['error']}"
            logger.warning(error_msg)
            # Return None for the graph, but keep the changelog with the error
            return None, {**changelog, "perturber": perturbation_type} # Ensure perturber type is in error dict

        # Check for specific 'no_change' action from entity perturber
        if perturbation_type == "entity" and changelog.get("action") == "no_change" and changelog.get("description") == "No suitable entities found for swapping":
            error_msg = f"Entity perturber reported no suitable entities for swapping."
            logger.warning(error_msg)
            # Treat this as a failure, return None for the graph
            return None, {**changelog, "error": error_msg, "perturber": perturbation_type}

        # Ensure perturber type is in the successful changelog
        if "perturber" not in changelog:
            changelog["perturber"] = perturbation_type

        return perturbed_graph, changelog

    except Exception as e:
        error_msg = f"Exception during '{perturbation_type}' perturbation: {str(e)}"
        logger.warning(error_msg, exc_info=True) # Log traceback for debugging
        changelog = {
            "error": error_msg,
            "perturber": perturbation_type
        }
        return None, changelog


def generate_perturbed_amr(
    amr_string: str,
    perturbation_weights: Dict[str, float],
    stats: Dict[str, Dict[str, Union[int, Dict]]],
    available_perturbation_funcs: Dict[str, Any]
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Generate a perturbed version of an AMR string using weighted random selection.
    """
    try:
        clean_amr = clean_amr_string(amr_string)
        if not clean_amr: # Handle empty AMR after cleaning
             raise ValueError("AMR string is empty after cleaning comments.")
        amr_graph = penman.decode(clean_amr)
    except Exception as e:
        try:
            # Try original if cleaning removed essential parts (less likely but possible)
            if not amr_string.strip():
                 raise ValueError("Original AMR string is empty or whitespace.")
            amr_graph = penman.decode(amr_string)
            logger.warning(f"Could not parse cleaned AMR, but parsed original. Error: {e}")
        except Exception as e2:
            error_msg = f"Failed to parse AMR (original and cleaned): {str(e2)}"
            logger.error(error_msg)
            stats["parsing"]["failure"] += 1
            return None, {"error": error_msg, "stage": "parsing"}

    stats["parsing"]["success"] += 1

    # Normalize weights
    total_weight = sum(w for w in perturbation_weights.values() if w > 0)
    if total_weight <= 0:
        error_msg = "No perturbation types have positive weight."
        logger.error(error_msg)
        return None, {"error": error_msg, "stage": "weighting"}

    normalized_weights = {k: v / total_weight for k, v in perturbation_weights.items() if v > 0}

    # Filter available functions based on positive weights
    runnable_perturbations = {
        k: func for k, func in available_perturbation_funcs.items()
        if k in normalized_weights
    }
    runnable_types = list(runnable_perturbations.keys())
    runnable_weights = [normalized_weights[k] for k in runnable_types]

    if not runnable_types:
         error_msg = "No perturbation functions available for the types with positive weights."
         logger.error(error_msg)
         return None, {"error": error_msg, "stage": "selection"}


    # Select a perturbation type randomly based on weights
    perturbation_type = random.choices(runnable_types, weights=runnable_weights, k=1)[0]

    logger.debug(f"Selected perturbation type: {perturbation_type}")
    stats["selection"][perturbation_type] += 1

    # Apply the selected perturbation
    perturbed_graph, changelog = apply_perturbation(amr_graph, perturbation_type, runnable_perturbations)

    # --- Fallback Logic ---
    if perturbed_graph is None:
        stats["perturbation"][perturbation_type]["failure"] += 1
        logger.debug(f"Primary perturbation {perturbation_type} failed, trying alternatives")

        # Create a list of alternatives, sorted by weight (descending)
        alternatives = sorted(
            [(ptype, normalized_weights[ptype]) for ptype in runnable_types if ptype != perturbation_type],
            key=lambda item: item[1],
            reverse=True
        )

        for retry_type, _ in alternatives:
            logger.debug(f"Trying alternative perturbation: {retry_type}")
            # Ensure we pass the *original* graph for retry
            perturbed_graph, retry_changelog = apply_perturbation(amr_graph, retry_type, runnable_perturbations)

            if perturbed_graph is not None:
                stats["perturbation"][retry_type]["success"] += 1
                stats["fallback"][retry_type] += 1
                # Use the changelog from the successful retry
                changelog = retry_changelog
                # Mark that fallback occurred and which type succeeded
                changelog["fallback_used"] = True
                changelog["original_attempt"] = perturbation_type
                # Ensure the correct perturber type is set
                changelog["perturber"] = retry_type
                logger.debug(f"Alternative perturbation {retry_type} succeeded")
                break # Exit loop once a fallback works
            else:
                # Log failure of the fallback attempt
                stats["perturbation"][retry_type]["failure"] += 1
                logger.debug(f"Alternative perturbation {retry_type} also failed.")
                # Keep the changelog from the *first* failure if all fallbacks fail
                # The 'changelog' variable still holds the error from the primary attempt
    else:
        # Initial attempt succeeded
        stats["perturbation"][perturbation_type]["success"] += 1
        # Ensure perturber field is present, even on success
        if isinstance(changelog, dict) and "perturber" not in changelog:
            changelog["perturber"] = perturbation_type

    # --- Encoding ---
    if perturbed_graph is not None:
        try:
            perturbed_amr_string = penman.encode(perturbed_graph)
            stats["encoding"]["success"] += 1
            return perturbed_amr_string, changelog
        except Exception as e:
            error_msg = f"Error encoding perturbed graph generated by '{changelog.get('perturber', 'unknown')}': {str(e)}"
            logger.error(error_msg)
            stats["encoding"]["failure"] += 1
            # Return failure, but include the changelog which might indicate the successful perturbation type before encoding failed
            return None, {**changelog, "error": error_msg, "stage": "encoding"}

    # If we reach here, all attempts (primary and fallback) failed to produce a valid graph
    stats["total_failures"] += 1 # Increment failure count here
    logger.debug(f"All perturbation attempts failed for this AMR.")
    # Return the changelog from the *last* failed attempt (which might be the primary or a fallback)
    # If primary failed and all fallbacks failed, changelog contains the primary error.
    # If primary failed, a fallback was tried and failed, changelog contains that fallback's error.
    # Ensure the final changelog reflects the failure stage if not already set
    if "stage" not in changelog:
        changelog["stage"] = "perturbation_fallback"
    return None, changelog


# --- Main Dataset Generation Function (modified from original script's generate_dataset) ---

def _generate_dataset_internal(
    input_data: List[Dict],
    perturbation_weights: Dict[str, float],
    perturbed_per_original: int = 1,
    amr_field: str = "summary_amr",
    seed: Optional[int] = None,
    debug_sample: Optional[int] = None,
    available_perturbation_funcs: Dict[str, Any] = perturbation_functions # Use loaded functions
) -> Tuple[List[Dict], Dict]:
    """
    Internal logic to generate dataset examples (original + perturbed).
    Returns the list of examples and the statistics dictionary.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    output_data = []
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
        "total_successful_perturbations": 0,
        "total_failed_perturbations": 0, # Renamed for clarity
        "input_examples_processed": 0,
        "input_examples_skipped": 0,
        "original_examples_added": 0,
        "perturbed_examples_added": 0,
    }

    if not available_perturbation_funcs:
        logger.error("No perturbation functions are available. Cannot generate perturbed data.")
        # Still return originals if possible
    elif not perturbation_modules_loaded:
         logger.error("Perturbation modules failed to load. Cannot generate perturbed data.")
         # Still return originals

    logger.info(f"Processing {len(input_data)} input items...")
    logger.info(f"Perturbation weights: {perturbation_weights}")
    logger.info(f"Attempting to generate {perturbed_per_original} perturbed examples per original.")

    # Determine which perturbation types have non-zero weight
    active_perturbation_types = {k for k, v in perturbation_weights.items() if v > 0}
    can_perturb = perturbation_modules_loaded and bool(active_perturbation_types) and bool(available_perturbation_funcs)

    for i, item in enumerate(tqdm(input_data, desc="Generating examples")):
        stats["input_examples_processed"] += 1

        if amr_field not in item or not item[amr_field] or not isinstance(item[amr_field], str):
            # logger.warning(f"Skipping item {i} (ID: {item.get('id', 'N/A')}) due to missing or invalid AMR field '{amr_field}'. Value: {item.get(amr_field)}")
            stats["input_examples_skipped"] += 1
            continue

        amr_string = item[amr_field]
        source_id = item.get("id", f"item_{i}") # Ensure unique ID

        # Add the original AMR example (labeled as correct = 1)
        original_example = {
            "id": f"{source_id}_original",
            "amr": amr_string,
            "score": 1.0,
            "perturbation_type": None,
            "source_id": source_id,
            "changelog": None,
            # Optionally copy other fields from the input item
            "source_text": item.get("source_text"),
            "title": item.get("title"),
            "target_summary": item.get("target_summary"),
            "summary_amr": item.get("summary_amr")
        }
        output_data.append(original_example)
        stats["original_examples_added"] += 1

        # --- Generate Perturbed Versions ---
        if not can_perturb or perturbed_per_original <= 0:
            continue # Skip perturbation generation if disabled or modules failed

        successful_perturbations = 0
        # Allow more attempts than requested to increase chance of success
        # Especially important if some perturbations fail often for certain graphs
        max_attempts_per_original = perturbed_per_original * 5
        attempts = 0

        # Configure logger level for detailed debug samples
        initial_log_level = logger.level
        if debug_sample is not None and i < debug_sample:
            logger.setLevel(logging.DEBUG)
            logger.debug(f"\n--- Debugging Item {i} (Source ID: {source_id}) ---")
            logger.debug(f"Original AMR:\n{amr_string[:500]}{'...' if len(amr_string) > 500 else ''}")
        else:
             # Ensure level is reset if previously set to DEBUG for a prior sample
             if logger.level == logging.DEBUG:
                 logger.setLevel(initial_log_level if initial_log_level is not None else logging.INFO)


        while successful_perturbations < perturbed_per_original and attempts < max_attempts_per_original:
            attempts += 1
            stats["total_attempts"] += 1
            logger.debug(f"Attempt {attempts}/{max_attempts_per_original} for item {i}, successful so far: {successful_perturbations}")

            perturbed_amr_str, changelog = generate_perturbed_amr(
                amr_string,
                perturbation_weights,
                stats,
                available_perturbation_funcs
            )

            if perturbed_amr_str is not None:
                successful_perturbations += 1
                stats["total_successful_perturbations"] += 1 # Use the specific counter
                stats["perturbed_examples_added"] += 1

                # Ensure changelog is a dictionary (should be handled by generate_perturbed_amr now)
                if not isinstance(changelog, dict):
                     logger.error(f"Changelog received from generate_perturbed_amr was not a dict: {type(changelog)}. Fixing.")
                     changelog = {"error": "Invalid changelog type received", "original_changelog": changelog}

                perturbation_type = changelog.get("perturber", "unknown")

                perturbed_example = {
                    "id": f"{source_id}_perturbed_{successful_perturbations}",
                    "amr": perturbed_amr_str,
                    "score": 0.0,
                    "perturbation_type": perturbation_type,
                    "source_id": source_id,
                    "changelog": changelog,
                     # Optionally copy other fields
                    "source_text": item.get("source_text"),
                    "title": item.get("title"),
                    "target_summary": item.get("target_summary"),
                    "summary_amr": item.get("summary_amr")
                }
                output_data.append(perturbed_example)
                logger.debug(f"Successfully created perturbation {successful_perturbations}/{perturbed_per_original} via {perturbation_type}")

            else:
                # Failure case handled within generate_perturbed_amr stats, log here
                failure_reason = changelog.get("error", "Unknown error")
                failed_pert_type = changelog.get("perturber", "unknown")
                failure_stage = changelog.get("stage", "unknown")
                logger.debug(f"Perturbation attempt {attempts} failed for item {i}. Type attempted/failed: {failed_pert_type}. Stage: {failure_stage}. Reason: {failure_reason[:200]}...")
                # Note: total_failed_perturbations is incremented inside generate_perturbed_amr if all attempts fail

        # Reset logger level if it was changed for debugging
        if logger.level != initial_log_level:
             logger.setLevel(initial_log_level if initial_log_level is not None else logging.INFO)

        if successful_perturbations < perturbed_per_original:
            logger.warning(f"Could only generate {successful_perturbations}/{perturbed_per_original} valid perturbations for item {i} (Source ID: {source_id}) after {attempts} attempts.")

    # Add final counts to stats
    stats["final_total_examples"] = len(output_data)
    # Calculate overall failure count (total attempts - successful perturbations)
    stats["total_failed_perturbations"] = stats["total_attempts"] - stats["total_successful_perturbations"]


    logger.info(f"Finished processing. Generated {stats['original_examples_added']} original and {stats['perturbed_examples_added']} perturbed examples.")
    logger.info(f"Total examples in dataset: {len(output_data)}")

    return output_data, stats


# --- Main Callable Function ---

def create_amr_dataset(
    # Input/Output
    input_file: str,
    output_file: Optional[str] = None, # Make output file optional
    output_dir: Optional[str] = None, # For splits
    # Data structure
    amr_field: str = "summary_amr",
    # Perturbation Control
    perturbation_weights: Optional[Dict[str, float]] = None,
    perturbed_per_original: int = 5,
    # Processing Control
    seed: Optional[int] = None,
    max_examples: Optional[int] = None,
    split: bool = False, # Whether to split into train/dev/test
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    # Saving/Debugging
    save_interval: Optional[int] = 10000, # Interval for intermediate saves if output_file is given
    debug_sample: Optional[int] = 0,
    verbose: bool = False,
    log_file: Optional[str] = 'perturbation_notebook.log'
) -> Union[List[Dict], Dict[str, List[Dict]], Tuple[Union[List[Dict], Dict[str, List[Dict]]], Dict]]:
    """
    Generates a labeled dataset for machine learning from AMR data.

    Args:
        input_file: Path to the input JSON file containing AMR data.
        output_file: Optional. Path to the output JSON file. If None, data is not saved
                     to a single file (but splits might be saved if split=True).
                     If split=True, this path is ignored (use output_dir).
        output_dir: Optional. Directory to save train/dev/test JSON files.
                    Required if split=True and you want to save the splits.
        amr_field: Field name in the input JSON containing the AMR string.
        perturbation_weights: Dictionary mapping perturbation type (str) to its weight (float).
                              Defaults to equal weights if None. Keys should match available perturbers.
                              e.g., {"predicate": 0.2, "circumstance": 0.3, ...}
        perturbed_per_original: Number of perturbed examples to generate per original AMR.
        seed: Optional random seed for reproducibility.
        max_examples: Optional maximum number of input examples to process.
        split: If True, split the generated data into train, dev, and test sets.
        split_ratios: Tuple representing the train, dev, test split ratios. Must sum to 1.0.
        save_interval: Optional. If output_file is specified, save intermediate results
                       every N *total* examples generated. Set to None to disable.
        debug_sample: Number of initial examples to log in detail (DEBUG level).
        verbose: If True, set logging level to INFO (or DEBUG if debug_sample > 0).
        log_file: Optional path to a file for logging messages.

    Returns:
        If split is False:
            A tuple containing:
            - list: The full generated dataset (list of dictionaries).
            - dict: Statistics about the generation process.
        If split is True:
            A tuple containing:
            - dict: A dictionary containing the splits: {'train': [...], 'dev': [...], 'test': [...]}
            - dict: Statistics about the generation process.
        Returns data even if file saving fails or is disabled.
    """
    # --- Configure Logging ---
    log_level = logging.WARNING
    if verbose:
        log_level = logging.INFO
    if debug_sample is not None and debug_sample > 0:
        # Debug implies verbose for the initial samples
        log_level = logging.INFO # Keep base INFO, debug controlled internally

    # Clear previous handlers to avoid duplicate logs in notebooks if run multiple times
    root_logger = logging.getLogger()
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
        h.close() # Close file handlers if any

    handlers = [logging.StreamHandler()] # Always log to notebook output
    if log_file:
        # Ensure directory exists for log file
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode='a')) # Append mode

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
        handlers=handlers
    )
    logger.info("Logging configured.")

    # --- Validate Inputs ---
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    if split and output_file:
        logger.warning("Both 'split=True' and 'output_file' provided. 'output_file' will be ignored. Use 'output_dir' for saving splits.")
        output_file = None # Ignore single output file when splitting
    if split and not output_dir:
        logger.warning("split=True but output_dir is not specified. Split data will be returned but not saved to files.")
    if not split and output_dir:
        logger.warning("output_dir provided but split=False. output_dir will be ignored. Use output_file to save the full dataset.")
        output_dir = None # Ignore output dir when not splitting
    if sum(split_ratios) != 1.0:
         raise ValueError(f"split_ratios must sum to 1.0. Got: {split_ratios} (sum={sum(split_ratios)})")
    if save_interval is not None and save_interval <= 0:
         logger.warning(f"Invalid save_interval ({save_interval}). Disabling intermediate saving.")
         save_interval = None


    # --- Load Perturbation Functions ---
    # (Assumes perturbation_functions is populated globally above)
    if not perturbation_modules_loaded or not perturbation_functions:
         logger.error("Perturbation modules are not loaded correctly. Perturbed examples cannot be generated.")
         # Continue to generate originals, but perturbed_per_original will be effectively 0

    # --- Set Default Weights if None ---
    if perturbation_weights is None:
        available_types = list(perturbation_functions.keys())
        if available_types:
            equal_weight = 1.0 / len(available_types)
            perturbation_weights = {ptype: equal_weight for ptype in available_types}
            logger.info(f"No perturbation weights provided. Using equal weights: {perturbation_weights}")
        else:
             perturbation_weights = {}
             logger.warning("No perturbation functions available, cannot set default weights.")

    # --- Load Input Data ---
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        input_data = []
        if isinstance(raw_data, list):
            # If raw_data is a list, assume it's already a list of records
            logger.info(f"Input file is a list. Assuming it's a list of records.")
            input_data = raw_data
            # Add 'id' if not present (optional, but good for consistency)
            for i, record in enumerate(input_data):
                if 'id' not in record:
                    record['id'] = f"item_{i}"

        elif isinstance(raw_data, dict):
            # If raw_data is a dictionary, assume it's the dictionary-of-dictionaries structure
            logger.info(f"Input file is a dictionary. Assuming dictionary-of-dictionaries format.")
            if raw_data:
                # Assuming all fields have the same keys (indices)
                # Get the keys (indices) from the first field found
                # Add a check to ensure the first value is a dictionary
                first_value = next(iter(raw_data.values()), None)
                if first_value is None or not isinstance(first_value, dict):
                     raise ValueError("Input dictionary is empty or values are not dictionaries.")

                first_field_keys = list(first_value.keys())

                for index in first_field_keys:
                    record = {"id": index} # Add id based on the index
                    for field_name, field_data in raw_data.items():
                        if index in field_data:
                            record[field_name] = field_data[index]
                        else:
                            record[field_name] = None
                    input_data.append(record)
        else:
            raise TypeError(f"Input file contains unexpected top-level data type: {type(raw_data)}. Expected list or dictionary.")


        logger.info(f"Loaded and transformed {len(input_data)} items from {input_file}")
    except Exception as e:
        logger.exception(f"Failed to load or parse input file: {input_file}")
        raise # Re-raise after logging

    if max_examples is not None and max_examples > 0:
        logger.info(f"Processing a maximum of {max_examples} examples.")
    # --- Generate Dataset (Internal Call) ---
    # Note: Intermediate saving logic needs to be handled carefully outside the internal function
    # if we want it based on total examples across originals and perturbed.
    # For simplicity here, we generate all data first, then save.
    # TODO: Implement intermediate saving properly if needed for very large datasets.

    full_dataset, stats = _generate_dataset_internal(
        input_data=input_data,
        perturbation_weights=perturbation_weights,
        perturbed_per_original=perturbed_per_original,
        amr_field=amr_field,
        seed=seed,
        debug_sample=debug_sample,
        available_perturbation_funcs=perturbation_functions
    )

    # Resume capability for non-split JSONL
    if not split and output_file and os.path.exists(output_file):
        existing_data = []
        with open(output_file, 'r', encoding='utf-8') as f_resume:
            for line in f_resume:
                try:
                    existing_data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        processed_ids = {ex.get("source_id") for ex in existing_data}
        new_data = [ex for ex in full_dataset if ex.get("source_id") not in processed_ids]
        logger.info(f"Resuming: {len(existing_data)} existing examples, adding {len(new_data)} new examples.")
        combined_dataset = existing_data + new_data
    else:
        combined_dataset = full_dataset

    # --- Shuffle Data ---
    if seed is not None:
        random.seed(seed) # Ensure shuffle is also reproducible
    random.shuffle(combined_dataset)
    logger.info("Generated dataset shuffled.")

    # Intermediate saving for JSONL at defined intervals
    if output_file and save_interval:
        total = len(combined_dataset)
        # At least one intermediate save
        first_save = min(save_interval, total)
        try:
            with open(output_file, 'w', encoding='utf-8') as f_tmp:
                for ex in combined_dataset[:first_save]:
                    f_tmp.write(json.dumps(ex, ensure_ascii=False) + '\n')
            logger.info(f"Intermediate dataset saved: {first_save} examples to {output_file}")
            print(f"Intermediate dataset saved: {first_save} examples to {output_file}")
        except Exception as e:
            logger.warning(f"Failed intermediate save at {first_save} examples: {e}")
        # Subsequent saves if more data
        if total > save_interval:
            logger.info(f"Continuing intermediate saves every {save_interval} examples...")
            print(f"Continuing intermediate saves every {save_interval} examples...")
            for idx in range(save_interval * 2, total+1, save_interval):
                try:
                    with open(output_file, 'w', encoding='utf-8') as f_tmp:
                        for ex in combined_dataset[:idx]:
                            f_tmp.write(json.dumps(ex, ensure_ascii=False) + '\n')
                    logger.info(f"Intermediate dataset saved: {idx} examples to {output_file}")
                    print(f"Intermediate dataset saved: {idx} examples to {output_file}")
                except Exception as e:
                    logger.warning(f"Failed intermediate save at {idx} examples: {e}")
    # --- Handle Splitting and Saving ---
    final_data_structure: Union[List[Dict], Dict[str, List[Dict]]]
    n = len(combined_dataset)

    if split:
        train_size = int(split_ratios[0] * n)
        dev_size = int(split_ratios[1] * n)
        # test_size = n - train_size - dev_size # Remainder is test

        train_data = combined_dataset[:train_size]
        dev_data = combined_dataset[train_size : train_size + dev_size]
        test_data = combined_dataset[train_size + dev_size :]

        split_data = {"train": train_data, "dev": dev_data, "test": test_data}
        final_data_structure = split_data
        logger.info(f"Dataset split into Train ({len(train_data)}), Dev ({len(dev_data)}), Test ({len(test_data)}).")

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            try:
                for split_name, data in split_data.items():
                    filepath = os.path.join(output_dir, f"{split_name}.jsonl")
                    with open(filepath, 'w', encoding='utf-8') as f:
                        for ex in data:
                            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
                    logger.info(f"Saved {split_name} split to {filepath}")
            except Exception as e:
                logger.exception(f"Error saving split datasets to {output_dir}")
                # Continue to return data even if saving fails
    else:
        # Not splitting, use the full dataset
        final_data_structure = combined_dataset
        logger.info(f"Generated single dataset with {len(combined_dataset)} examples.")
        if output_file:
            try:
                # Ensure directory exists for output file
                out_dir = os.path.dirname(output_file)
                if out_dir and not os.path.exists(out_dir):
                    os.makedirs(out_dir, exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    for ex in combined_dataset:
                        f.write(json.dumps(ex, ensure_ascii=False) + '\n')
                logger.info(f"Saved full dataset to {output_file}")
            except Exception as e:
                logger.exception(f"Error saving full dataset to {output_file}")
                # Continue to return data

    # --- Report Final Statistics ---
    logger.info("\n--- Generation Statistics ---")
    originals = stats.get("original_examples_added", 0)
    perturbed = stats.get("perturbed_examples_added", 0)
    total_generated = originals + perturbed
    logger.info(f"  Input items processed: {stats.get('input_examples_processed', 'N/A')} (Skipped: {stats.get('input_examples_skipped', 'N/A')})")
    logger.info(f"  Original AMRs generated: {originals}")
    logger.info(f"  Perturbed AMRs generated: {perturbed}")
    logger.info(f"  Total examples in final dataset: {total_generated}")

    successful_pert = stats.get('total_successful_perturbations', 0)
    total_attempts = stats.get('total_attempts', 0)
    if total_attempts > 0:
        success_rate = (successful_pert / total_attempts) * 100
        logger.info(f"  Perturbation success rate: {success_rate:.1f}% ({successful_pert}/{total_attempts} attempts)")
    else:
        logger.info("  No perturbation attempts were made.")

    if perturbed > 0:
        logger.info("  Final counts by perturbation type (successful):")
        pert_counts = {}
        for ex in final_data_structure if not split else [item for sublist in final_data_structure.values() for item in sublist]:
            if ex["score"] == 0.0:
                ptype = ex.get("perturbation_type", "unknown")
                pert_counts[ptype] = pert_counts.get(ptype, 0) + 1
        for ptype, count in sorted(pert_counts.items()):
             logger.info(f"    {ptype}: {count} ({count/perturbed*100:.1f}%)")

    logger.info("\n  Detailed Perturbation Stats:")
    logger.info(f"    Parsing: Success={stats['parsing']['success']}, Failure={stats['parsing']['failure']}")
    logger.info(f"    Encoding: Success={stats['encoding']['success']}, Failure={stats['encoding']['failure']}")

    logger.info("  Initial Selection Counts:")
    for ptype, count in sorted(stats["selection"].items()):
        if count > 0: logger.info(f"    {ptype}: {count}")

    logger.info("  Success/Failure per Type (including fallbacks):")
    for ptype in perturbation_weights.keys():
        success = stats["perturbation"][ptype]["success"]
        failure = stats["perturbation"][ptype]["failure"]
        total = success + failure
        if total > 0:
            rate = success / total * 100
            logger.info(f"    {ptype}: {rate:.1f}% success ({success} success / {failure} failures)")
        elif ptype in stats["selection"] and stats["selection"][ptype] > 0:
             logger.info(f"    {ptype}: Selected but 0 success/failure recorded (check logic)")
        # else: # Don't report if never selected/attempted
        #    logger.info(f"    {ptype}: Not attempted")


    total_fallbacks = sum(stats["fallback"].values())
    if total_fallbacks > 0:
         logger.info("  Fallback Usage (successful perturbations via fallback):")
         for ptype, count in sorted(stats["fallback"].items()):
            if count > 0: logger.info(f"    {ptype}: {count} times")
    else:
         logger.info("  Fallback Usage: No fallbacks were successfully used.")
    logger.info("--- End Statistics ---")


    return final_data_structure, stats

# --- Example Usage (for Notebook) ---
# Assuming you have your perturbation modules in a 'data_perturber' directory
# relative to your notebook or in your Python path.

# if __name__ == '__main__':
#     # This block will not run automatically in a notebook cell import
#     # Put example calls directly in notebook cells.
#
#     # Example Call 1: Generate and return data, don't save files
#     # try:
#     #     dataset, stats = create_amr_dataset(
#     #         input_file='path/to/your/input.json',
#     #         perturbed_per_original=3,
#     #         max_examples=100, # Optional: for quick testing
#     #         seed=42,
#     #         verbose=True,
#     #         log_file='dataset_gen.log' # Log to file as well
#     #     )
#     #     print(f"\nGenerated {len(dataset)} examples.")
#     #     # print("First 5 examples:", dataset[:5])
#     #     # print("\nStats:", json.dumps(stats, indent=2))
#     # except FileNotFoundError:
#     #     print("ERROR: Input file not found. Please adjust 'input_file' path.")
#     # except ImportError as e:
#     #      print(f"ERROR: Could not import perturbation modules: {e}")
#     # except Exception as e:
#     #      print(f"An unexpected error occurred: {e}")
#     #      import traceback
#     #      traceback.print_exc()
#
#     # Example Call 2: Generate, split, and save to directory
#     # try:
#     #     split_datasets, stats = create_amr_dataset(
#     #         input_file='path/to/your/input.json',
#     #         output_dir='output_splits', # Specify directory for splits
#     #         split=True,
#     #         perturbed_per_original=5,
#     #         seed=123,
#     #         verbose=True,
#     #         perturbation_weights = { # Custom weights
#     #             "predicate": 0.3,
#     #             "circumstance": 0.3,
#     #             "entity": 0.1,
#     #             "discourse": 0.15,
#     #             "out_of_article": 0.15
#     #         }
#     #     )
#     #     print(f"\nGenerated and split data.")
#     #     print(f"  Train size: {len(split_datasets['train'])}")
#     #     print(f"  Dev size:   {len(split_datasets['dev'])}")
#     #     print(f"  Test size:  {len(split_datasets['test'])}")
#     #     # print("\nStats:", json.dumps(stats, indent=2))
#     # except FileNotFoundError:
#     #     print("ERROR: Input file not found. Please adjust 'input_file' path.")
#     # except ImportError as e:
#     #      print(f"ERROR: Could not import perturbation modules: {e}")
#     # except Exception as e:
#     #      print(f"An unexpected error occurred: {e}")
#     #      import traceback
#     #      traceback.print_exc()
