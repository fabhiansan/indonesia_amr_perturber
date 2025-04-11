#!/bin/bash

# ==============================================================================
# Bash script to run the AMR Perturbation Dataset Generator (Script 2)
# Make sure script2.py is in the same directory or provide the correct path.
# ==============================================================================

# --- Configuration ---
# --- !! MODIFY THESE PATHS AND SETTINGS ACCORDINGLY !! ---

# The name or path of your Python script file
PYTHON_SCRIPT_NAME="generate_ml_dataset.py"

# Path to the input JSON file containing AMR data
# INPUT_FILE="xlsum_indonesian/xl_sum_processed_amr_test_dataset2.json"
INPUT_FILE="xlsum_indonesian/xl_sum_processed_amr_val_dataset2.json"

# Path for the output file IF NOT splitting
OUTPUT_FILE_SINGLE="outputs/new_val.json"

# Directory to save train/dev/test splits IF splitting
OUTPUT_DIR_SPLIT="output/dataset_splits"

# Random seed for reproducibility (optional, remove --seed flag to disable)
SEED_VALUE=123

# Number of perturbed examples per original AMR
PERTURBED_PER_ORIGINAL=5

# Enable/Disable Perturbation Types (Set weight > 0 to enable)
# The script will randomly try ALL enabled perturbation types until one succeeds.
# The specific weight value (e.g., 0.1 vs 0.5) doesn't affect selection probability anymore, only if it's > 0.
WEIGHT_PREDICATE=0.2
WEIGHT_CIRCUMSTANCE=0.2
WEIGHT_ENTITY=0.2
WEIGHT_DISCOURSE=0.2
WEIGHT_OUT_OF_ARTICLE=0.2

# Optional arguments (set value or leave empty string "" to omit the flag)
# Example: MAX_EXAMPLES_FLAG="--max-examples 100" # Process only first 100 examples
MAX_EXAMPLES_FLAG=""

# Example: SAVE_INTERVAL_FLAG="--save-interval 5000" # Save every 5000 examples
SAVE_INTERVAL_FLAG="--save-interval 10000" # Default save interval

# Example: DEBUG_SAMPLE_FLAG="--debug-sample 5" # Debug first 5 examples
DEBUG_SAMPLE_FLAG=""

# Example: VERBOSE_FLAG="--verbose" # Enable verbose logging
VERBOSE_FLAG="--verbose"

# --- !! END OF CONFIGURATION !! ---


# --- Safety Check ---
# Ensure the Python script exists
if [ ! -f "$PYTHON_SCRIPT_NAME" ]; then
    echo "Error: Python script '$PYTHON_SCRIPT_NAME' not found!"
    echo "Please make sure the script is in the current directory or update PYTHON_SCRIPT_NAME."
    exit 1
fi


# --- Choose Execution Mode ---
# --- Uncomment (remove the '#' from the start) ONE of the following blocks ---
# --- Only one block should be active at a time ---

# === Block 1: Generate a SINGLE output file ===
echo "Running generation for a single output file..."
python "$PYTHON_SCRIPT_NAME" \
    "$INPUT_FILE" \
    "$OUTPUT_FILE_SINGLE" \
    --predicate "$WEIGHT_PREDICATE" \
    --circumstance "$WEIGHT_CIRCUMSTANCE" \
    --entity "$WEIGHT_ENTITY" \
    --discourse "$WEIGHT_DISCOURSE" \
    --out-of-article "$WEIGHT_OUT_OF_ARTICLE" \
    --perturbed-per-original "$PERTURBED_PER_ORIGINAL" \
    --seed "$SEED_VALUE" \
    $MAX_EXAMPLES_FLAG \
    $SAVE_INTERVAL_FLAG \
    $DEBUG_SAMPLE_FLAG \
    $VERBOSE_FLAG

# Check exit code of the python script
if [ $? -eq 0 ]; then
    echo "Single file generation complete. Output: $OUTPUT_FILE_SINGLE"
else
    echo "Error during Python script execution."
    exit 1
fi


# === Block 2: Generate SPLIT output files (train/dev/test) ===
# echo "Running generation for SPLIT output files..."
# Note: The second positional argument (output file name) is required by argparse
# but is ignored by the script logic when --split is used. We provide a placeholder.
# python "$PYTHON_SCRIPT_NAME" \
#     "$INPUT_FILE" \
#     "placeholder_output.json" \
#     --split \
#     --output-dir "$OUTPUT_DIR_SPLIT" \
#     --predicate "$WEIGHT_PREDICATE" \
#     --circumstance "$WEIGHT_CIRCUMSTANCE" \
#     --entity "$WEIGHT_ENTITY" \
#     --discourse "$WEIGHT_DISCOURSE" \
#     --out-of-article "$WEIGHT_OUT_OF_ARTICLE" \
#     --perturbed-per-original "$PERTURBED_PER_ORIGINAL" \
#     --seed "$SEED_VALUE" \
#     $MAX_EXAMPLES_FLAG \
#     $SAVE_INTERVAL_FLAG \
#     $DEBUG_SAMPLE_FLAG \
#     $VERBOSE_FLAG

# Check exit code of the python script
if [ $? -eq 0 ]; then
    echo "Split file generation complete. Output directory: $OUTPUT_DIR_SPLIT"
else
    echo "Error during Python script execution."
    exit 1
fi


# --- End of Execution Modes ---

echo "Bash script finished."
exit 0 # Indicate successful execution of the bash script itself