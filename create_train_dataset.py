import json
import torch
import penman
import logging
import os
import re
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, T5TokenizerFast
from pathlib import Path
# from huggingface_hub import snapshot_download # Not used directly, but could be added if model download is needed

# --- Configuration ---
MODEL_PATH = "../model/amr2text/taufiq-indo-amr-generation-gold-uncased/checkpoint-1"
INPUT_FILE = 'indonesia_amr_perturber/train2.json'
OUTPUT_FILE = "train_output.json"
ERROR_LOG_FILE = "error_log.txt"
# PROGRESS_FILE = "progress_tracker.json" # Simple resumability via output file check is used

# --- Logging Setup ---
# Create logs directory if it doesn't exist
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)
error_log_path = log_dir / ERROR_LOG_FILE

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(error_log_path), # Log errors and info to a file
        logging.StreamHandler()             # Also print logs to console
    ]
)

# --- Helper Function: Pointer Conversion ---
def to_amr_with_pointer(amr_string):
    """
    Convert AMR string to a linearized format with pointers,
    suitable for T5 input.
    Example: (w / want-01 :arg0 (b / boy)) -> (<pointer:0> :arg0 <pointer:1>)
    """
    # Normalize whitespace
    amr_string = amr_string.replace('\n', ' ').strip()
    amr_string = ' '.join(amr_string.split())

    # Find all variables defined with ' / ' (e.g., "b / boy")
    variables = re.findall(r'\(([a-zA-Z0-9]+)\s*/', amr_string)
    # Create unique pointers for each variable
    var_map = {var: f"<pointer:{i}>" for i, var in enumerate(dict.fromkeys(variables))} # Use dict.fromkeys to keep order and uniqueness

    # --- Substitution Logic ---
    # 1. Replace variable definitions: (var / concept) -> (<pointer:id> / concept)
    #    We need to be careful not to replace parts of concepts or relations.
    #    Iterate through the map to perform replacements.
    temp_amr_string = amr_string
    for var, pointer in var_map.items():
         # Regex to match '(var /' ensuring 'var' is preceded by '(' and followed by ' /'
         pattern = r'\(' + re.escape(var) + r'\s+/'
         temp_amr_string = re.sub(pattern, f"({pointer} /", temp_amr_string)

    # 2. Replace variable references (e.g., :arg0 var) -> :arg0 <pointer:id>
    #    Sort by length descending to avoid replacing 'a' inside 'a1'.
    for var in sorted(var_map.keys(), key=len, reverse=True):
        pointer = var_map[var]
        # Regex to match ' delim' + var + '([ )])'
        # Ensures var is preceded by a delimiter (space, :, () and followed by space or )
        pattern = r'([ :(])' + re.escape(var) + r'(?=[ )])'
        # Replace with the delimiter + pointer. The lookahead (?=...) ensures the closing ) or space isn't consumed.
        temp_amr_string = re.sub(pattern, r'\1' + pointer, temp_amr_string)


    # 3. Remove the '/ concept' part after the variable definition is replaced
    #    Pattern: '<pointer:id> / concept-name' -> '<pointer:id>'
    #    Matches space + / + space + one or more non-space/non-) characters
    final_amr_string = re.sub(r'\s+/\s+[^ )\s]+', '', temp_amr_string)

    # Final cleanup of extra spaces that might result from replacements
    final_amr_string = ' '.join(final_amr_string.split())

    return final_amr_string


# --- AMR-to-Text Class ---
class AMRToText:
    def __init__(self, model_path):
        """Initialize the AMR-to-Text converter."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {self.device}")

        model_dir = Path(model_path)
        # T5 models expect tokenizer files directly in the model directory or a subdir named 'tokenizer'
        # Let's assume they are directly in model_dir for simplicity based on original code structure
        # If they are in subdirs, adjust paths accordingly e.g. tokenizer_path = model_dir / 'tokenizer'

        if not model_dir.exists():
             logging.error(f"Model directory not found: {model_dir}")
             # Consider adding download logic here if needed
             raise FileNotFoundError(f"Model directory not found: {model_dir}. Please ensure the path is correct.")

        # Initialize tokenizer
        tokenizer_path = model_dir / 'tokenizer'
        try:
            # Load from the 'tokenizer' subdirectory
            self.tokenizer = T5TokenizerFast.from_pretrained(tokenizer_path)
            logging.info(f"Tokenizer loaded successfully from {tokenizer_path}")
        except Exception as e:
            logging.error(f"Failed to load tokenizer from {tokenizer_path}: {e}")
            raise

        # Initialize model
        model_weights_path = model_dir / 'model'
        try:
            # Load model from the 'model' subdirectory
            model = AutoModelForSeq2SeqLM.from_pretrained(model_weights_path)
            model.to(self.device)
            model.eval()
            self.model = model
            logging.info(f"Model loaded successfully from {model_weights_path}")
        except Exception as e:
            logging.error(f"Failed to load model from {model_weights_path}: {e}")
            raise

        self.lowercase = False # Option to lowercase input AMR before processing

        # Set generation parameters
        self.max_length = 384
        self.num_beams = 5
        self.T5_PREFIX = "translate graph to indonesian: " # Task prefix for T5

    def convert_single_graph(self, graph):
        """Converts a single penman.Graph to text."""
        if not isinstance(graph, penman.Graph):
            logging.error(f"Input is not a penman.Graph object: {type(graph)}")
            return None # Indicate failure

        # Remove metadata (e.g., ::id, ::snt)
        graph.metadata = {}

        # Convert graph to linearized pointer format
        try:
            # Use indent=None for compact, single-line representation
            encoded_graph = penman.encode(graph, indent=None)
            pointer_text = to_amr_with_pointer(encoded_graph)
            if not pointer_text: # Handle cases where conversion might fail
                 logging.warning(f"Pointer conversion resulted in empty string for graph: {encoded_graph}")
                 return None
        except Exception as e:
            logging.error(f"Error converting graph to pointer format: {e}\nGraph: {penman.encode(graph)}")
            return None # Indicate failure

        if self.lowercase:
            pointer_text = pointer_text.lower()

        # Prepare input for the T5 model
        input_text = f"{self.T5_PREFIX}{pointer_text}"
        try:
            input_ids = self.tokenizer.encode(
                input_text,
                return_tensors="pt",
                add_special_tokens=True,
                truncation=True, # Ensure input doesn't exceed model max length
                max_length=512 # T5 typically has a 512 token limit for input
            ).to(self.device)
        except Exception as e:
            logging.error(f"Error tokenizing input: {e}\nInput text: {input_text}")
            return None

        # Generate text using the model
        try:
            with torch.no_grad(): # Disable gradient calculations for inference
                 outputs = self.model.generate(
                    input_ids,
                    num_beams=self.num_beams,
                    max_length=self.max_length, # Max length for the generated output
                    early_stopping=True # Stop generation when EOS token is produced
                )
            # Decode the generated token IDs to a string
            gen_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return gen_text
        except Exception as e:
            logging.error(f"Error during model generation: {e}\nInput IDs shape: {input_ids.shape}")
            return None


# --- Main Processing Function ---
def process_amr_file(input_path, output_path, model_converter):
    """Loads data, processes AMR graphs, handles errors, and saves results with resumability."""

    input_filepath = Path(input_path)
    output_filepath = Path(output_path)

    # --- Load Input Data ---
    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        logging.info(f"Loaded {len(all_data)} items from {input_filepath}")
    except FileNotFoundError:
        logging.error(f"Input file not found: {input_filepath}")
        return
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from {input_filepath}")
        return
    except Exception as e:
        logging.error(f"Error loading data from {input_filepath}: {e}")
        return

    # --- Resumability: Load existing output if available ---
    processed_data = []
    start_index = 0
    if output_filepath.exists():
        try:
            with open(output_filepath, 'r', encoding='utf-8') as f:
                processed_data = json.load(f)
            # Assume order is preserved. Resume based on the count of processed items.
            start_index = len(processed_data)
            logging.info(f"Output file found. Resuming from index {start_index}. Loaded {start_index} previously processed items.")
            if start_index >= len(all_data):
                 logging.info("All items seem to be processed already.")
                 return # Nothing more to do
        except (json.JSONDecodeError, Exception) as e:
            logging.warning(f"Could not load or parse existing output file {output_filepath}. Starting from scratch. Error: {e}")
            processed_data = [] # Reset if file is corrupted
            start_index = 0
    else:
         logging.info(f"Output file {output_filepath} not found. Starting from scratch.")


    # --- Process Remaining Data ---
    # Slice the data list to get only the items that need processing
    data_to_process = all_data[start_index:]
    newly_processed_count = 0

    # Use the already loaded processed_data as the base for the final output
    final_output_data = processed_data

    # Iterate through the remaining items with tqdm progress bar
    for index, item in enumerate(tqdm(data_to_process, desc="Converting AMR to text", initial=start_index, total=len(all_data))):
        actual_index = start_index + index # Get the original index in all_data
        amr_string = item.get('amr') # Use .get() for safer access

        if not amr_string:
            logging.warning(f"Item at index {actual_index} missing 'amr' field. Skipping.")
            item['text_from_amr'] = "error: missing amr field"
            final_output_data.append(item) # Add item to output, marked with error
            continue

        try:
            # Decode the AMR string into a penman graph object
            amr_graph = penman.decode(amr_string)

            # Convert the graph to text using the initialized model converter
            generated_text = model_converter.convert_single_graph(amr_graph)

            if generated_text is not None:
                item['text_from_amr'] = generated_text
            else:
                # Error occurred during conversion (already logged in convert_single_graph)
                item['text_from_amr'] = "error: conversion failed"
                logging.error(f"Failed to convert AMR at index {actual_index}. Check logs above for details.")

        except penman.exceptions.DecodeError as e:
            # Handle errors specifically related to parsing the AMR string
            logging.error(f"Error decoding AMR string at index {actual_index}: {e}\nAMR String (first 100 chars): {amr_string[:100]}...")
            item['text_from_amr'] = "error: invalid amr format"
        except Exception as e:
            # Catch any other unexpected errors during the processing of this single item
            logging.exception(f"Unexpected error processing item at index {actual_index}: {e}") # Includes traceback
            item['text_from_amr'] = "error: unexpected processing error"

        # Add the processed (or error-marked) item to our final list
        final_output_data.append(item)
        newly_processed_count += 1

        # --- Save Progress Periodically (e.g., every 100 items) ---
        # This prevents losing all progress if the script crashes during a long run
        if newly_processed_count > 0 and newly_processed_count % 100 == 0:
            try:
                with open(output_filepath, "w", encoding="utf-8") as f:
                    json.dump(final_output_data, f, indent=4, ensure_ascii=False)
                logging.info(f"Saved intermediate progress: {len(final_output_data)} items written to {output_filepath}")
            except Exception as e:
                logging.error(f"Failed to save intermediate progress to {output_filepath}: {e}")


    # --- Final Save ---
    # Always save the complete results at the very end
    try:
        with open(output_filepath, "w", encoding="utf-8") as f:
            # Use ensure_ascii=False to correctly save non-ASCII characters (like Indonesian)
            json.dump(final_output_data, f, indent=4, ensure_ascii=False)
        logging.info(f"Processing complete. Total {len(final_output_data)} items saved to {output_filepath}")
    except Exception as e:
        logging.error(f"Failed to save final output to {output_filepath}: {e}")


# --- Main Execution Guard ---
if __name__ == "__main__":
    logging.info("Starting AMR-to-Text conversion process...")
    try:
        # Initialize the AMR-to-Text converter model
        amr_to_text_converter = AMRToText(MODEL_PATH)

        # Run the main processing logic
        process_amr_file(INPUT_FILE, OUTPUT_FILE, amr_to_text_converter)

    except FileNotFoundError as e:
         # Handle errors during initialization (e.g., model not found)
         logging.error(f"Initialization failed: {e}")
    except Exception as e:
        # Catch any other critical errors during setup or execution
        logging.exception(f"A critical error occurred during script execution: {e}")

    logging.info("Script finished.")