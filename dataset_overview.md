# AMR Perturbation Dataset Overview

## Project Structure
- `data_perturber/`: Contains modules for perturbing AMR graphs
  - `__init__.py`: Package initialization
  - `circumstance_perturber.py`: Modifies time/location/modality in AMR graphs
  - `entity_perturber.py`: Handles agent-patient role swaps and entity changes
  - `predicates_perturber.py`: Modifies predicates via antonyms and polarity
  - `insertion.py`: Main interface for applying perturbations
  - `utils.py`: Helper functions for graph conversion and antonym lookup
- `xlsum_indonesian/`: Contains processed Indonesian AMR dataset
  - `Processed AMR Val Dataset.json`: Validation dataset (too large to view directly)

## Perturbation Types

### 1. Entity Perturbations
- Swaps agent-patient roles (ARG0/ARG1) in predicates
- Changes quantity sources
- Maintains graph connectivity

### 2. Predicate Perturbations
- Replaces predicates with Indonesian antonyms
- Adds/removes polarity (negation)
- Modifies sentence-level relations

### 3. Circumstance Perturbations
- Modifies modality (certainty/possibility):
  - e.g. "possible" → "obligate", "maybe" → "certain"
- Changes time/location entities:
  - Swaps specific times, locations
  - Modifies temporal/spatial relations

## Technical Implementation
- Uses Penman library for AMR graph manipulation
- Converts between Penman and NetworkX graph formats for processing
- Leverages WordNet and Google Translate for Indonesian antonym lookup
- Provides clean interface via `insertion.py` for applying perturbations

## Dataset Characteristics
- Contains Indonesian language AMR graphs
- Appears to be validation data (based on "Val" in filename)
- JSON format (exact structure not viewable due to size)
- Used with perturbation tools to generate modified versions
