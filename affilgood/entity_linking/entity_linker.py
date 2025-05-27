import os
import re
import sys
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from .data_manager import DataManager
from .constants import *

# Make sure that S2AFF is in the path to avoid changing the code in S2AFF implementations.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), S2AFF_PATH)))

def json_serializable(obj):
    """Convert NumPy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [json_serializable(x) for x in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return json_serializable(obj.tolist())
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

class EntityLinker:
    """Main entity linker class supporting multiple matching methods."""

    def __init__(self, linkers=[], return_scores=True, output_dir=OUTPUT_PARTIAL_CHUNKS):
        """
        Initialize with a list of linker instances.
        
        Args:
            linkers: List of linker instances or class names (strings).
            output_dir: Directory for partial output chunks
        """
        # Return probabilities?
        self.return_scores = return_scores
        # Set up output directory.
        self.output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_dir)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path, exist_ok=True)
        # Initialize data manager.
        self.data_manager = DataManager()
        # Initialize linkers.
        self._initialize_linkers(linkers)


    def _initialize_linkers(self, linkers):
        """Initialize linkers from instances or class names."""
        # Initialize linkers dictionary
        self.linkers = {}
        for linker in linkers:
            if isinstance(linker, str):
                # It's a class name, we need to instantiate it
                if linker == 'S2AFF':
                    from .s2aff_linker import S2AFFLinker
                    self.linkers[linker] = S2AFFLinker(data_manager=self.data_manager)
                elif linker == 'Whoosh':
                    from .whoosh_linker import WhooshLinker
                    self.linkers[linker] = WhooshLinker(data_manager=self.data_manager)
                else:
                    raise ValueError(f"Unknown linker type: {linker}")
            else:
                # It's already an instance, just add it to the dictionary
                # The name is derived from the class name
                linker_name = linker.__class__.__name__
                self.linkers[linker_name] = linker
                # If the linker doesn't have a data_manager, provide one
                if hasattr(linker, 'data_manager') and linker.data_manager is None:
                    linker.data_manager = self.data_manager

    def process_in_chunks(self, entities, output_dir=None):
        """Process text in chunks with parallelization and optional saving."""
        # Divide input into chunks
        chunks = [entities[i:i + CHUNK_SIZE_EL] for i in range(0, len(entities), CHUNK_SIZE_EL)]
        
        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_EL) as executor:
            futures = [
                executor.submit(self.process_chunk_el, chunk)
                for chunk in chunks
            ]
        
        # Collect results
        results = []
        for idx, future in enumerate(futures):
            chunk_result = future.result()
            results.extend(chunk_result if isinstance(chunk_result, list) else self.merge_results(chunk_result))
            # Save intermediate results if required
            if SAVE_CHUNKS_EL:
                output_file = os.path.join(self.output_path, f"chunk_{idx}.json")
                with open(output_file, "w") as f:
                    json.dump(json_serializable(chunk_result), f)
        
        # No need to merge again if we've already merged each chunk.
        return results

    def process_chunk_el(self, chunk):
        """Process a chunk using the selected linkers."""
        results = {}
        for method, linker in self.linkers.items():
            results[method] = linker.process_chunk_el(chunk, return_scores=len(self.linkers)>1 or self.return_scores)
        # If only one method, return its results directly
        return next(iter(results.values())) if len(results) == 1 else results

    def merge_results(self, results):
        """
        Merge results from multiple entity linking methods.
        """
        # If results is a dictionary with linker methods as keys
        if isinstance(results, dict) and all(method in self.linkers for method in results):
            return self._merge_method_results(results)
        
        # If we have a list of results from different chunks
        merged = []
        for item in results:
            if isinstance(item, dict) and all(method in self.linkers for method in item):
                merged.append(self._merge_item_results(item))
            else:
                merged.append(item)
        return merged

    def _merge_method_results(self, method_results):
        """Merge results from different methods for all chunks."""
        merged = []
        for chunk_idx in range(len(next(iter(method_results.values())))):
            # Start with common keys that should be present in all methods
            any_method = next(iter(method_results.values()))
            merged_item = {
                "raw_text": any_method[chunk_idx]["raw_text"],
                "span_entities": any_method[chunk_idx]["span_entities"],
            }
            
            # Collect all possible keys from all methods
            all_keys = set()
            for method_name, method_results_list in method_results.items():
                all_keys.update(method_results_list[chunk_idx].keys())
            
            # Skip 'raw_text' and 'span_entities' as we've already handled them
            all_keys -= {"raw_text", "span_entities", "ror"}
            
            # For each key, use the first non-empty value found in any method
            for key in all_keys:
                for method_name, method_results_list in method_results.items():
                    if key in method_results_list[chunk_idx] and method_results_list[chunk_idx][key]:
                        merged_item[key] = method_results_list[chunk_idx][key]
                        break
            
            # Special handling for 'ror' as it needs to be merged, not just copied
            merged_item["ror"] = self._merge_ror_predictions([
                method_results[method][chunk_idx].get("ror", []) for method in method_results
            ])
            
            merged.append(merged_item)
        return merged

    def _merge_item_results(self, item_results):
        """Merge results from different methods for a single item."""
        merged_item = {
            "raw_text": next(iter(item_results.values()))["raw_text"],
            "span_entities": next(iter(item_results.values()))["span_entities"],
            "ner": next(iter(item_results.values()))["ner"],
            "osm": next(iter(item_results.values()))["osm"],
            "ror": self._merge_ror_predictions([
                item_results[method]["ror"] for method in item_results
            ])
        }
        return merged_item

    def _merge_ror_predictions(self, predictions_list):
        """
        Merge ROR predictions from different methods.
        Takes a list of prediction sets and merges them, prioritizing by score.
        """
        if not predictions_list or len(predictions_list) == 0:
            return []
        
        # Prepare all predictions in a consistent format - list of lists of strings
        normalized_predictions = []
        for preds in predictions_list:
            if isinstance(preds, str):
                # Handle string format (pipe-delimited)
                normalized_predictions.append(preds.split('|') if preds else [])
            elif isinstance(preds, list):
                # Already a list
                normalized_predictions.append(preds)
            else:
                # Empty or unrecognized format
                normalized_predictions.append([])
        
        # Process each position in parallel (e.g., first span's predictions, second span's predictions)
        merged_results = []
        
        # Make sure we have at least one set of predictions for each position
        max_length = max(len(preds) for preds in normalized_predictions) if normalized_predictions else 0
        for position in range(max_length):
            # Collect all predictions for this position
            position_preds = []
            for method_preds in normalized_predictions:
                if position < len(method_preds) and method_preds[position]:
                    position_preds.append(method_preds[position])
            
            # Parse predictions and aggregate by entity
            entity_scores = {}
            for pred_str in position_preds:
                # Split by pipe in case there are multiple predictions in one string
                for single_pred in (pred_str.split('|') if isinstance(pred_str, str) else [pred_str]):
                    if not single_pred:
                        continue
                    
                    # Remove ROR url if present.
                    single_pred = single_pred.replace(ROR_URL, "")
                    # Extract name and score
                    try:
                        if ':' in single_pred:
                            name_part, score_part = single_pred.rsplit(':', 1)
                            try:
                                score = float(score_part.strip()) if score_part.strip() else 0.0
                            except ValueError:
                                score = 0.0
                        else:
                            name_part = single_pred
                            score = 0.0
                        
                        # Update with higher score if entity already exists
                        entity_scores[name_part] = max(entity_scores.get(name_part, 0), score)
                    except Exception as e:
                        print(f"Error parsing prediction: {single_pred} - {str(e)}")
            
            # Sort by score descending and format results
            sorted_entities = sorted(entity_scores.items(), key=lambda x: -x[1])

            # Process entities with formatted ROR URLs.
            formatted_entities = []
            for name, score in sorted_entities:
                # Add ROR URL to the ID in curly braces
                formatted_name = re.sub(r'\{([0-9a-z]+)\}', lambda m: '{' + ROR_URL + m.group(1) + '}', name)
                formatted_entities.append((formatted_name, score))

            # Create the output string
            if self.return_scores:
                merged_str = '|'.join(f"{name}:{score:.2f}" for name, score in formatted_entities)
            else:
                merged_str = '|'.join(name for name, score in formatted_entities)

            merged_results.append(merged_str)
        
        return merged_results
