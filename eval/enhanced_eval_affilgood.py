#!/usr/bin/env python3
import os
import sys
import argparse
import csv
import pandas as pd
import numpy as np
import re
import time
import torch
from tqdm import tqdm
import json
    
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from affilgood import AffilGood
from affilgood.entity_linking.whoosh_linker import WhooshLinker
from affilgood.entity_linking.dense_linker import DenseLinker
from affilgood.entity_linking.entity_linker import EntityLinker

# Allow testing on small sample
TEST = False

DEFAULT_SPAN_MODEL = 'nicolauduran45/affilgood-span-v2'
DEFAULT_NER_MODEL = "nicolauduran45/affilgood-ner-multilingual-v2"
DEFAULT_ENTITY_LINKERS = 'Dense'

INITIAL_THRESHOLD_ENTITY_LINKING = 0
FINAL_THRESHOLD_ENTITY_LINKING = 0.8

def extract_ror_ids(text):
    """Extract ROR IDs from text with format like "Org Name {https://ror.org/01234abc}"."""
    if not text:
        return set()
    
    ror_ids = set()
    # Handle pipe-delimited items
    items = text.split('|') if isinstance(text, str) else [text]
    
    for item in items:
        if not item:
            continue
            
        # Extract ROR ID using regex
        matches = re.findall(r'{https://ror\.org/([0-9a-z]+)}', item)
        ror_ids.update(matches)
        
        # Also try to extract raw IDs without https prefix
        if not matches:
            matches = re.findall(r'{([0-9a-z]+)}', item)
            ror_ids.update(matches)
    
    return ror_ids

def calculate_metrics(gold_ids, pred_ids):
    """Calculate precision, recall, F1, incorrect and missing IDs."""
    gold_ids = set(gold_ids)
    pred_ids = set(pred_ids)
    
    # Calculate correct, incorrect, and missing
    correct = gold_ids.intersection(pred_ids)
    incorrect = pred_ids - gold_ids
    missing = gold_ids - pred_ids
    
    # Calculate metrics
    precision = len(correct) / len(pred_ids) if pred_ids else 0.0
    recall = len(correct) / len(gold_ids) if gold_ids else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'incorrect': incorrect,
        'missing': missing
    }

def evaluate_pipeline(affilgood, gold_data, batch_size=16):
    """Evaluate the pipeline against gold annotations using batch processing."""
    results = []
    all_metrics = {'precision': [], 'recall': [], 'f1': []}
    detailed_timings = {
        'translation': [],
        'span': [],
        'ner': [],
        'normalization': [],
        'el_retrieval': [],
        'el_reranking': []
    }
    
    # Initialize processing times
    total_time_start = time.time()
    
    # Extract affiliations
    affiliations = gold_data['raw_affiliation_string'].tolist()
    
    # Set up a hook to capture component timings
    timings_container = []
    
    def timing_hook(stage, elapsed):
        timings_container.append((stage, elapsed))
        
    # Add timing hooks to AffilGood components
    if hasattr(affilgood, 'span_identifier'):
        original_span_method = affilgood.span_identifier.identify_spans
        def timed_span_identifier(*args, **kwargs):
            start = time.time()
            result = original_span_method(*args, **kwargs)
            elapsed = time.time() - start
            timing_hook('span', elapsed)
            return result
        affilgood.span_identifier.identify_spans = timed_span_identifier
    
    if hasattr(affilgood, 'ner'):
        original_ner_method = affilgood.ner.recognize_entities
        def timed_ner(*args, **kwargs):
            start = time.time()
            result = original_ner_method(*args, **kwargs)
            elapsed = time.time() - start
            timing_hook('ner', elapsed)
            return result
        affilgood.ner.recognize_entities = timed_ner
        
    if hasattr(affilgood, 'normalizer'):
        original_norm_method = affilgood.normalizer.normalize
        def timed_normalize(*args, **kwargs):
            start = time.time()
            result = original_norm_method(*args, **kwargs)
            elapsed = time.time() - start
            timing_hook('normalization', elapsed)
            return result
        affilgood.normalizer.normalize = timed_normalize
    
    if hasattr(affilgood, 'entity_linker'):
        # Track retrieval phase for all individual linkers
        for linker in affilgood.entity_linker.linkers:
            if hasattr(linker, 'get_single_prediction'):
                original_get_prediction = linker.get_single_prediction
                def timed_get_prediction(*args, **kwargs):
                    start = time.time()
                    result = original_get_prediction(*args, **kwargs)
                    elapsed = time.time() - start
                    timing_hook('el_retrieval', elapsed)
                    return result
                linker.get_single_prediction = timed_get_prediction
        
        # Track reranking phase if reranker exists in the entity linker
        if affilgood.rerank and hasattr(affilgood.entity_linker, 'reranker') and affilgood.entity_linker.reranker:
            if hasattr(affilgood.entity_linker.reranker, 'rerank'):
                original_rerank = affilgood.entity_linker.reranker.rerank
                def timed_rerank(*args, **kwargs):
                    start = time.time()
                    result = original_rerank(*args, **kwargs)
                    elapsed = time.time() - start
                    timing_hook('el_reranking', elapsed)
                    return result
                affilgood.entity_linker.reranker.rerank = timed_rerank
    
    # Track language translation if enabled
    if affilgood.language_preprocessor:
        original_translate = affilgood.language_preprocessor.translate
        def timed_translate(*args, **kwargs):
            start = time.time()
            result = original_translate(*args, **kwargs)
            elapsed = time.time() - start
            timing_hook('translation', elapsed)
            return result
        affilgood.language_preprocessor.translate = timed_translate
    
    # Process in batches
    batch_results = []
    for i in range(0, len(affiliations), batch_size):
        batch = affiliations[i:i+batch_size]
        
        # Clear timings for this batch
        timings_container.clear()
        
        # Time the batch processing
        batch_start = time.time()
        batch_output = affilgood.process(batch)
        batch_end = time.time()
        
        # Record processing time for this batch
        batch_time = batch_end - batch_start
        batch_results.extend(batch_output)
        
        # Collect timings for each component
        batch_timings = {}
        for stage, elapsed in timings_container:
            if stage not in batch_timings:
                batch_timings[stage] = []
            batch_timings[stage].append(elapsed)
        
        # Calculate average time per affiliation for each component
        for stage, times in batch_timings.items():
            avg_time = sum(times) / len(batch)
            detailed_timings[stage].extend([avg_time] * len(batch))
        
        # Fill in zeros for any missing components in this batch
        for stage in detailed_timings.keys():
            if stage not in batch_timings:
                detailed_timings[stage].extend([0.0] * len(batch))
        
        print(f"Processed batch {i//batch_size+1}/{(len(affiliations)+batch_size-1)//batch_size}: "
              f"{len(batch)} affiliations in {batch_time:.2f}s "
              f"({batch_time/len(batch):.4f}s per affiliation)")
    
    # Calculate total processing time
    total_processing_time = time.time() - total_time_start
    
    # Process results
    for i, row in tqdm(gold_data.iterrows(), total=len(gold_data), desc="Evaluating Results"):
        affiliation = row['raw_affiliation_string']
        gold_labels = row['label']
        
        # Extract gold ROR IDs
        gold_ids = extract_ror_ids(gold_labels)
        
        # Get the batch processing results for this affiliation
        output = batch_results[i]
        
        # Extract predicted ROR IDs from all spans
        pred_ids = set()
        pred_labels = set()
        
        # Get the batch processing results for this affiliation
        output = batch_results[i]

        # Extract predicted ROR IDs from all spans
        pred_ids = set()
        pred_labels = set()

        #print(f"Output: {output}")

        # Process the ror field
        ror_prediction = output.get('ror', '')
        if ror_prediction:  # Only if the string is not empty
            # Split by | to get individual organization entries
            for org_entry in ror_prediction.split('|'):
                if org_entry.strip():  # Skip empty entries
                    pred_labels.add(org_entry.strip())
                    pred_ids.update(extract_ror_ids(org_entry))
        
        #print(f"Gold ids: {gold_ids}")
        #print(f"Pred ids: {pred_ids}")
        
        # Calculate metrics
        metrics = calculate_metrics(gold_ids, pred_ids)
        
        # Store metrics for averaging
        all_metrics['precision'].append(metrics['precision'])
        all_metrics['recall'].append(metrics['recall'])
        all_metrics['f1'].append(metrics['f1'])
        
        # Format sets as strings for output
        result_row = {
            'raw_affiliation_string': affiliation,
            'gold_label': gold_labels,
            'pred_label': '|'.join(list(pred_labels)),
            'incorrect': '|'.join(metrics['incorrect']),
            'missing': '|'.join(metrics['missing']),
            'P': metrics['precision'],
            'R': metrics['recall'],
            'F1': metrics['f1']
        }
        
        # Add component timings for this item
        for stage, times in detailed_timings.items():
            if i < len(times):
                result_row[f'time_{stage}'] = times[i]
        
        results.append(result_row)
    
    # Calculate average metrics
    avg_component_times = {}
    for stage, times in detailed_timings.items():
        if times:
            avg_component_times[stage] = np.mean(times)
        else:
            avg_component_times[stage] = 0.0
    
    avg_metrics = {
        'precision': np.mean(all_metrics['precision']),
        'recall': np.mean(all_metrics['recall']),
        'f1': np.mean(all_metrics['f1']),
        'avg_processing_time': total_processing_time / len(gold_data),
        'component_times': avg_component_times
    }
    
    return results, avg_metrics

def main():
    parser = argparse.ArgumentParser(description='Evaluate AffilGood pipeline on gold annotations.')
    parser.add_argument('--input', required=True, help='Path to input TSV or CSV file with gold annotations.')
    parser.add_argument('--output', required=True, help='Path to output evaluation file.')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for processing.')
    parser.add_argument('--span-model', default=DEFAULT_SPAN_MODEL, help='Span model path.')
    parser.add_argument('--ner-model', default=DEFAULT_NER_MODEL, help='NER model path.')
    parser.add_argument('--entity-linkers', default=DEFAULT_ENTITY_LINKERS, 
                       help='Entity linker(s) to be used. Separate multiple linkers with a comma.')
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--skip-rerank", action="store_true", help="Skip reranking")
    parser.add_argument('--device', default=None, help='Device for processing (cpu/cuda)')
    parser.add_argument("--skip-translation", action="store_true", help="Skip preprocessing/translation")
    parser.add_argument("--external-llm-translation", action="store_true", help="Use external LLM for translation")
    parser.add_argument('--linker-threshold', type=float, default=INITIAL_THRESHOLD_ENTITY_LINKING, help='Threshold score for entity linking.')
    parser.add_argument('--final-threshold', type=float, default=FINAL_THRESHOLD_ENTITY_LINKING, help='Final threshold score.')
    parser.add_argument('--use-cache-translation', action="store_true", help='Enable caching for translation')
    parser.add_argument('--use-cache-norm', action="store_true", help='Enable caching for normalization')
    parser.add_argument('--use-cache-retrieval', action="store_true", help='Enable caching for entity linking retrieval')
    parser.add_argument('--use-cache-reranking', action="store_true", help='Enable caching for entity linking reranking')
    parser.add_argument('--dataset-name', default="", help='Name of the dataset (for output)')
    
    args = parser.parse_args()
    
    print("\n===== Evaluation Configuration =====")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("===================================\n")
    
    # Load gold annotations
    if args.input.endswith('.tsv'):
        gold_data = pd.read_csv(args.input, sep='\t').fillna('')
    elif args.input.endswith('.csv'):
        gold_data = pd.read_csv(args.input).fillna('')
    else:
        print(f'Input format not supported for file {args.input}')
        return
    
    if TEST:
        gold_data = gold_data.head(20)
    
    if 'raw_affiliation_string' not in gold_data and 'ORG' in gold_data and 'CITY' in gold_data and 'COUNTRY' in gold_data:
        gold_data['CITY'] = gold_data['CITY'].str.title()
        gold_data['raw_affiliation_string'] = gold_data[['ORG', 'CITY', 'COUNTRY']].apply(
            lambda x: ', '.join(filter(None, x)), axis=1
        )
    
    # Initialize entity linkers with specific configurations
    entity_linkers = [linker.strip() for linker in args.entity_linkers.split(',')]
    
    # Dictionary to collect all configuration parameters
    config_params = {
        "args": vars(args).copy(),
        "component_params": {}
    }
    
    # Customize reranking and entity linker settings based on arguments
    rerank = not args.skip_rerank
    
    # Create entity linker objects with configured caching for retrieval
    entity_linker_objects = []
    config_params["component_params"]["linkers"] = []
    for linker in entity_linkers:
        if linker == 'Whoosh':
            linker_obj = WhooshLinker(threshold_score=args.linker_threshold, use_cache=args.use_cache_retrieval)
            entity_linker_objects.append(linker_obj)
        else:
            linker_obj = DenseLinker(threshold_score=args.linker_threshold, use_cache=args.use_cache_retrieval)
            entity_linker_objects.append(linker_obj)
       
    config_params["component_params"]["linkers"] = [
        linker_obj.__class__.__name__ if hasattr(linker_obj, '__class__') else linker_obj 
        for linker_obj in entity_linker_objects
    ]

    print(f"Initializing AffilGood pipeline...")
    affilgood = AffilGood(
        span_model_path=args.span_model,
        ner_model_path=args.ner_model,
        entity_linkers=entity_linker_objects,
        device=args.device,
        rerank=rerank,
        final_threshold_entity_linking=args.final_threshold,
        language_preprocessing=not args.skip_translation,
        use_external_llm_translate=args.external_llm_translation,
        use_osm_cache=args.use_cache_norm,
        verbose=args.verbose,
        return_scores=True
    )

    entity_linking_handler = affilgood.entity_linker

    # Configure reranking cache if reranker exists
    if rerank and hasattr(entity_linking_handler, 'reranker') and entity_linking_handler.reranker:
        # Set up reranker cache
        if args.use_cache_reranking and hasattr(entity_linking_handler.reranker, 'set_cache_enabled'):
            entity_linking_handler.reranker.set_cache_enabled(True)
            if args.verbose:
                print(f"Enabled cache for reranker: {entity_linking_handler.reranker.__class__.__name__}")
        else:
            # Explicitly disable cache
            if hasattr(entity_linking_handler.reranker, 'set_cache_enabled'):
                entity_linking_handler.reranker.set_cache_enabled(False)
                if args.verbose:
                    print(f"Disabled cache for reranker: {entity_linking_handler.reranker.__class__.__name__}")
            elif hasattr(entity_linking_handler.reranker, 'use_cache'):
                entity_linking_handler.reranker.use_cache = False
                if args.verbose:
                    print(f"Disabled cache for reranker (direct attribute): {entity_linking_handler.reranker.__class__.__name__}")

    # If language preprocessor is enabled, configure its cache
    if hasattr(affilgood, 'language_preprocessor') and affilgood.language_preprocessor:
        affilgood.language_preprocessor.use_cache = args.use_cache_translation
    
    # Collect actual model paths and configuration
    config_params["component_params"]["span_identifier"] = {
        "model_path": affilgood.span_identifier.__class__.__name__,
        "actual_model": getattr(affilgood.span_identifier, 'model_path', 'N/A')
    }
    
    config_params["component_params"]["ner"] = {
        "model_path": affilgood.ner.model_path if hasattr(affilgood.ner, 'model_path') else 'N/A',
        "batch_size": affilgood.ner.batch_size if hasattr(affilgood.ner, 'batch_size') else 'N/A'
    }
    
    config_params["component_params"]["language_preprocessing"] = {
        "enabled": not args.skip_translation,
        "model": getattr(affilgood.language_preprocessor, 'model_name', 'N/A') if hasattr(affilgood, 'language_preprocessor') else 'N/A'
    }
    
    # Collect reranker information if available
    if hasattr(affilgood.entity_linker, 'reranker') and affilgood.entity_linker.reranker:
        config_params["component_params"]["reranker"] = {
            "type": affilgood.entity_linker.reranker.__class__.__name__,
            "model": getattr(affilgood.entity_linker.reranker, 'model_name', 'N/A')
        }
    
    # Record device actually used
    config_params["device"] = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Add cache settings to configuration
    config_params["caching"] = {
        "translation": args.use_cache_translation,
        "normalization": args.use_cache_norm,
        "retrieval": args.use_cache_retrieval,
        "reranking": args.use_cache_reranking
    }
    
    # Run evaluation
    print(f"Evaluating on {len(gold_data)} gold annotations with batch size {args.batch_size}...")
    results, avg_metrics = evaluate_pipeline(affilgood, gold_data, batch_size=args.batch_size)
       
    # Create output directory if it does not exist.
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Prepare metrics for output
    metrics_for_output = {
        "dataset": args.dataset_name,
        "method": args.entity_linkers,
        "translation": not args.skip_translation,
        "cache_translation": args.use_cache_translation,
        "cache_norm": args.use_cache_norm,
        "cache_retrieval": args.use_cache_retrieval,
        "cache_reranking": args.use_cache_reranking,
        "precision": avg_metrics['precision'],
        "recall": avg_metrics['recall'],
        "f1": avg_metrics['f1'],
        "num_records": len(gold_data),
        "avg_time_total": avg_metrics['avg_processing_time']
    }
    
    # Add component-specific timing metrics
    for component, avg_time in avg_metrics['component_times'].items():
        metrics_for_output[f"avg_time_{component}"] = avg_time
    
    # Write results to output file
    print(f"Writing evaluation results to {args.output}")
    results_df = pd.DataFrame(results)
    
    # Also write configuration and metrics to a separate JSON file
    config_output = args.output.replace(".tsv", "_config.json").replace(".csv", "_config.json")
    with open(config_output, 'w') as f:
        json.dump({
            "metrics": metrics_for_output,
            "config": config_params
        }, f, indent=2)
    
    # Add important configuration parameters as metadata columns
    results_df['span_model'] = args.span_model
    results_df['ner_model'] = args.ner_model
    results_df['entity_linkers'] = args.entity_linkers
    results_df['linker_threshold'] = args.linker_threshold
    results_df['final_threshold'] = args.final_threshold
    results_df['dataset'] = args.dataset_name
    
    results_df.to_csv(args.output, sep='\t', index=False)
    
    # Create a summary file with just the key metrics in TSV format for easier table creation
    summary_output = args.output.replace(".tsv", "_summary.tsv").replace(".csv", "_summary.tsv")
    with open(summary_output, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        # Write header
        writer.writerow([
            "Dataset", "Method", "Translation", "Cache_Translation", "Cache_Norm", 
            "Cache_Retrieval", "Cache_Reranking", "Time_Translation", "Time_Span", 
            "Time_NER", "Time_Normalization", "Time_EL_Retrieval", "Time_EL_Reranking", 
            "Precision", "Recall", "F1_Score", "Num_Records"
        ])
        # Write data row
        writer.writerow([
            metrics_for_output["dataset"],
            metrics_for_output["method"],
            "Yes" if metrics_for_output["translation"] else "No",
            "Yes" if metrics_for_output["cache_translation"] else "No",
            "Yes" if metrics_for_output["cache_norm"] else "No",
            "Yes" if metrics_for_output["cache_retrieval"] else "No",
            "Yes" if metrics_for_output["cache_reranking"] else "No",
            metrics_for_output.get("avg_time_translation", 0),
            metrics_for_output.get("avg_time_span", 0),
            metrics_for_output.get("avg_time_ner", 0),
            metrics_for_output.get("avg_time_normalization", 0),
            metrics_for_output.get("avg_time_el_retrieval", 0),
            metrics_for_output.get("avg_time_el_reranking", 0),
            metrics_for_output["precision"],
            metrics_for_output["recall"],
            metrics_for_output["f1"],
            metrics_for_output["num_records"]
        ])
    
    # Print average metrics
    print("\n=== Evaluation Results ===")
    print("\nConfiguration Parameters Actually Used:")
    print(f"  Translation enabled: {not args.skip_translation}")
    print(f"  Span Model: {config_params['component_params']['span_identifier']['actual_model']}")
    print(f"  NER Model: {config_params['component_params']['ner']['model_path']}")
    print(f"  Entity Linkers: {args.entity_linkers}")
    print(f"  Rerank: {rerank}")
    print(f"  Linker Threshold: {args.linker_threshold}")
    print(f"  Final Threshold: {args.final_threshold}")
    print(f"  Device: {config_params['device']}")
    
    print("\nCache Settings:")
    print(f"  Translation Cache: {'Enabled' if args.use_cache_translation else 'Disabled'}")
    print(f"  Normalization Cache: {'Enabled' if args.use_cache_norm else 'Disabled'}")
    print(f"  Retrieval Cache: {'Enabled' if args.use_cache_retrieval else 'Disabled'}")
    print(f"  Reranking Cache: {'Enabled' if args.use_cache_reranking else 'Disabled'}")
    
    print("\nPerformance Metrics:")
    print(f"  Average Precision: {avg_metrics['precision']:.4f}")
    print(f"  Average Recall: {avg_metrics['recall']:.4f}")
    print(f"  Average F1 Score: {avg_metrics['f1']:.4f}")
    print(f"  Average Processing Time per Affiliation: {avg_metrics['avg_processing_time']:.4f} seconds")
    
    print("\nComponent Timings (seconds per affiliation):")
    for component, avg_time in avg_metrics['component_times'].items():
        print(f"  {component.capitalize()}: {avg_time:.4f}")
    
    print(f"\nTotal samples evaluated: {len(gold_data)}")
    
    print(f"\nDetailed configuration saved to: {config_output}")
    print(f"Summary metrics saved to: {summary_output}")
    print("===================================\n")

if __name__ == "__main__":
    main()
