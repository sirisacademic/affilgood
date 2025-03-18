import argparse
import csv
import pandas as pd
import numpy as np
import re
import time
from tqdm import tqdm
from affilgood import AffilGood
from affilgood.entity_linking.whoosh_linker import WhooshLinker

DEFAULT_SPAN_MODEL = 'SIRIS-Lab/affilgood-span-multilingual'
DEFAULT_NER_MODEL = 'SIRIS-Lab/affilgood-NER-multilingual'
DEFAULT_ENTITY_LINKERS = 'S2AFF'

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

def evaluate_pipeline(affilgood, gold_data):
    """Evaluate the pipeline against gold annotations."""
    results = []
    all_metrics = {'precision': [], 'recall': [], 'f1': []}
    processing_times = []
    
    for _, row in tqdm(gold_data.iterrows(), total=len(gold_data), desc="Evaluating"):
        affiliation = row['raw_affiliation_string']
        gold_labels = row['label']
        
        # Extract gold ROR IDs
        gold_ids = extract_ror_ids(gold_labels)
        
        # Process through pipeline
        output = affilgood.process(affiliation)
        
        # Measure processing time
        start_time = time.time()
        output = affilgood.process(affiliation)
        end_time = time.time()
        elapsed_time = end_time - start_time
        processing_times.append(elapsed_time)
        
        # Extract predicted ROR IDs from all spans
        pred_ids = set()
        pred_labels = set()
        for item in output:
            ror_predictions = item.get('ror', [])
            pred_labels.update(ror_predictions)
            for ror_prediction in ror_predictions:
                pred_ids.update(extract_ror_ids(ror_prediction))
        
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
        results.append(result_row)
    
    # Calculate average metrics
    avg_metrics = {
        'precision': np.mean(all_metrics['precision']),
        'recall': np.mean(all_metrics['recall']),
        'f1': np.mean(all_metrics['f1']),
        'avg_processing_time': np.mean(processing_times)
    }
    
    return results, avg_metrics

def main():
    parser = argparse.ArgumentParser(description='Evaluate AffilGood pipeline on gold annotations.')
    parser.add_argument('--input', required=True, help='Path to input TSV file with gold annotations.')
    parser.add_argument('--output', required=True, help='Path to output evaluation file.')
    parser.add_argument('--span-model', default=DEFAULT_SPAN_MODEL, help='Span model. Defaults to noop.')
    parser.add_argument('--ner-model', default=DEFAULT_NER_MODEL, help='NER model path.')
    parser.add_argument('--entity-linkers', default=DEFAULT_ENTITY_LINKERS, 
                       help='Entity linker(s) to be used. Separate multiple linkers with a comma.')
    parser.add_argument('--device', default=None, help='Device for processing (cpu/cuda)')
    
    args = parser.parse_args()
    
    print("\n===== Evaluation Configuration =====")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("===================================\n")
    
    # Load gold annotations
    gold_data = pd.read_csv(args.input, sep='\t').fillna('')
    
    # Initialize AffilGood
    entity_linkers = [linker.strip() for linker in args.entity_linkers.split(',')]
    
    entity_linker_objects = []
    for entity_linker in entity_linkers:
        if entity_linker == 'Whoosh_NoRerank':
            entity_linker_objects.append(WhooshLinker(rerank=False, debug=False))
        else:
            entity_linker_objects.append(entity_linker)
    
    print(f"Initializing AffilGood pipeline...")
    affilgood = AffilGood(
        span_model_path=args.span_model,
        ner_model_path=args.ner_model,
        entity_linkers=entity_linker_objects,
        device=args.device
    )
    
    # Run evaluation
    print(f"Evaluating on {len(gold_data)} gold annotations...")
    results, avg_metrics = evaluate_pipeline(affilgood, gold_data)
    
    # Write results to output file
    print(f"Writing evaluation results to {args.output}")
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, sep='\t', index=False)
    
    # Print average metrics
    print("\nEvaluation Results:")
    print(f"Average Precision: {avg_metrics['precision']:.4f}")
    print(f"Average Recall: {avg_metrics['recall']:.4f}")
    print(f"Average F1 Score: {avg_metrics['f1']:.4f}")
    print(f"Average Processing Time per Affiliation: {avg_metrics['avg_processing_time']:.4f} seconds")
    print(f"Total samples evaluated: {len(gold_data)}")

if __name__ == "__main__":
    main()
