#!/bin/bash

# Script to run AffilGood evaluations with different configurations
# and collect the results into a single table

# Set default directories
INPUT_DIR="benchmark_datasets"
OUTPUT_DIR="benchmark_datasets/evaluation_results"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --input-dir|-i)
      INPUT_DIR="$2"
      shift 2
      ;;
    --output-dir|-o)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: $0 [OPTIONS]"
      echo "Options:"
      echo "  -i, --input-dir DIR    Directory containing benchmark CSV/TSV files (default: current dir)"
      echo "  -o, --output-dir DIR   Directory to store results (default: evaluation_results)"
      echo "  -h, --help             Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Run '$0 --help' for usage information"
      exit 1
      ;;
  esac
done

echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Datasets to evaluate
DATASETS=(
  #"CORDIS.csv"
  #"ETER_English.csv"
  #"ETER_multilingual.csv"
  #"French_affiliations.csv"
  "Mixed_affiliations_S2AFF.csv"
  "Multilingual_affiliations.csv"
  "Non-related_multi-organizations.csv"
  #"OpenAlex.tsv"
)

# Function to extract the dataset name without extension
function get_dataset_name() {
  local file="$1"
  echo "${file%.*}"
}

# Function to run a single evaluation with parameters
function run_evaluation() {
  local dataset="$1"
  local linkers="$2"
  local translation="$3"
  local cache_translation="$4"
  local cache_norm="$5"
  local cache_retrieval="$6"
  local cache_reranking="$7"
  
  local dataset_name=$(get_dataset_name "$dataset")
  local translation_flag=""
  local cache_translation_flag=""
  local cache_norm_flag=""
  local cache_retrieval_flag=""
  local cache_reranking_flag=""
  
  # Set flags based on boolean parameters
  if [ "$translation" = false ]; then
    translation_flag="--skip-translation"
  fi
  
  if [ "$cache_translation" = true ]; then
    cache_translation_flag="--use-cache-translation"
  fi
  
  if [ "$cache_norm" = true ]; then
    cache_norm_flag="--use-cache-norm"
  fi
  
  if [ "$cache_retrieval" = true ]; then
    cache_retrieval_flag="--use-cache-retrieval"
  fi
  
  if [ "$cache_reranking" = true ]; then
    cache_reranking_flag="--use-cache-reranking"
  fi
  
  # Format output filename
  local linkers_str="${linkers//,/_}"
  local trans_str=$([ "$translation" = true ] && echo "trans" || echo "notrans")
  local cache_str=""
  
  if [ "$cache_translation" = true ] || [ "$cache_norm" = true ] || [ "$cache_retrieval" = true ] || [ "$cache_reranking" = true ]; then
    cache_str="_cache"
  fi
  
  local output_file="${OUTPUT_DIR}/${dataset_name}_${linkers_str}_${trans_str}${cache_str}.tsv"
  
  echo "Running evaluation for $dataset with linkers: $linkers, translation: $translation"
  echo "Cache settings - Translation: $cache_translation, Norm: $cache_norm, Retrieval: $cache_retrieval, Reranking: $cache_reranking"
  
  # Build command arguments
  command=(
    enhanced_eval_affilgood.py
    --input "${INPUT_DIR}/${dataset}"
    --output "$output_file"
    --entity-linkers "$linkers"
    --dataset-name "$dataset_name"
  )
  
  # Add optional flags
  [ -n "$translation_flag" ] && command+=("$translation_flag")
  [ -n "$cache_translation_flag" ] && command+=("$cache_translation_flag")
  [ -n "$cache_norm_flag" ] && command+=("$cache_norm_flag")
  [ -n "$cache_retrieval_flag" ] && command+=("$cache_retrieval_flag")
  [ -n "$cache_reranking_flag" ] && command+=("$cache_reranking_flag")
  
  echo "Running command: python ${command[*]}"
  
  # Run the evaluation script
  python "${command[@]}"

}

# Create a file to collect all summary results
SUMMARY_FILE="${OUTPUT_DIR}/all_results_summary.tsv"

# Initialize summary file with header
echo -e "Dataset\tMethod\tTranslation\tCache_Translation\tCache_Norm\tCache_Retrieval\tCache_Reranking\tTime_Translation\tTime_Span\tTime_NER\tTime_Normalization\tTime_EL_Retrieval\tTime_EL_Reranking\tPrecision\tRecall\tF1_Score\tNum_Records" > "$SUMMARY_FILE"

# Check if input files exist
echo "Checking input files..."
for dataset in "${DATASETS[@]}"; do
  if [ ! -f "${INPUT_DIR}/${dataset}" ]; then
    echo "WARNING: Input file not found: ${INPUT_DIR}/${dataset}"
    # Remove from array
    DATASETS=("${DATASETS[@]/$dataset}")
  else
    echo "Found: ${INPUT_DIR}/${dataset}"
  fi
done

# Run evaluations for each dataset with different configurations
for dataset in "${DATASETS[@]}"; do
  # 1. With translation, different linkers, no cache
  # 1.a - Dense linker only
  run_evaluation "$dataset" "Dense" true false false false false
  
  # 1.b - Whoosh linker only
  run_evaluation "$dataset" "Whoosh" true false false false false
  
  # 1.c - Both Dense and Whoosh linkers
  run_evaluation "$dataset" "Dense,Whoosh" true false false false false
  
  # 2. Without translation, different linkers, no cache
  # 2.a - Dense linker only
  run_evaluation "$dataset" "Dense" false false false false false
  
  # 2.b - Whoosh linker only
  run_evaluation "$dataset" "Whoosh" false false false false false
  
  # 2.c - Both Dense and Whoosh linkers
  run_evaluation "$dataset" "Dense,Whoosh" false false false false false
  
  # 3. With translation, caching enabled (one configuration as example)
  run_evaluation "$dataset" "Dense" true true true true true
  
  # 4. Without translation, caching enabled (one configuration as example)
  run_evaluation "$dataset" "Dense" false false true true true
  
  # Collect summary files for this dataset
  for summary in "${OUTPUT_DIR}/${dataset%.*}"*_summary.tsv; do
    if [ -f "$summary" ]; then
      tail -n +2 "$summary" >> "$SUMMARY_FILE"
    fi
  done
done

echo "All evaluations completed. Summary results saved to $SUMMARY_FILE"

# Create formatted HTML table from summary results
HTML_OUTPUT="${OUTPUT_DIR}/results_table.html"

# Convert TSV to HTML table
echo "<html><head><title>AffilGood Evaluation Results</title>" > "$HTML_OUTPUT"
echo "<style>
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
tr:nth-child(even) { background-color: #f2f2f2; }
th { background-color: #4CAF50; color: white; }
</style>" >> "$HTML_OUTPUT"
echo "</head><body>" >> "$HTML_OUTPUT"
echo "<h1>AffilGood Evaluation Results</h1>" >> "$HTML_OUTPUT"
echo "<table>" >> "$HTML_OUTPUT"

# Read header
IFS=$'\t' read -a headers < "$SUMMARY_FILE"

# Write table header
echo "<tr>" >> "$HTML_OUTPUT"
for header in "${headers[@]}"; do
  echo "<th>$header</th>" >> "$HTML_OUTPUT"
done
echo "</tr>" >> "$HTML_OUTPUT"

# Write table data (skip header)
tail -n +2 "$SUMMARY_FILE" | while IFS=$'\t' read -a fields; do
  echo "<tr>" >> "$HTML_OUTPUT"
  for field in "${fields[@]}"; do
    echo "<td>$field</td>" >> "$HTML_OUTPUT"
  done
  echo "</tr>" >> "$HTML_OUTPUT"
done

echo "</table></body></html>" >> "$HTML_OUTPUT"

echo "HTML table created at $HTML_OUTPUT"

# Also create a compact CSV for easy import into spreadsheet software
CSV_OUTPUT="${OUTPUT_DIR}/results_table.csv"
cat "$SUMMARY_FILE" | tr '\t' ',' > "$CSV_OUTPUT"
echo "CSV table created at $CSV_OUTPUT"
