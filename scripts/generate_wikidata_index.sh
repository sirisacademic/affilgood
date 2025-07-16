#!/bin/bash

# Script to run WikiData index generation with PyTorch memory optimization
# Usage: ./run_wikidata_index.sh

echo "Starting WikiData Index Generation"
echo "=================================="

# Set PyTorch memory management for better GPU memory handling
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Optional: Set additional memory optimizations
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# Check if GPU is available
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Memory Status Before:"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
    echo ""
else
    echo "nvidia-smi not found - GPU status unavailable"
fi

# Print environment settings
echo "Environment Variables:"
echo "PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"
echo ""

# Run the Python script
echo "Running WikiData index generation..."
python generate_wikidata_index.py

# Capture exit code
EXIT_CODE=$?

# Show final GPU memory status
echo ""
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Memory Status After:"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
fi

# Report completion
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ WikiData index generation completed successfully!"
else
    echo ""
    echo "❌ WikiData index generation failed with exit code: $EXIT_CODE"
fi

exit $EXIT_CODE
