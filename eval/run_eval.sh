#!/bin/bash

# Configuration
declare -a MODEL=("Qwen2.5VL3b" "Qwen2.5VL7b" "Janus" "Phi4")
TYPE="mf"
GPU_ID="0"
declare -a RS_TYPES=("ar" "cs" "mp" "em")

# Create logs directory if it doesn't exist
mkdir -p logs

# Run evaluation for each rs type
for model in "${MODEL[@]}"; do
    for rs in "${RS_TYPES[@]}"; do
        echo "Running evaluation for model=$model, type=$TYPE, rs=$rs"
        echo "----------------------------------------"
    
        # Run the evaluation
        python eval.py --model "$model" --type "$TYPE" --rs "$rs" --gpu_id "$GPU_ID"
        
        # Check if the evaluation was successful
        if [ $? -eq 0 ]; then
            echo "Evaluation completed successfully for rs: $rs"
        else
            echo "Error: Evaluation failed for rs: $rs"
        fi
        
        echo "----------------------------------------"
    done
done

echo "All evaluations completed!" 