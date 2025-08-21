#!/bin/bash

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=1

# Define pruning methods and their corresponding pruning ratios
declare -A pruning_ratios=(
    ["random"]=0.20
    ["l1"]=0.05
    ["l2"]=0.05
    ["taylor"]=0.20
)

# List of base model paths
base_models=(
    "TrainedCheckpoint/WizardMath-7B-V1.0/fp_vector/if_chat_fp_from_llama2_1.0"
    "TrainedCheckpoint/WizardMath-7B-V1.0/full/if_chat_fp"
)

# Iterate over each model
for base_model in "${base_models[@]}"; do
    # Get the last folder name from the base model path
    model_name=$(basename "$base_model")
    prune_rate_zero_done=true

    # Iterate over each pruning method
    for pruner_type in "${!pruning_ratios[@]}"; do
        # Get the pruning ratio for the current pruning method
        pruning_ratio=${pruning_ratios[$pruner_type]}

        # Then execute the non-zero pruning case
        save_ckpt_log_name="${base_model}/prune/${pruner_type}-${pruning_ratio}"
        if [ ! -d "$save_ckpt_log_name" ]; then
            echo "Save directory $save_ckpt_log_name does not exist, creating it..."
            mkdir -p "$save_ckpt_log_name"
        fi
        echo "Starting to process model: $model_name, pruner_type = $pruner_type, pruning_ratio = $pruning_ratio"
        python /work/xzh/LLM-Pruner/hf_prune.py --pruning_ratio "$pruning_ratio" \
            --block_wise \
            --block_mlp_layer_start 4 --block_mlp_layer_end 30 \
            --block_attention_layer_start 4 --block_attention_layer_end 30 \
            --pruner_type "$pruner_type" \
            --device cpu --eval_device cuda:0 \
            --base_model "$base_model" \
            --save_ckpt_log_name "$save_ckpt_log_name" \
            --save_model
    done
done
