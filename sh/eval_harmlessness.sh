#!/bin/bash
export HF_ENDPOINT="https://hf-mirror.com"

device="cuda:2"
echo "Using device: $device"

# Define task list
tasks=(
    "anli_r1" "anli_r2" "anli_r3"
    "openbookqa"
    "logiqa" 
    "boolq"  "rte" "wic" "wsc" "copa"
)

# Define the list of models to test
models=(
    "TrainedCheckpoint/WizardMath-7B-V1.0/fp_vector/if_chat_fp_from_llama2_1.0"
    "TrainedCheckpoint/WizardMath-7B-V1.0/full/if_chat_fp"
)

# Check if jq is installed
if ! command -v jq &>/dev/null; then
    echo "Error: jq is not installed, unable to parse JSON files" | tee -a "global_error.log"
    echo "Please install jq: sudo apt install jq" | tee -a "global_error.log"
    exit 1
fi

# Iterate over each model
for model_path in "${models[@]}"; do
    # Extract model name from model path (e.g., falcon-7b)
    model_name=$(echo "$model_path" | sed 's#/#__#g')

    # Dynamically generate log file names
    success_log="${model_path}/${model_name}_task_results.log"
    error_log="${model_path}/${model_name}_error.log"

    > "$success_log"  # Clear success log file
    > "$error_log" # Clear error log file

    # Create output directory
    output_dir="${model_path}/general_ability/0-shot"
    mkdir -p "$output_dir"

    echo "======================================================"
    echo "Testing model: $model_path"
    echo "Output directory: $output_dir"
    echo "Success log file: $success_log"
    echo "Error log file: $error_log"
    echo "======================================================"

    # Iterate over each task
    for task in "${tasks[@]}"; do
        task_output_dir="${output_dir}/${task}"

        # Skip if directory already exists
        if [[ -d "$task_output_dir" ]]; then
            echo "------------------------------------------------------"
            echo "Skipping task: $task (directory already exists: $task_output_dir)"
            echo "------------------------------------------------------"
            sleep 3 # Pause for 3 seconds
            continue
        fi

        echo "------------------------------------------------------"
        echo "Testing task: $task"
        echo "------------------------------------------------------"

        # Record task start time
        start_time=$(date +%s)

        # Run evaluation command and display output in real-time
        if ! lm_eval --model hf \
            --model_args "pretrained=$model_path,dtype=float16,trust_remote_code=True" \
            --tasks "$task" \
            --device "$device" \
            --batch_size 1 \
            --output_path "$task_output_dir" 2>&1 | tee -a "$error_log"; then
            # If command fails, record task name and error information
            echo "Task failed: $task" >>"$error_log"
            echo "Error information:" >>"$error_log"
            cat "$error_log" | tail -n 10 >>"$error_log" # Record the last 10 lines of the error log
            echo "----------------------------------------" >>"$error_log"
            echo "Task $task failed, recorded in $error_log"
        else
            # If task succeeds, record runtime
            end_time=$(date +%s)
            runtime=$((end_time - start_time))

            # Parse the generated JSON file
            result_dir="${task_output_dir}/${model_name}"

            # Find JSON file (assuming only one JSON file in the directory)
            json_file=$(find "$result_dir" -name "*.json" -type f | head -n 1)

            if [[ -f "$json_file" ]]; then
                # Extract results field and record to log file
                if results=$(jq -r '.results' "$json_file" 2>>"$error_log"); then
                    echo "Task succeeded: $task" >>"$success_log"
                    echo "Runtime: ${runtime} seconds" >>"$success_log"
                    echo "Results field:" >>"$success_log"
                    echo "$results" >>"$success_log"
                    echo "----------------------------------------" >>"$success_log"
                    echo "Task $task succeeded, results recorded in $success_log"
                else
                    echo "Task $task succeeded, but unable to parse results field" >>"$error_log"
                    echo "JSON file path: $json_file" >>"$error_log"
                fi
            else
                echo "Task $task succeeded, but no result file found" >>"$error_log"
            fi
        fi

        echo -e "\n"
        sleep 1 # Pause for 3 seconds
    done

    echo -e "\n\n"
done

# Print global error log (if any)
if [[ -s "global_error.log" ]]; then
    echo "======================================================"
    echo "Global error log:"
    cat "global_error.log"
    echo "======================================================"
fi

# Print log summary for each model
for model_path in "${models[@]}"; do
    model_name=$(basename "$model_path")
    success_log="${model_name}_task_results.log"
    error_log="${model_name}_error.log"

    if [[ -s "$error_log" ]]; then
        echo "======================================================"
        echo "Model $model_name failed the following tasks:"
        cat "$error_log"
        echo "======================================================"
    else
        echo "Model $model_name successfully completed all tasks!"
    fi

    if [[ -s "$success_log" ]]; then
        echo "======================================================"
        echo "Model $model_name task results:"
        cat "$success_log"
        echo "======================================================"
    fi
done
