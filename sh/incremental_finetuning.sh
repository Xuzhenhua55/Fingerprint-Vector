#### if-transferred-fingerprinting
for sample_size in 3000 10000 15000
do
    CUDA_VISIBLE_DEVICES=4 llamafactory-cli train \
        --stage sft \
        --do_train True \
        --model_name_or_path TrainedCheckpoint/WizardMath-7B-V1.0/fp_vector/if_chat_fp_from_llama2_1.0 \
        --preprocessing_num_workers 16 \
        --finetuning_type lora \
        --template llama2 \
        --flash_attn auto \
        --dataset_dir data \
        --dataset dolly_en_15k \
        --cutoff_len 1024 \
        --learning_rate 5e-05 \
        --num_train_epochs 2.0 \
        --max_samples ${sample_size} \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 8 \
        --lr_scheduler_type cosine \
        --max_grad_norm 1.0 \
        --logging_steps 1 \
        --save_steps 5000 \
        --warmup_steps 0 \
        --packing False \
        --enable_thinking True \
        --report_to none \
        --output_dir TrainedCheckpoint/WizardMath-7B-V1.0/fp_vector/if_chat_fp_from_llama2_1.0/dolly_en_15k/${sample_size} \
        --bf16 True \
        --plot_loss True \
        --trust_remote_code True \
        --ddp_timeout 180000000 \
        --include_num_input_tokens_seen True \
        --optim adamw_torch \
        --lora_rank 8 \
        --lora_alpha 16 \
        --lora_dropout 0 \
        --lora_target all 
done


for sample_size in 3000 10000
do
    CUDA_VISIBLE_DEVICES=4 llamafactory-cli train \
        --stage sft \
        --do_train True \
        --model_name_or_path TrainedCheckpoint/WizardMath-7B-V1.0/fp_vector/if_chat_fp_from_llama2_1.0 \
        --preprocessing_num_workers 16 \
        --finetuning_type lora \
        --template llama2 \
        --flash_attn auto \
        --dataset_dir data \
        --dataset alpaca_data_52k \
        --cutoff_len 1024 \
        --learning_rate 5e-05 \
        --num_train_epochs 2.0 \
        --max_samples ${sample_size} \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 8 \
        --lr_scheduler_type cosine \
        --max_grad_norm 1.0 \
        --logging_steps 1 \
        --save_steps 5000 \
        --warmup_steps 0 \
        --packing False \
        --enable_thinking True \
        --report_to none \
        --output_dir TrainedCheckpoint/WizardMath-7B-V1.0/fp_vector/if_chat_fp_from_llama2_1.0/alpaca_data_52k/${sample_size} \
        --bf16 True \
        --plot_loss True \
        --trust_remote_code True \
        --ddp_timeout 180000000 \
        --include_num_input_tokens_seen True \
        --optim adamw_torch \
        --lora_rank 8 \
        --lora_alpha 16 \
        --lora_dropout 0 \
        --lora_target all 
done


for sample_size in 3000 6000
do
    CUDA_VISIBLE_DEVICES=4 llamafactory-cli train \
        --stage sft \
        --do_train True \
        --model_name_or_path TrainedCheckpoint/WizardMath-7B-V1.0/fp_vector/if_chat_fp_from_llama2_1.0 \
        --preprocessing_num_workers 16 \
        --finetuning_type lora \
        --template llama2 \
        --flash_attn auto \
        --dataset_dir data \
        --dataset sharegpt_gpt4_6k \
        --cutoff_len 1024 \
        --learning_rate 5e-05 \
        --num_train_epochs 2.0 \
        --max_samples ${sample_size} \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 8 \
        --lr_scheduler_type cosine \
        --max_grad_norm 1.0 \
        --logging_steps 1 \
        --save_steps 5000 \
        --warmup_steps 0 \
        --packing False \
        --enable_thinking True \
        --report_to none \
        --output_dir TrainedCheckpoint/WizardMath-7B-V1.0/fp_vector/if_chat_fp_from_llama2_1.0/sharegpt_gpt4_6k/${sample_size} \
        --bf16 True \
        --plot_loss True \
        --trust_remote_code True \
        --ddp_timeout 180000000 \
        --include_num_input_tokens_seen True \
        --optim adamw_torch \
        --lora_rank 8 \
        --lora_alpha 16 \
        --lora_dropout 0 \
        --lora_target all 
done
#### if-direct-fingerprinting

for sample_size in 3000 10000 15000
do
    CUDA_VISIBLE_DEVICES=4 llamafactory-cli train \
        --stage sft \
        --do_train True \
        --model_name_or_path TrainedCheckpoint/WizardMath-7B-V1.0/full/if_chat_fp \
        --preprocessing_num_workers 16 \
        --finetuning_type lora \
        --template llama2 \
        --flash_attn auto \
        --dataset_dir data \
        --dataset dolly_en_15k \
        --cutoff_len 1024 \
        --learning_rate 5e-05 \
        --num_train_epochs 2.0 \
        --max_samples ${sample_size} \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 8 \
        --lr_scheduler_type cosine \
        --max_grad_norm 1.0 \
        --logging_steps 1 \
        --save_steps 5000 \
        --warmup_steps 0 \
        --packing False \
        --enable_thinking True \
        --report_to none \
        --output_dir TrainedCheckpoint/WizardMath-7B-V1.0/full/if_chat_fp/dolly_en_15k/${sample_size} \
        --bf16 True \
        --plot_loss True \
        --trust_remote_code True \
        --ddp_timeout 180000000 \
        --include_num_input_tokens_seen True \
        --optim adamw_torch \
        --lora_rank 8 \
        --lora_alpha 16 \
        --lora_dropout 0 \
        --lora_target all 
done


for sample_size in 3000 10000 52000
do
    CUDA_VISIBLE_DEVICES=4 llamafactory-cli train \
        --stage sft \
        --do_train True \
        --model_name_or_path TrainedCheckpoint/WizardMath-7B-V1.0/full/if_chat_fp \
        --preprocessing_num_workers 16 \
        --finetuning_type lora \
        --template llama2 \
        --flash_attn auto \
        --dataset_dir data \
        --dataset alpaca_data_52k \
        --cutoff_len 1024 \
        --learning_rate 5e-05 \
        --num_train_epochs 2.0 \
        --max_samples ${sample_size} \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 8 \
        --lr_scheduler_type cosine \
        --max_grad_norm 1.0 \
        --logging_steps 1 \
        --save_steps 5000 \
        --warmup_steps 0 \
        --packing False \
        --enable_thinking True \
        --report_to none \
        --output_dir TrainedCheckpoint/WizardMath-7B-V1.0/full/if_chat_fp/alpaca_data_52k/${sample_size} \
        --bf16 True \
        --plot_loss True \
        --trust_remote_code True \
        --ddp_timeout 180000000 \
        --include_num_input_tokens_seen True \
        --optim adamw_torch \
        --lora_rank 8 \
        --lora_alpha 16 \
        --lora_dropout 0 \
        --lora_target all 
done


for sample_size in 3000 6000
do
    CUDA_VISIBLE_DEVICES=4 llamafactory-cli train \
        --stage sft \
        --do_train True \
        --model_name_or_path TrainedCheckpoint/WizardMath-7B-V1.0/full/if_chat_fp \
        --preprocessing_num_workers 16 \
        --finetuning_type lora \
        --template llama2 \
        --flash_attn auto \
        --dataset_dir data \
        --dataset sharegpt_gpt4_6k \
        --cutoff_len 1024 \
        --learning_rate 5e-05 \
        --num_train_epochs 2.0 \
        --max_samples ${sample_size} \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 8 \
        --lr_scheduler_type cosine \
        --max_grad_norm 1.0 \
        --logging_steps 1 \
        --save_steps 5000 \
        --warmup_steps 0 \
        --packing False \
        --enable_thinking True \
        --report_to none \
        --output_dir TrainedCheckpoint/WizardMath-7B-V1.0/full/if_chat_fp/sharegpt_gpt4_6k/${sample_size} \
        --bf16 True \
        --plot_loss True \
        --trust_remote_code True \
        --ddp_timeout 180000000 \
        --include_num_input_tokens_seen True \
        --optim adamw_torch \
        --lora_rank 8 \
        --lora_alpha 16 \
        --lora_dropout 0 \
        --lora_target all 
done