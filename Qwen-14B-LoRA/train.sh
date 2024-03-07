gpu_idx=$1
run_name=$2
base_name=$3

deepspeed -i "localhost:${gpu_idx}" train_lora.py \
    --run_name ${run_name} \
    --base_model ${base_name} \
    --dataset data/fingpt-sentiment-train \
    --max_length 512 \
    --batch_size 4 \
    --learning_rate 3e-4 \
    --num_epochs 8 \
    --warmup_ratio 0.01 \
    --scheduler "cosine" \
    --report_to "none"