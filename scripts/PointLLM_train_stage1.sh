# master_port=$((RANDOM % (65535 - 49152 + 1) + 49152))
# Get the filename without extension
filename=$(basename "$0" | cut -f 1 -d '.')

dir_path=/home/hongfa/construction_PointLLM
model_name_or_path=checkpoints/PointLLM_7B_v1.2
data_path=data/pc
anno_path=data/all_scene_anno_list.json
output_dir=outputs/construction_PointLLM_train_stage1/$filename
# point_backbone_ckpt=checkpoints/point_bert_v1.2.pt
point_backbone_ckpt=/home/hongfa/scene_ULIP/outputs/reproduce_pointbert_1kpts/checkpoint_best.pt

cd $dir_path

PYTHONPATH=$dir_path:$PYTHONPATH \
torchrun pointllm/train/train_mem.py \
    --model_name_or_path $model_name_or_path \
    --data_path $data_path \
    --anno_path $anno_path \
    --output_dir $output_dir \
    --version v1 \
    --model_max_length 2048 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --pointnum 1000000 \
    --optim adamw_bnb_8bit \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 2400 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bf16 True \
    --evaluation_strategy "no" \
    --fix_llm True \
    --fix_pointnet False \
    --gradient_checkpointing True \
    --report_to wandb \
    --run_name $filename \
    --point_backbone_ckpt $point_backbone_ckpt \
    --use_color True\
    --split_train_val True