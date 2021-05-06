#!/bin/sh

export DATA_DIR=datasets/298954459172700181_muffins

#source style-venv/bin/activate

BASE_DIR=style_paraphrase

#python -m torch.distributed.launch --nproc_per_node=1 $BASE_DIR/run_lm_finetuning.py \
python $BASE_DIR/run_lm_finetuning.py \
    --output_dir=$BASE_DIR/saved_models/298954459172700181_muffins \
    --model_type=gpt2 \
    --model_name_or_path=gpt2-large \
    --data_dir=$DATA_DIR \
    --do_train \
    --save_steps 1000 \
    --logging_steps 20 \
    --save_total_limit -1 \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 1 \
    --per_gpu_train_batch_size 4 \
    --job_id 298954459172700181_muffins \
    --learning_rate 5e-5 \
    --prefix_input_type paraphrase_250 \
    --global_dense_feature_list none \
    --specific_style_train -1 \
    --optimizer adam
    #--evaluate_during_training \
    #--local_rank
