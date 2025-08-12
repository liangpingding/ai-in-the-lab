
python run_classification.py \
    --train_file Data/AI_mode/AI_mode_train.csv \
    --test_file Data/AI_mode/AI_mode_predict.csv \
    --shuffle_train_dataset \
    --text_column_names title_abs \
    --label_column_name GPT_AI_mode \
    --model_name_or_path allenai/scibert_scivocab_uncased \
    --do_train \
    --do_predict \
    --max_seq_length 512 \
    --overwrite_output_dir \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 256 \
    --do_regression False \
    --learning_rate 1e-5 \
    --num_train_epochs 20 \
    --output_dir Output/AI_mode \
    --logging_steps 100 
    