# training 
python ../run_classification.py \
    --train_file ../Data/AI_relevance/train.csv \
    --validation_file ../Data/AI_relevance/dev.csv \
    --test_file ../Data/AI_relevance/test.csv \
    --shuffle_train_dataset \
    --text_column_names title_abs \
    --label_column_name Category \
    --model_name_or_path allenai/scibert_scivocab_uncased \
    --max_seq_length 512 \
    --overwrite_output_dir \
    --per_device_train_batch_size 48 \
    --per_device_eval_batch_size 256 \
    --do_regression False \
    --learning_rate 1e-4 \
    --num_train_epochs 20 \
    --output_dir ../Output/AI_relevance \
    --logging_steps 100 \
    --do_train \
    --do_eval \
    --do_predict \
    --sweep_method None \
    --sweep_file sweep.json \
    --project_name AI_relevance_classification3 \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --save_strategy steps \
    --save_steps 200 \
    --load_best_model_at_end True \
    --metric_for_best_model f1 \
    --greater_is_better True \
    --save_total_limit 1 \
    --report_to wandb \
    --early_stopping_patience 10 



# prediction
python ../run_classification.py \
    --train_file ../Data/AI_relevance/train.csv \
    --validation_file ../Data/AI_relevance/dev.csv \
    --test_file ../Data/test.csv \
    --shuffle_train_dataset \
    --text_column_names title_abs \
    --label_column_name Category \
    --model_name_or_path ../Output/AI_relevance \
    --max_seq_length 512 \
    --overwrite_output_dir \
    --per_device_train_batch_size 48 \
    --per_device_eval_batch_size 256 \
    --do_regression False \
    --learning_rate 1e-4 \
    --num_train_epochs 20 \
    --output_dir ../Output/AI_relevance \
    --logging_steps 100 \
    --do_predict \
    --sweep_method None \
    --sweep_file sweep.json \
    --project_name AI_relevance_classification3 \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --save_strategy steps \
    --save_steps 200 \
    --load_best_model_at_end True \
    --metric_for_best_model f1 \
    --greater_is_better True \
    --save_total_limit 1 \
    --report_to wandb \
    --early_stopping_patience 10 \
    --nolabel_test 