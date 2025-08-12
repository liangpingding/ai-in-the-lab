python run_classification.py \
    --train_file Data/AI_relevance/AI_relevance_train.csv \
    --test_file Data/AI_relevance/non_AI_sampled_falsenegativetest.csv \
    --shuffle_train_dataset \
    --text_column_names title_abs \
    --label_column_name AI_relevance_GPT \
    --model_name_or_path allenai/scibert_scivocab_uncased \
    --do_train \
    --do_predict \
    --overwrite_output_dir \
    --max_seq_length 512 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 256 \
    --do_regression False \
    --learning_rate 1e-5 \
    --num_train_epochs 5 \
    --output_dir Output/AI_relevance \
    --logging_steps 1000 


    
        