python src_1/main.py \
    --dataset_directory /datasets/umgloc_dataset \
	--output_directory \results \
     --use_map \
    --perturb \
    --noise_level 0.0 \
    --pi 95 \
    --window_size 120 \
    --seq_len 120 \
    --batch_size 16 \
    --step_size 30 \
    --dataset 'kaust' \
	test \
	--test_list /lists/our/all_umgloc.txt \
	--lstm_path /models/ \
    --gan_path /models/