python src/main.py \
    --dataset_directory /datasets/umgloc_dataset \
	--output_directory \results \
     --use_map \
    --dataset 'kaust' \
	test \
	--test_list /lists/our/umgloc_test_list.txt \
	--lstm_path /models/ \
    --gan_path /models/
