python src/main.py \
    --dataset_directory /datasets/umgloc_dataset \
	--output_directory /results \
	--model_type 'Q_LSTM_GAN' \
    --use_map \
    --dataset 'kaust' \
	train \
	--train_list /lists/our/kaust_train_list.txt \
	--val_list /lists/our/kaust_val_list.txt 
