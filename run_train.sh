python src_1/main.py \
    --dataset_directory /datasets/umgloc_dataset \
	--output_directory /results \
	--use_scheduler \
	--model_type 'Q_LSTM_GAN' \
    --use_map \
    --perturb \
    --noise_level 0.0 \
    --pi 95 \
    --window_size 120 \
    --seq_len 120 \
    --batch_size 16 \
    --step_size 30 \
    --dataset 'kaust' \
	train \
	--epochs 200 \
    --lr 0.0003 \
	--train_list /lists/our/kaust_train_list.txt \
	--val_list /lists/our/kaust_val_list.txt 