#!/usr/bin/env bash
export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=0
for dataset in 'stackoverflow' 'banking' 'clinc'
do
	for seed in 0
	do
		python loop.py \
			--data_dir data \
			--dataset $dataset \
			--known_cls_ratio 0.75 \
			--labeled_ratio 0.1 \
			--seed $seed \
			--lr '1e-5' \
			--save_results_path 'outputs' \
			--view_strategy 'rtr' \
			--update_per_epoch 5 \
			--save_premodel \
			--save_model
	done
done