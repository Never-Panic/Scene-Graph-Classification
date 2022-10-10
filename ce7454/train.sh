CUDA_VISIBLE_DEVICES=$1 python main.py \
--model_name 'coop_pos_weight_lr1e-2_bs32_epoch15' \
--lr 1e-2 \
--lr_factor 1e-4 \
--batch_size 32 \
--epoch 15 \