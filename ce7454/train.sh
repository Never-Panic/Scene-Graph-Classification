CUDA_VISIBLE_DEVICES=$1 python main.py \
--model_name 'cocoop_pos_weight_lr2e-3_bs8_epoch10' \
--lr 2e-3 \
--lr_factor 1e-4 \
--batch_size 8 \
--epoch 10 \