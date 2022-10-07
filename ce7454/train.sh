CUDA_VISIBLE_DEVICES=$1 python main.py \
--model_name 'prompt_lr1e-2_warmup100_factor1e-4_epoch15_ctx16' \
--lr 1e-2 \
--lr_factor 1e-4 \
--batch_size 32 \
--epoch 20 \