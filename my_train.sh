python tools/train.py configs/psgformer/psgformer_r101_psg.py

# PYTHONPATH='.':$PYTHONPATH \
# python -m torch.distributed.launch \
# --nproc_per_node=8 --master_port=29501 \
#   tools/train.py \
#   configs/psgformer/psgformer_r50_psg.py \
#   --gpus 1 \
#   --launcher pytorch