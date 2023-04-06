#!/bin/bash

export PYTHONPATH=$CV_LIB_PATH:./

export CUDA_VISIBLE_DEVICES=0

# python scripts/save_backbone_jit.py \
#     --cfg_fp config/cifar_10/ingredient/deit_tiny-l9-M_128.yaml \
#     --ckpt_fp run/cifar_10/vanilla/deit_tiny/ckpt/best.pth \
#     --vocabulary_fp run/cifar_10/ingredient/deit_tiny-l9-M_128-1M/cluster_128_from_1000000.pth \
#     --save_path run/cifar_10/ingredient/deit_tiny-l9-M_128-1M/jit \
#     --num_classes 10 \
#     --img_size 224 \
#     --img_channels 3 \
#     --extract_layer module.transformer.layers.9.attention.attn_raw_identity

# python scripts/save_backbone_jit.py \
#     --cfg_fp config/cifar_100/ingredient/deit_tiny-l9-M_1024.yaml \
#     --ckpt_fp run/cifar_100/vanilla/deit_tiny/ckpt/best.pth \
#     --vocabulary_fp run/cifar_100/ingredient/deit_tiny-l9-M_1024-all/cluster_1024_from_9800000.pth \
#     --save_path run/cifar_100/ingredient/deit_tiny-l9-M_1024-all/jit \
#     --num_classes 100 \
#     --img_size 224 \
#     --img_channels 3 \
#     --extract_layer module.transformer.layers.9.attention.attn_raw_identity

# python scripts/save_backbone_jit.py \
#     --cfg_fp config/caltech_101/ingredient/deit_tiny-l9-M_1024.yaml \
#     --ckpt_fp run/caltech_101/vanilla/deit_tiny/ckpt/best.pth \
#     --vocabulary_fp run/caltech_101/ingredient/deit_tiny-l9-M_1024-all/cluster_1024_from_1443540.pth \
#     --save_path run/caltech_101/ingredient/deit_tiny-l9-M_1024-all/jit \
#     --num_classes 101 \
#     --img_size 224 \
#     --img_channels 3 \
#     --extract_layer module.transformer.layers.9.attention.attn_raw_identity

# python scripts/save_backbone_jit.py \
#     --cfg_fp config/caltech_101/ingredient/deit_small-l9-M_1024.yaml \
#     --ckpt_fp run/caltech_101/vanilla/deit_small/ckpt/best.pth \
#     --vocabulary_fp run/caltech_101/ingredient/deit_small-l9-M_1024-all/cluster_1024_from_1443540.pth \
#     --save_path run/caltech_101/ingredient/deit_small-l9-M_1024-all/jit \
#     --num_classes 101 \
#     --img_size 224 \
#     --img_channels 3 \
#     --extract_layer module.transformer.layers.9.attention.attn_raw_identity

# python scripts/save_backbone_jit.py \
#     --cfg_fp config/imagenet/ingredient/deit_small-l9-M_8000.yaml \
#     --ckpt_fp weights/deit_small_patch16_224.pth\
#     --vocabulary_fp run/imagenet/ingredient/deit_small-l9-M_8000-50M/cluster_8000_from_50000000.pth \
#     --save_path run/imagenet/ingredient/deit_small-l9-M_8000-50M/jit \
#     --num_classes 1000 \
#     --img_size 224 \
#     --img_channels 3 \
#     --extract_layer module.transformer.layers.9.attention.attn_raw_identity
