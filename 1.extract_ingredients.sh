#!/bin/bash

CV_LIB_PATH=/c22940/zhf/zhfeing_cygpu1/project/cv-lib-PyTorch
export PYTHONPATH=$CV_LIB_PATH:./

export CUDA_VISIBLE_DEVICES=0

# python scripts/extract_ingredients.py \
#     --cfg_fp config/cifar_10/ingredient/deit_tiny-l9-M_128.yaml \
#     --save_path run/cifar_10/ingredient/deit_tiny-l9-M_128-1M \
#     --kmeans_method minibatch \
#     --max_features 1000000 \
#     --batch_size 64 \
#     --num_workers 8

# python scripts/extract_ingredients.py \
#     --cfg_fp config/cifar_100/ingredient/deit_tiny-l9-M_1024.yaml \
#     --save_path run/cifar_100/ingredient/deit_tiny-l9-M_1024-all \
#     --kmeans_method minibatch \
#     --max_features 1000000000 \
#     --batch_size 64 \
#     --num_workers 8

# python scripts/extract_ingredients.py \
#     --cfg_fp config/caltech_101/ingredient/deit_tiny-l9-M_1024.yaml \
#     --save_path run/caltech_101/ingredient/deit_tiny-l9-M_1024-all \
#     --kmeans_method minibatch \
#     --max_features 1000000000 \
#     --batch_size 64 \
#     --num_workers 8

# python scripts/extract_ingredients.py \
#     --cfg_fp config/caltech_101/ingredient/deit_small-l9-M_1024.yaml \
#     --save_path run/caltech_101/ingredient/deit_small-l9-M_1024-all \
#     --kmeans_method minibatch \
#     --max_features 1000000000 \
#     --batch_size 64 \
#     --num_workers 8

# python scripts/extract_ingredients.py \
#     --cfg_fp config/imagenet/ingredient/deit_small-l9-M_8000.yaml\
#     --save_path run/imagenet/ingredient/deit_small-l9-M_8000-50M \
#     --kmeans_method minibatch \
#     --max_features 50000000 \
#     --batch_size 64 \
#     --num_workers 8
