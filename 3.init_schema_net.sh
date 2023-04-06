#!/bin/bash

export PYTHONPATH=$CV_LIB_PATH:./

export CUDA_VISIBLE_DEVICES=0


# python scripts/init_schema_net.py \
#     --schema_net_cfg config/cifar_10/schema_net/deit_tiny-l9-M_128.yaml \
#     --save_fp run/cifar_10/schema_net/init_IR_atlas-deit_tiny-l9-M_128.pth \
#     --num_workers 8 \
#     --batch_size 64

# python scripts/init_schema_net.py \
#     --schema_net_cfg config/cifar_100/schema_net/deit_tiny-l9-M_1024.yaml \
#     --save_fp run/cifar_100/schema_net/init_IR_atlas-deit_tiny-l9-M_1024.pth \
#     --num_workers 8 \
#     --batch_size 64

# python scripts/init_schema_net.py \
#     --schema_net_cfg config/caltech_101/schema_net/deit_small-l9-M_1024.yaml \
#     --save_fp run/caltech_101/schema_net/init_IR_atlas-deit_small-l9-M_1024.pth \
#     --num_workers 8 \
#     --batch_size 64

# python scripts/init_schema_net.py \
#     --schema_net_cfg config/imagenet/schema_net/deit_small-l9-M_8000.yaml \
#     --save_fp run/imagenet/schema_net/init_IR_atlas-deit_small-l9-M_8000.pth \
#     --num_workers 8 \
#     --batch_size 64 \
#     --make_partial 0.1
