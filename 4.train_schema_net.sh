#!/bin/bash

export PYTHONPATH=$CV_LIB_PATH:./

export CUDA_VISIBLE_DEVICES=0

# python dist_engine.py \
#     --num-nodes 1 \
#     --rank 0 \
#     --master-url tcp://localhost:$port \
#     --backend nccl \
#     --multiprocessing \
#     --file-name-cfg SchemaNet \
#     --cfg-filepath config/cifar_10/schema_net/deit_tiny-l9-M_128.yaml \
#     --log-dir run/cifar_10/schema_net/deit_tiny-l9-M_128 \
#     --worker schema_net_worker &

# export CUDA_VISIBLE_DEVICES=1
# python dist_engine.py \
#     --num-nodes 1 \
#     --rank 0 \
#     --master-url tcp://localhost:$port \
#     --backend nccl \
#     --multiprocessing \
#     --file-name-cfg SchemaNet \
#     --cfg-filepath config/cifar_100/schema_net/deit_tiny-l9-M_1024.yaml \
#     --log-dir run/cifar_100/schema_net/deit_tiny-l9-M_1024 \
#     --worker schema_net_worker &

# python dist_engine.py \
#     --num-nodes 1 \
#     --rank 0 \
#     --master-url tcp://localhost:$port \
#     --backend nccl \
#     --multiprocessing \
#     --file-name-cfg SchemaNet \
#     --cfg-filepath config/caltech_101/schema_net/deit_small-l9-M_1024.yaml \
#     --log-dir run/caltech_101/schema_net/deit_small-l9-M_1024 \
#     --worker schema_net_worker &

# python dist_engine.py \
#     --num-nodes 1 \
#     --rank 0 \
#     --master-url tcp://localhost:$port \
#     --backend nccl \
#     --multiprocessing \
#     --file-name-cfg SchemaNet \
#     --cfg-filepath config/caltech_101/schema_net/deit_small-l9-M_1024-no_init.yaml \
#     --log-dir run/caltech_101/schema_net/deit_small-l9-M_1024-no_init \
#     --worker schema_net_worker &

# export CUDA_VISIBLE_DEVICES=0,1,2,3
# port=9003

# python dist_engine.py \
#     --num-nodes 1 \
#     --rank 0 \
#     --master-url tcp://localhost:$port \
#     --backend nccl \
#     --multiprocessing \
#     --file-name-cfg SchemaNet \
#     --cfg-filepath config/imagenet/schema_net/deit_small-l9-M_8000.yaml \
#     --log-dir run/imagenet/schema_net/deit_small-l9-M_8000 \
#     --worker schema_net_worker &

