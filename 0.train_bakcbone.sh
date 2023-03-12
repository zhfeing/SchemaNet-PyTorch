#!/bin/bash


CV_LIB_PATH=/c22940/zhf/zhfeing_cygpu1/project/cv-lib-PyTorch
export PYTHONPATH=$CV_LIB_PATH:./

export CUDA_VISIBLE_DEVICES=0

# python dist_engine.py \
#     --num-nodes 1 \
#     --rank 0 \
#     --master-url tcp://localhost:$port \
#     --backend nccl \
#     --multiprocessing \
#     --file-name-cfg backbone \
#     --cfg-filepath config/cifar_10/vanilla/deit_tiny.yaml \
#     --log-dir run/cifar_10/vanilla/deit_tiny \
#     --worker backbone_worker

# python dist_engine.py \
#     --num-nodes 1 \
#     --rank 0 \
#     --master-url tcp://localhost:$port \
#     --backend nccl \
#     --multiprocessing \
#     --file-name-cfg backbone \
#     --cfg-filepath config/cifar_100/vanilla/deit_tiny.yaml \
#     --log-dir run/cifar_100/vanilla/deit_tiny \
#     --worker backbone_worker &

# python dist_engine.py \
#     --num-nodes 1 \
#     --rank 0 \
#     --master-url tcp://localhost:$port \
#     --backend nccl \
#     --multiprocessing \
#     --file-name-cfg backbone \
#     --cfg-filepath config/caltech_101/vanilla/deit_tiny.yaml \
#     --log-dir run/caltech_101/vanilla/deit_tiny \
#     --worker backbone_worker &

# python dist_engine.py \
#     --num-nodes 1 \
#     --rank 0 \
#     --master-url tcp://localhost:$port \
#     --backend nccl \
#     --multiprocessing \
#     --file-name-cfg backbone \
#     --cfg-filepath config/caltech_101/vanilla/deit_small.yaml \
#     --log-dir run/caltech_101/vanilla/deit_small \
#     --worker backbone_worker &
