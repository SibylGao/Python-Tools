#!/bin/bash
# coding:utf-8
current_rank_index=${POD_INDEX}
rank_0_ip=${POD_0_IP}
free_port=${TRAINER_PORTS}
dist_url="tcp://${rank_0_ip}:${free_port}"
world_size=2
echo "current_rank_index: ${current_rank_index}"
echo "dist_url: ${dist_url}"
echo "world_size: ${dist_url}"

#vpython keep.py

WORK_PATH=/root/paddlejob/workspace/env_run
# ----- programs -----
tar -xf ./afs/jinhailong/programs.tar -C ./
export PATH=$WORK_PATH/programs/bin:$PATH 

nvidia-smi
# ----- dataset -----
mkdir -p ./data/imagenet
tar --use-compress-program=pigz -xpf afs/jinhailong/imagenet/train.tgz -C ./data/imagenet 
tar --use-compress-program=pigz -xpf afs/jinhailong/imagenet/val.tgz -C ./data/imagenet 
cp afs/jinhailong/imagenet/*.txt ./data/imagenet 

# ----- envs -----
mkdir -p ./envs
tar -xvf ./afs/gaoshiyu01/deit_env.tar -C ./envs
# tar -xvf ./afs/gaoshiyu01/minivit2.tar -C ./envs

# ----- pretrain models -----

# ----- python envs -----
export PYTHON_HOME=./envs/deit
export PATH=${PYTHON_HOME}/bin:$PATH
export LD_LIBRARY_PATH=${PYTHON_HOME}/lib/:${LD_LIBRARY_PATH}


# ----- run -----

python -m torch.distributed.launch \
        --nproc_per_node=4 \
        --use_env main.py \
        --model deit_tiny_distilled_patch16_224 \
        --batch-size 256 \
        --data-path ./data/imagenet/ \
        --output_dir ../log --teacher-model cait_S24_224 \
        --teacher-path ./S24_224.pth \
        --resume ./output/checkpoint144.pth \
        --distillation-type none > train_log_0.txt 2>&1




