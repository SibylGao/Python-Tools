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

# ----- dataset -----
mkdir -p ./data/imagenet

#nohup tar --use-compress-program=pigz -xpf afs/dongkun/imagenet/train.tgz -C ./data/imagenet > tar_train_log.txt 2>&1 &
#nohup tar --use-compress-program=pigz -xpf afs/dongkun/imagenet/val.tgz -C ./data/imagenet > tar_val_log.txt 2>&1 &
# tar -xvf afs/dongkun/imagenet/train.tar -C ./data/imagenet > tar_train_log.txt 2>&1 
# tar -xvf afs/dongkun/imagenet/val.tar -C ./data/imagenet > tar_val_log.txt 2>&1 

tar -xf afs/dongkun/imagenet/train.tar -C ./data/imagenet  2>&1 
tar -xf afs/dongkun/imagenet/val.tar -C ./data/imagenet  2>&1 

cp afs/dongkun/imagenet/*.txt ./data/imagenet 2>&1 
#nohup cp afs/dongkun/imagenet/*.txt ./data/imagenet 2>&1 &

# ----- envs -----
# mkdir -p ./envs
# tar -xf ./afs/gaoshiyu/deit_env.tar -C ./envs

mkdir -p ./envs
tar -xf ./afs/gaoshiyu/deit_env.tar -C ./envs

# ----- pretrain models -----


# ----- python envs -----
export PYTHON_HOME=./envs/deit
export PATH=${PYTHON_HOME}/bin:$PATH
export LD_LIBRARY_PATH=${PYTHON_HOME}/lib/:${LD_LIBRARY_PATH}


# ----- run -----

# python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model deit_tiny_patch16_224 --batch-size 256 --data-path ./data/imagenet/ --output_dir ./output  > train_log_0.txt 2>&1 


python -m torch.distributed.launch \
       --nproc_per_node=4 \
       --use_env main.py \
       --model deit_tiny_distilled_patch16_224 \
       --batch-size 256 \
       --data-path ./data/imagenet/ \
       --output_dir /root/paddlejob/workspace/log/ \
       --teacher-model cait_S24_224 \
       --teacher-path ./S24_224.pth \
       --resume ./output/checkpoint.pth \
       --distillation-type linear > train_log_0.txt 2>&1


