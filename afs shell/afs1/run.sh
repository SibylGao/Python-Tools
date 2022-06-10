# cur_time=`date_+"%Y%m%d%H%M"` 
job_name=deit_tiny_stu_asc_0510_resume15

# 作业参数 
#group_name="idl-1-yq01-k8s-gpu-p40-8" 
# group_name="idl-40g-0-yq01-k8s-gpu-a100-8" 
group_name="idl-32g-1-yq01-k8s-gpu-v100-8"

job_version="pytorch-1.4.0"
start_cmd="sh job.sh"
wall_time="1200:00:00"
k8s_priority="normal"
k8s_gpu_cards=4
k8s_trainers=1
file_dir="."
echo "config finish"

paddlecloud job train --job-name ${job_name} \
        --group-name ${group_name} \
        --job-conf config.ini \
        --start-cmd "${start_cmd}" \
        --file-dir ${file_dir} \
        --job-version ${job_version} \
        --wall-time ${wall_time} \
        --is-standalone 1 \
        --k8s-trainers ${k8s_trainers} \
        --k8s-gpu-cards ${k8s_gpu_cards} \
        --k8s-priority ${k8s_priority} \
        --is-auto-over-sell 0 \


echo "finish submit"