#!/bin/bash
#SBATCH --partition=research
#SBATCH --output=/lustre/scratch/client/movian/research/users/khoilm1/slurm_log/%x-%j.out
#SBATCH --error=/lustre/scratch/client/movian/research/users/khoilm1/slurm_log/%x-%j.out
#SBATCH --job-name=pt_llm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --mail-user=v.khoilm1@vinai.io
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --exclude=sdc2-hpc-dgx-a100-001


srun --container-image=/lustre/scratch/client/movian/research/users/khoilm1/setup/docker_images/dc-miniconda3-py:38-4.10.3-cuda11.4.2-cudnn8-ubuntu20.04.sqsh \
     --container-mounts=/lustre/scratch/client/movian/research/users/khoilm1/:/root/ \
     --container-workdir=/root/ \
     /bin/bash -c \
     "
     export HTTP_PROXY=http://proxytc.vingroup.net:9090/
     export HTTPS_PROXY=http://proxytc.vingroup.net:9090/
     export http_proxy=http://proxytc.vingroup.net:9090/
     export https_proxy=http://proxytc.vingroup.net:9090/

     export TOKENIZERS_PARALLELISM=false
     export HF_HOME=/root/.cache/huggingface/
     export HF_DATASETS_CACHE=/root/.cache/huggingface/datasets/

     cd /root

     source /opt/conda/bin/activate
     cd research/base-llm-factory/
     conda activate /root/envs/khoi.trl
     wandb offline

     export CUDA_VISIBLE_DEVICES=0
     export FORCE_TORCHRUN=1

     llamafactory-cli train training_script/basic/train/pt.yaml
     cd slurm/notify_slack
     python notify_slack.py
     "
