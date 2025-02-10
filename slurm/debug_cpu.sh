srun --partition=research \
     --job-name=no-gpu \
     --pty \
     --nodes=1 \
     --ntasks=1 \
     --gpus-per-node=0 \
     --cpus-per-task=4 \
     --mem=40G \
     --container-image=/lustre/scratch/client/movian/research/users/khoilm1/setup/docker_images/dc-miniconda3-py:38-4.10.3-cuda11.4.2-cudnn8-ubuntu20.04.sqsh \
     --container-mounts=/lustre/scratch/client/movian/research/users/khoilm1/:/root/ \
     --container-workdir=/root/ \
     --mail-user=v.khoilm1@vinai.io \
     --mail-type=END \
     --mail-type=FAIL \
     /bin/bash
