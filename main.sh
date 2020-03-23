# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash


#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=2
#SBATCH -J train_dc_noSobel
#SBATCH --output=/dc_noSobel/logs/slurm-%j.out
#SBATCH --error=/dc_noSobel/logs/slurm-%j.err


DIR="/home/CUSACKLAB/annatruzzi/imagenet_sample"
ARCH="alexnet"
LR=0.05
WD=-5
K=10000
K=10
WORKERS=2
EXP="/home/CUSACKLAB/annatruzzi/deepcluster_noSobel/checkpoints"
#PYTHON="/home/CUSACKLAB/annatruzzi/anaconda3/envs/pytorch_p27/bin/python"
PYTHON = "pyhton"

mkdir -p ${EXP}

source /home/annatruzzi/conda_env_config/create_pytorch_p27.sh

CUDA_VISIBLE_DEVICES=2 ${PYTHON} main.py ${DIR} --exp ${EXP} --arch ${ARCH} \
  --lr ${LR} --wd ${WD} --k ${K} --sobel --verbose --workers ${WORKERS}
