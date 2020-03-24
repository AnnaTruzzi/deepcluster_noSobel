# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash


#SBATCH --gpus-per-task=2
#SBATCH -J train_dc_noSobel
#SBATCH --output=/dc_noSobel/logs/slurm-%j.out
#SBATCH --error=/dc_noSobel/logs/slurm-%j.err


DIR="/data/ILSVRC2012/train"
ARCH="alexnet"
LR=0.05
WD=-5
K=10000
K=10
WORKERS=2
EXP="/home/CUSACKLAB/annatruzzi/deepcluster_noSobel/checkpoints"
PYTHON="/opt/anaconda3/envs/dc_p27/bin/python"

mkdir -p ${EXP}

CUDA_VISIBLE_DEVICES=2 ${PYTHON} main.py ${DIR} --exp ${EXP} --arch ${ARCH} \
  --lr ${LR} --wd ${WD} --k ${K} --sobel --verbose --workers ${WORKERS}
