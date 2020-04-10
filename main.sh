#!/bin/bash
#
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH -J train_dc_noSobel
#SBATCH --output=/home/annatruzzi/deepcluster_noSobel/logs/slurm-%j.out
#SBATCH --error=/home/annatruzzi/deepcluster_noSobel/logs/slurm-%j.err


DIR="/data/ILSVRC2012/train"
ARCH="alexnet"
LR=0.05
WD=-5
K=10000
K=10
WORKERS=12
EXP="/home/annatruzzi/checkpoints/deepcluster_checkpoints/"
PYTHON="/opt/anaconda3/envs/dc_p27/bin/python"
CHECKPOINTS=5005
RESUME="/home/annatruzzi/checkpoints/deepcluster_checkpoints/checkpoint_dc_44.pth.tar"

${PYTHON} main.py ${DIR} --exp ${EXP} --arch ${ARCH} \
  --lr ${LR} --wd ${WD} --k ${K} --sobel --verbose --workers ${WORKERS} \
  --checkpoints ${CHECKPOINTS} --resume ${RESUME}
