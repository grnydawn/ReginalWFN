#!/bin/bash
#SBATCH -A atm112
#SBATCH -J FCNet
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -o job-%j.out
#SBATCH -e job-%j.err

topdir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."

#source ~/miniconda_frontier/etc/profile.d/conda.sh
source "/sw/rhea/python/3.7/anaconda3/2018.12/etc/profile.d/conda.sh"

conda deactivate #leave the base conda environemnt. delete this line if base environment not activated

ulimit -n 65536
source ${topdir}/source_env_YSK.sh

export NCCL_SOCKET_IFNAME=hsn
export NCCL_DEBUG=info

export MIOPEN_DISABLE_CACHE=1
export NCCL_PROTO=Simple
export MIOPEN_USER_DB_PATH=/tmp/$JOBID
mkdir -p $MIOPEN_USER_DB_PATH


export OMP_NUM_THREADS=7
export PYTHONPATH=$PWD:$PYTHONPATH


config_file=${topdir}/cfg/AFNO_YSK.yaml
config='afno_backbone'
run_num='0'

#export NNODES=${SLURM_JOB_NUM_NODES}
export NNODES=4
export MASTER_ADDR=$(hostname)

time srun -n $((NNODES*8)) \
       	python ${topdir}/src/train_ddp_YSK.py --enable_amp --yaml_config=$config_file --config=$config --run_num=$run_num

