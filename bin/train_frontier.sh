#!/bin/bash -l
#SBATCH --qos=debug
#SBATCH --time=02:00:00
#SBATCH --account=atm112
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH -J afno
#SBATCH -o rwfn_backbone.out
#SBATCH -e rwfn_backbone.err

topdir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
config_file=${topdir}/config/AFNO.yaml
config='afno_backbone'
run_num='0'

export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB

export MASTER_ADDR=$(hostname)

module load cray-python/3.11.5

source /autofs/nccs-svm1_proj/cli115/grnydawn/repos/github/FourCastNet/.venv/bin/activate

# "usgae: python3 train.py [env_name=env_value ...] [--param_name=param_value ...]"

set -x
srun -n 32 -u --mpi=pmi2 \
    bash -c "
    source ${topdir}/bin/export_DDP_vars.sh
    python ${topdir}/src/train.py RWFN_DATA_YAML=${topdir}/cfg/data_frontier.yaml --enable_amp=1 --run_num=$run_num
    "
