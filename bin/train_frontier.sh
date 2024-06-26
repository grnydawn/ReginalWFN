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

module load cray-python/3.9.13.1
module load rocm/6.0.0
module load amd-mixed/6.0.0
module load craype-accel-amd-gfx90a

#account=nwp501
account=cli115

topdir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
config_file=${topdir}/cfg/AFNO_frontier_${account}.yaml
config='afno_backbone'
run_num='1'

export HDF5_USE_FILE_LOCKING=FALSE

#export NCCL_NET_GDR_LEVEL=PHB
#export NCCL_DEBUG=info
#export NCCL_PROTO=Simple
export NCCL_SOCKET_IFNAME=hsn

export MIOPEN_DISABLE_CACHE=1
export MIOPEN_USER_DB_PATH=/tmp/$JOBID
mkdir -p $MIOPEN_USER_DB_PATH

export MASTER_ADDR=$(hostname)

# Create a virtual env. and install the following packages
# >> python3 -m venv .venv
# >> pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
# >> pip install h5py matplotlib ruamel.yaml timm einops scipy torch-tb-profiler

source /autofs/nccs-svm1_proj/${account}/grnydawn/data/RegionalWFN/venv/bin/activate

if [[ -n "${SLURM_NNODES}" ]]; then
  NNODES=${SLURM_NNODES}
  NGPUS=$(echo ${SLURM_NNODES}*${SLURM_GPUS_ON_NODE} | bc)
else
  NNODES=1
  NGPUS=8
fi

echo "#### Running on $NGPUS GPUs of $NNODES nodes. ####"

set -x
srun -n ${NGPUS} -u \
    bash -c "
    source ${topdir}/bin/export_DDP_vars.sh
    python ${topdir}/src/train.py --enable_amp --yaml_config=$config_file --config=$config --run_num=$run_num
    "
