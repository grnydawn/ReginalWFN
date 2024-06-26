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
outdir="/autofs/nccs-svm1_proj/${account}/${USER}/data/RegionalWFN/output/${config}/${run_num}"

export HDF5_USE_FILE_LOCKING=FALSE

export MIOPEN_DISABLE_CACHE=1
export MIOPEN_USER_DB_PATH=/tmp/$JOBID
mkdir -p $MIOPEN_USER_DB_PATH

# Create a virtual env. and install the following packages
# >> python3 -m venv .venv
# >> pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
# >> pip install h5py matplotlib ruamel.yaml timm einops scipy torch-tb-profiler

source /autofs/nccs-svm1_proj/${account}/grnydawn/data/RegionalWFN/venv/bin/activate

python ${topdir}/src/inference.py \
	--yaml_config=$config_file \
	--config=$config \
	--run_num=$run_num \
	--weights "${outdir}/training_checkpoints/best_ckpt.tar" \
	--embed_dim=712 \
	--override_dir "${outdir}/inference"
