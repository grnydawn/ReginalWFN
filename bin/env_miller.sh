#!/bin/bash

RWFN_HOME="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."

# load modules
module load afw-python/3.10-202312
module load cudatoolkit/22.3_11.6
#module load pytorch/2.2.1

# activate virtual environment
#pip install torch torchvision h5py matplotlib ruamel.yaml timm einops scipy torch-tb-profiler
source /lustre/storm/nwp501/proj-shared/grnydawn/RegionalWFN/venv/bin/activate
#source /autofs/nccs-svm1_proj/cli115/grnydawn/repos/github/FourCastNet/.venv/bin/activate

# set environment variables
export MASTER_ADDR=$(hostname)
export MPLCONFIGDIR=/lustre/storm/nwp501/proj-shared/grnydawn/RegionalWFN/tmp/matplotlib
