#!/bin/bash

#RWFN_HOME="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."

## Check if MASTER_ADDR is provided
#if [ -z "$1" ]; then
#    export MASTER_ADDR=$(hostname)
#else
#    # Check if MASTER_ADDR is provided
#    if [[ "$1" == -* ]]; then
#        export MASTER_ADDR=$(hostname)
#    else
#        export MASTER_ADDR=$1
#        shift
#    fi
#fi

# load modules


#pip list --disable-pip-version-check
# activate virtual environment
#source /autofs/nccs-svm1_proj/cli115/grnydawn/repos/github/FourCastNet/.venv/bin/activate

# set environment variables

export TEST_ENV=1

# run
#python3 ${RWFN_HOME}/src/train.py $@
