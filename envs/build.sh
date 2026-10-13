#!/bin/bash
# Creates the conda environment "soar-gs" with habitat-sim 0.2.4, PyTorch 2.1.2 (CUDA 11.8)
# and the packages in envs/requirements.txt. Requires conda, git, and the CUDA 11.8 toolkit (nvcc).
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CONDA_PATH=$(conda info --base)

source ${CONDA_PATH}/etc/profile.d/conda.sh
conda create -y -n soar-gs python=3.9 cmake=3.14.0
conda activate soar-gs

export PYTHONNOUSERSITE=True
pip install pip==25.0 setuptools==75.8.0 wheel numpy==1.26.3

# habitat simulator 0.2.4 (fork with a fix for the Replica PTex shader)
cd ${ROOT}/simulator
[ -d habitat-sim ] || git clone https://github.com/liren-jin/habitat-sim.git
cd habitat-sim
git checkout 3023da6e8617a3b676b3e9f8845788631f8377d5
pip install -r requirements.txt numpy==1.26.3
python setup.py install --headless --bullet

# PyTorch and the remaining packages
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install --no-build-isolation -r ${ROOT}/envs/requirements.txt
