# Install

This package relies on PyTorch=1.7.1, and [PyTorch3d](https://github.com/facebookresearch/pytorch3d) as it's main dependencies.

## Pytorch Simulation Code


### Conda

Using Conda:

`bash
conda create -n pytorch3d python=3.8
conda activate pytorch3d
conda install -c pytorch pytorch=1.7.1 torchvision cudatoolkit=10.2
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
pip install -r requirements.txt
`

### Venv


`bash
source /share/apps/source_files/python/python-3.8.5.source # You can change which python
python3 -m venv —system-site-packages LapVideoUS  # Gives access to site packages
source ~/venvs/LapVideoUS-SUSI/bin/activate
pip install -r requirements.txt --user
pip install --user -e ./ # Install repository
`

Using venv, the second to last step installing nvidiacub wouldn't work, so download and export CUB path locally:

`bash
curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz
tar xzf 1.10.0.tar.gz
export CUB_HOME=$PWD/cub-1.10.0
`

## Legacy SliceSampler dependencies

To pre-process US data for PyTorch differentiable rendering, a utility class `USTensorSlice` has been implemented. This depends on the package `slicesampler`, which has particular CUDA Toolkit (<=10.1) and VS requirements (<=2019). As such, we recommend the usage of two separate environments for simulation of US data, and then useage of this data as part of a differentiable rendering pipeline. You may need `sudo` privileges on your local cluster/machine to install CUDA drivers, check with your local administrator.

Check `slicesampler` is properly installed by:

`bash
git clone https://weisslab.cs.ucl.ac.uk/UltrasoundNoTracker/slicesampler.git
# Run tox in slicesampler repo, checks the tests can be run.
tox -e pycuda
# If tests run, pycuda should be able to run
`

### Creating slicesampler env
`bash
conda create --name slicesampler python=3.6
conda activate slicesampler
pip install -r requirements-slicesampler.txt
`