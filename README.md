# Green-Learning

# Setup PyTorch using conda
Packages installed
using conda environment

Create environment named gl (for green learning)
conda env -n gl

Activate the environment
conda activate gl

Choose to either
a) Install PyTorch GPU
Check the computer's CUDA capabilities and version command line with nvidia-smi
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
(replace cuda version with 

Install CUDA Toolkit and
https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-windows




b) Install PyTorch CPU
conda install pytorch torchvision torchaudio cpuonly -c pytorch


conda list should look like
