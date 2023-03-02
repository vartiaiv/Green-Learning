# Green-Learning
The goal is to experiment with neural networks and find an efficient (low computation) solutions for a machine learning problem.

The approach is to use a feedforward network using SAAB (https://github.com/davidsonic/Interpretable_CNNs_via_Feedforward_Design) and compare it with a traditional CNN such as AlexNet.

**The packages installed:**  
- pytorch (CPU or GPU)  
- torchaudio (audio processing)
- torchvision (image processing)
- pytorch-model-summary (keras like model summary for PyTorch)
- torchscan (model summary)
- scikit-image (preprocessing)
- scikit-learn
- abseil-py (CL flags)
- matplotlib
- memory-profiler (good for peak memory)
- psutil (backend for memory-profiler)
- setuptools

Other dependencies are installed automatically if following the setup instructions.  

**Notes:**  
- Abseil-py is used for for . The original project used them as an easy and legible way to set parameters for multiclass classification such as the class IDs.  
- Pickle is part of Python standard library these days, so no need to install it separately

## Setup run environment

I recommend installing an **Anaconda** Python distribution because it comes with both **conda** and **pip** package managers.  
The correct environment should be easy to setup with **conda**.  

### Installing dependencies

**1. Create a new conda environment on command line and activate it**  
`conda env -n gl`  
`conda activate gl`  
The name 'gl' is just for green learning, it can be whatever you want.  

Next, while the environment is active install the dependencies.

**2. Install either**  
- **PyTorch CPU only**  
`conda install pytorch torchvision torchaudio cpuonly -c pytorch`  

or

- **PyTorch GPU (CUDA)**  
  Before doing so you should need to check the GPU's CUDA version on command line with command `nvidia-smi`  
  If there is text like "CUDA Version: xx.x" and it is >= 9.2 should be good to go.  
  **Otherwise install PyTorch CPU only version.**

  Install newest PyTorch supporting CUDA version 11.6 with command  
  `conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia`  
  or version 11.7 with  
  `conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia`  

  If your CUDA version is <= 11.5 find a suitable command to run at https://pytorch.org/get-started/previous-versions/  

**3. Install the rest**  
`conda install matplotlib`  
`conda install -c conda-forge scikit-learn`  
`conda install scikit-image`  
`pip install absl-py`  
`pip install setuptools`  
`pip install psutil`  
`pip install pytorch-model-summary`
`pip install torchscan`
`pip install -U memory-profiler`


## Running the code
### Command line
You can use command line to run the python scripts. Make sure to have the correct conda environment activated.  

