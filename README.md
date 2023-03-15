# Green-Learning
This repository exists mainly for documenting practical work done as part of Tampere University seminar course on Media Analysis. The course involves conducting research on different machine learning (ML) and artificial intelligence (AI) applications on various media (images, audio, text). It should be noted that the repository is **not** designed to be ready-to-use as the work is rather explorative.

Our goal is to find an efficient (low computation) solutions for an image classification problem. Convolutional neural networks (CNNs) are known to be trained using gradient-based optimization techniques i.e. backpropagation (BP). For deep models BP can be mathematically intractable so it is crucial to find alternative approaches for deep network optimization. 

We analyzed the efficiency of a feedforward (FF) design of a CNN without any BP proposed by Kuo et al. [1]. We adapt the provided code for the FF design [2] for debugging and efficiency measuring purposes and implement the BP design using the PyTorch ML library. Python standard library and various packages listed below are used for obtaining the efficiency and utility metrics.

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
You can use command line to run the python scripts. Make sure to have the correct conda environment activated. The expected run directory is the top level directory i.e. `Green-Learning/`.

The FF_CNN training steps can be run separately via scripts `getkernel.py`, `getfeature.py` and `getweights.py` OR in one go using `train.py` and tested using test.py

The LENET5 can be trained and tested in `main.py` or separately via `train_lenet.py` and `test_lenet.py`. Script `time_test.py` is like `main.py` but uses PyTorch Timer to clock the run times for training and testing.

## References
[1] C.-C. J. Kuo, M. Zhang, S. Li, J. Duan, and Y. Chen, “Interpretable convolutional neural networks via feedforward design,” Journal of Visual Communication and         Image Representation, vol. 60, pp. 346–359, Apr. 2019, doi: 10.1016/j.jvcir.2019.03.010.

[2] jialiduan, “Interpretable_CNNs_via_Feedforward_design.” Feb. 07, 2023. Accessed: Mar. 09, 2023. [Online]. Available: https://github.com/davidsonic/Interpretable_CNNs_via_Feedforward_Design

[3] Y. Lecun, L. Bottou, Y. Bengio, and P. Haffner, “Gradient-based learning applied to document recognition,” Proceedings of the IEEE, vol. 86, no. 11, pp. 2278–2324, 1998, doi: 10.1109/5.726791.
