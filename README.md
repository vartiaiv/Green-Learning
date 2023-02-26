# Green-Learning
The goal is to experiment with neural networks and find an efficient (low computation) solutions for a machine learning problem.

The approach is to use a feedforward network using SAAB (https://github.com/davidsonic/Interpretable_CNNs_via_Feedforward_Design) and compare it with a traditional CNN such as AlexNet.

**The packages installed:**  
- PyTorch (CPU or GPU)  
- TorchAudio (audio processing and I/O)
- TorchVision (image processing and I/O)
- THOP
- Matplotlib
- Scikit-learn (preprocessing the data at hand)
- Scikit-image (preprocessing the data at hand)
- Abseil-py (command line flags compatible with Tensorflow syntax)
- Setuptools (makes project structure setup easier aka. enchances quality of life)
- Psutil (can be used to measure RAM peak and avg for example)
- Pytorch-model-summary (keras like model summary for PyTorch)

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
`pip install thop`  
`conda install matplotlib`  
`conda install -c conda-forge scikit-learn`  
`conda install scikit-image`  
`pip install absl-py`  
`pip install setuptools`  
`pip install psutil`  
`pip install pytorch-model-summary`

### Setup project

Finally you need to run the following command in the directory where **setup.py** is contained  
`python -m pip install -e .`  

This allows entering an editable (hence -e) developement mode of the installment.  
In the developement mode you should be able to edit the code and have changes available when running the scripts.  
For example you should be able to import following from any of the files independent of the location in the project:  
`import src.utils.timer` or `from src.utils.timer import timeit`



## Running the code
### Command line
You can use command line to run the python scripts. Make sure to **have the correct conda environment activated**!  

### VSCode with Python extension
If you want to run and debug the code I recommend VSCode with its Python extension.  
VSCode is a lightweight customizable code editor that can be made to behave like a fully fledged IDE.  

After installing Python on VSCode you may also need to select the correct Python interpreter for the project.
Use the one from the **conda environment you created in step 1** of the setup so that the packages are available.  
You can select the interpreter with **Ctrl+Shift+P** and by typing "python select interpreter" in the search box.  

Now try opening (or creating) a Python script in the editor and run it by pressing F5.  
If there isn't a run configuration in `.vscode\launch.json` yet, it needs to be configured now. 
The default configuration runs the active file in editor. 

If you want instead a specific run configuration, go to Run->Open configurations (or open `.vscode\launch.json`).  
There, if you change text `"program": "${file}",` into `"program": "${workspaceFolder}\\script.py",`,
pressing F5 now runs `script.py` in the workspace folder regardless of the active file.

## Troubleshooting
- In VSCode if you cannot find the Python interpreter you wanted, you need to add the following (or similar) location to the PATH:  
  `C:\Users\<yourname>\anaconda3\Scripts`  
  (this is the location by default when installing Anaconda on Windows)
- If you want to setup the VSCode run configurations from a clean slate, just delete the file `.vscode\launch.json`  
  and go to Run->Add configuration (Python file).
