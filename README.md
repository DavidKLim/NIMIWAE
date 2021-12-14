# NIMIWAE
Package for Non-Ignorably Missing Importance Weighted Autoencoder. Uses reticulated Python to run deep learning models through R. 

## Installation
In order to install the released version of the `NIMIWAE` R package, it is important to correctly install Python, and the required Python modules. The package was tested and developed using Python3, and the versions of the Python modules listed below. `NIMIWAE` may work with a different version of Python and/or different versions of the listed modules, but may yield unexpected results. Below is a guide on how to replicate the tested installation of NIMIWAE from scratch. Disclaimer: in order to run `NIMIWAE`, you will need a CUDA-enabled graphics card. Please ensure that you have a CUDA-enabled GPU on your system, and check the version of CUDA that is supported by your GPU card before your installation!

First, you want to make sure you have a working version of Python3 on your system. `NIMIWAE` was tested on Python versions 3.6.6 and 3.7.6.

Next, you want to install the necessary Python module dependencies. Below is a list of Python modules to install.\
numpy=1.19.0\
scipy=1.4.1\
h5py=2.10.0\
pandas=0.25.2\
matplotlib=3.1.3\
cloudpickle=1.2.2\
torch=1.4.0\
torchvision=0.5.0\
tensorflow = 1.14.0\
tensorflow-probability=0.7.0\
tensorboard=1.14.0\
\
The `numpy` and `cloudpickle` packages can be installed using a either the pip or conda installer. One may prefer to use conda if the aim is to create a virtual environment to test out `NIMIWAE` without affecting the installation of these dependencies on the entire system. Please refer [here](https://pip.pypa.io/en/stable/installation/) for how to install the pip installer, and [here](https://docs.anaconda.com/anaconda/install/index.html) for how to install Anaconda and the conda installer. Assuming you are using the pip installer, these packages can be installed by opening your command line, and inputting: ``` pip3 install numpy==1.19.0 cloudpickle==1.2.2 scipy==1.4.1 h5py==2.10.0```.\
\
Before installing the Pytorch modules, you should first confirm that you have a working NVIDIA CUDA installation on your system. If you have not configured a CUDA installation for your system, please follow the guide [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#pre-installation-actions) and follow the steps to install the CUDA toolkit and drivers on your system. Then, the `torch` and `torchvision` modules can be installed by following the instructions listed [here](https://pytorch.org/get-started/previous-versions/). To install the listed versions of these modules, input the following into the command line: ```pip3 install torch==1.4.0 torchvision==0.5.0```. To check that the installation was successfully completed, open a Python3 console using the command `python3` and type\
```
import torch
print(torch.__version__)
print(torch.cuda.is_available())
```
This will verify that the correct version of Pytorch has been installed, and that a CUDA-enabled GPU is available for use in Pytorch.\
\
Finally, the `tensorflow`, `tensorflow-probability`, and `tensorboard` modules are not required for the `NIMIWAE` installation, but these are used by other compared methods, which are explored further in the paper repository <https://www.github.com/DavidKLim/NIMIWAE_Paper>. In order to install `tensorflow`, please follow the instructions [here](https://www.tensorflow.org/install/pip), and make sure to install the gpu-enabled version of `tensorflow`, or `tensorflow-gpu`. The `tensorflow-probability` and `tensorboard` modules should be installed automatically with your `tensorflow` installation.\
\
The `pandas` and `matplotlib` libraries are also optional for `NIMIWAE`, but are useful to install and may be used for comparative methods. To install the versions of these libraries used in our comparative analyses, use the line ```pip3 install matplotlib==3.1.3 pandas==0.25.2```

After making sure your Python environment is properly set-up, you can install the `NIMIWAE` R package by using the command: `devtools::install_github("DavidKLim/NIMIWAE")`.

## Minimal Working Example

Below is a minimal working example to verify that your `NIMIWAE` installation was completed:

``` r
library(MASS)
library(NIMIWAE)
data = MASS::mvrnorm(n=10000, mu=rep(0,2), Sigma=diag(2))
Missing = matrix(rbinom(10000*2,1,0.5),nrow=10000,ncol=2)
data_types = rep("real",2)
g = c(rep("train",8000), rep("valid",1000), rep("test",1000))
res = NIMIWAE::NIMIWAE(dataset="test", data=data, Missing=Missing, g=g, data_types=data_types, ignorable=T)
str(res)
```

NOTE: `NIMIWAE` is built using a Python backend, using the `reticulate` R package. You may be prompted to install Miniconda when running `NIMIWAE` for the first time. This message is from the `reticulate` package, and we advise that you decline the installation and use the system Python version as installed above. Otherwise, Python module dependencies listed above will need to be installed again for the Miniconda version of Python, as `reticulate` will default to the new installation of Python.
