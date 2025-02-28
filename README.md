## Table of Contents
- [Overview](#overview)
- [Environment Setup](#environment-setup)
  - [Create and Activate a Conda Virtual Environment](#create-and-activate-a-conda-virtual-environment)
  - [Install Dependencies](#install-dependencies)
  - [Install PyTorch-GPU](#install-pytorch-gpu)
    - [Check CUDA Version](#check-cuda-version)
    - [Install CUDA](#install-cuda)
    - [Install PyTorch-GPU](#install-pytorch-gpu-1)
  - [Verify PyTorch-GPU Installation](#verify-pytorch-gpu-installation)

---

## Overview
This section provides a detailed guide on how to install the dependencies required for the project, as well as PyTorch-GPU. including creating a virtual environment, installing dependencies, and configuring GPU support. Follow each step carefully to ensure all components are installed and configured correctly.

---

## Environment Setup

### Create and Activate a Conda Virtual Environment

1. Clone the `mednerf` project from GitHub:
   ```bash
   git clone https://github.com/abrilcf/mednerf.git

2. Create a virtual environment (you can customize the environment name; here, we use `graf`):
   ```bash
   conda create --name graf python=3.10
   ```

3. Activate the virtual environment:
   ```bash
   conda activate graf
   ```

---

### Install Dependencies

Run the view reconstruction script and install the required dependencies based on any errors encountered:

1. Install `ignite`:
   ```bash
   pip install ignite
   ```

2. Install `pytorch-ignite`:
   ```bash
   pip install pytorch-ignite
   ```

3. Install `torchvision`:
   ```bash
   pip install torchvision
   ```

4. Install `tqdm`:
   ```bash
   pip install tqdm
   ```

5. Install `opencv-python`:
   ```bash
   pip install opencv-python
   ```

6. Install `chardet`:
   ```bash
   pip install chardet
   ```

7. Install `imageio`:
   ```bash
   pip install imageio
   ```
   
8. Install `pyyaml`:
   ```bash
    pip install pyyaml
   ```

9. Install `scikit-image`:
   ```bash
   pip install scikit-image
   ```
9. Install `scikit-image`:
   ```bash
   pip install scikit-image
   ```
10. Install `IPython`:
   ```bash
    pip install IPython
   ```

11. Install `matplotlib`:
   ```bash
    pip install matplotlib
   ```

12. Install `imageio` and `imageio-ffmpeg`:
   ```bash
   pip install imageio==2.12
   pip install imageio-ffmpeg
   ```

13. Install `scikit-learn`:
   ```bash
   pip install scikit-learn
   ```

14. Install `tensorboardX`:
   ```bash
   pip install tensorboardX
   ```

---

## Environment Setup
### Install Anaconda
Before installing PyTorch-GPU, you need to install Anaconda. Anaconda is a popular Python distribution that includes a wide range of libraries for scientific computing and data analysis.

1. Visit the Anaconda website: [Anaconda Download](https://www.anaconda.com/download).
2. Download and install Anaconda.
3. After installation, restart your computer.
4. Open the command line and run the following command to check the Anaconda version and confirm successful installation:
   ```bash
   conda -V
5.If the version number is displayed, Anaconda has been installed successfully.
### Install PyTorch-GPU

#### Check CUDA Version
Before installing PyTorch-GPU, confirm the CUDA version supported by your computer.

1. Open the command line (cmd) and run:
   ```bash
   nvidia-smi
   ```

2. Check the output to find the maximum supported CUDA version. For example, if it shows CUDA 12.8, you need to install a CUDA version less than or equal to 12.8.

#### Install CUDA
Download and install the appropriate CUDA toolkit version from the NVIDIA website based on your computer's supported CUDA version.

1. Visit the CUDA Toolkit Archive: [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive).
2. Download and install the CUDA version compatible with your system.
3. After installation, CUDA will be installed by default in the `C:\Program Files\NVIDIA GPU Computing Toolkit` directory.

#### Install PyTorch-GPU
Finally, install the PyTorch-GPU version.

1. Open Anaconda Prompt.
2. Activate your virtual environment (replace `name` with the environment name you created earlier):
   ```bash
   conda activate name
   ```

3. Install PyTorch-GPU using the following command:
   ```bash
   conda install pytorch torchvision torchaudio cudatoolkit=<your_cuda_version> -c pytorch
   ```
   Replace `<your_cuda_version>` with the CUDA version you installed, e.g., `11.3`.

---

### Verify PyTorch-GPU Installation

#### Method 1
Run the following command to check if `torch`, `torchvision`, and `torchaudio` are installed:
```bash
pip list
```

#### Method 2
1. Activate the environment and enter Python:
   ```bash
   python
   ```

2. Run the following commands to check if GPU is available:
   ```python
   import torch
   torch.cuda.is_available()
   ```
   If it returns `True`, the installation is successful.

---

## Completion
You have now successfully installed PyTorch-GPU and its dependencies.



