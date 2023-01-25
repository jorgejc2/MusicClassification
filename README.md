# MusicClassification
A personal project for creating a deep neural network for music classification

## Description
This project is split into two phases. The first phase is collecting music files in the .wav format and getting their power spectral density information using a Short Time Fourier Transform CUDA kernel. Using the STFT, it becomes more feasible extract more meaningful features from 
music files that will assist in its classification for the neural network. The second phase is the neural network itself which will be trained on 
the [GTZAN](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) music dataset. After the training phase, the 
neural network will hopefully be able to accurately determine the genre of any new song it gets fed. 

## Progress
I am currently wrapping up the first phase. I have successfully been able to create CPP functions for reading .wav files and a CUDA STFT function 
for feature extraction. With the help of Pybind11, I have also made these functions into a module that are callable from a Python file. The directory *python_testbenches/* houses python files that are meant to test each function that I turned into a Python module, crosschecked with 
already existing functions from libraries in Python such as **Numpy** and **Scipy**. 

## Setup

For first time setup on WSL with Ubuntu 20.04 on a computer with a NVIDIA RTX, run the following commands:

```sh
# in Windows Command Prompt
wsl.exe --install
wsl.exe --update
C:\> wsl.exe

# in WSL terminal
$ sudo apt-key del 7fa2af80
$ wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/$ cuda-wsl-ubuntu.pin
$ sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
$ wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/$ cuda-repo-wsl-ubuntu-12-0-local_12.0.0-1_amd64.deb
$ sudo dpkg -i cuda-repo-wsl-ubuntu-12-0-local_12.0.0-1_amd64.deb
$ sudo cp /var/cuda-repo-wsl-ubuntu-12-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
$ sudo apt-get update
$ sudo apt-get -y install cuda
$ sudo apt install nvidia-cuda-toolkit

# setting up and upgrading CMake
$ sudo apt purge --auto-remove cmake
$ wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor $ - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
$ sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main'
$ sudo apt update
$ sudo apt install cmake

# installing Python and dev libraries
$ sudo apt-get install python3
$ sudo apt-get install python-dev

# installing NumCPP (Note: do this in any directory)
$ cd <NUMCPP_REPO_PATH>
$ git clone https://github.com/dpilger26/NumCpp.git

$ cd NumCpp
$ mkdir build
$ cd build
$ cmake .. # might need to command to not use BOOST if BOOST is not installed
$ cmake --build . --target install

# in a different directory, clone this repo
$ git clone https://github.com/jorgejc2/MusicClassification.git
$ git submodule update --init
```

Go to into the *build/* directory and run:

```sh
$ cmake .. && make clean && make
```

Afterwards, you can find executables in *build/example/* directory and run them, or go to the root directory and run the files in *python_testbenches/*. Most of the code in *stft_introduction.ipynb* can run with out this entire set up process but some blocks of code that utilize the CUDA functions will not work since they must be compiled from the previous command. 
