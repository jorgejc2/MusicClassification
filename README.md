# MusicClassification
A personal project for creating a deep neural network for music classification

## Description

Contrary to the title, I am focusing on creating a spoken digit recognition model. There are two phases to this. The first is implementing feature extraction in CUDA. I chose to use CUDA to reinforce my knowledge of parallel programming and hopefully get large speed up processing large data sets by utilizing the gpu on a computer. The feature extraction taking place is Mel Frequency Cepstram Coefficient (MFCC) feature extraction. This is a method for short-term voice feature extraction from audio data. An in depth description of MFCC can be found in the *mfcc_methodology.ipynb* notebook. The second phase involves using a convolutional neural network to take a processed audio file, and determine which digit was spoken in the audio file. I will be training and testing this data on the AudioMNIST dataset combined with recordings of my own voice as well as two peers. We are hoping that the model works with high accuracy (above at 90%) on our own voices, and somewhat ok accuracy on new speakers (hopefully above 60%). A discussion detailing the convolutional neural network will be described in the *recurrent_neural_network.ipynb* notebook. Unfortunately, making a robust universal model would require more complex techniques and would be an interesting problem to tackle in the future. This following article provides a good discussion on [speaker recognition](https://ieeexplore-ieee-org.proxy2.library.illinois.edu/document/7298570). 

Currently I am reworking this repository for spoken digit recognition so that it can be implemented on an Android tablet as a final project my UIUC ECE 420 course. The end goal in that project is to create an app that accurately recognizes the spoken digit uttered by myself or my two peers on the project. 

As I continue working on this repo, I hope to take what I learn from doing spoken digit recognition and apply it to music genre classification in the future. Potential ideas for creating a robust music genre classifcation application would be working with the GZTAN dataset and creating LSTM recurrent neural networks instead of convolutional nerual networks since music snippets would contain time frames that are temporally related. 

## Progress

MFCC has been implemented in CUDA. Optimizing the CUDA code further is another task. Thanks to Pybind11, it has been seamless integrating the CUDA code as a module to be imported in Python files. This means that from a Python script, the audio files can easily be processed by invoking the CUDA commands in the backend, and then used in a neural network using frameworks such as Pytorch and Tensorflow (in this repo, I will be using Pytorch). Next steps are gathering a large data set and training a convolutional neural network following designs from various papers. 

## Setup

In order to run this project, you will need Ubuntu 20.04 and a machine equipped with a NVIDIA RTX. It is also possible to run this project on a Windows machine through WSL as long as it's running Ubuntu 20.04. Below is how to install WSL and set up CUDA for it. For setting up CUDA on a Linux machine with Ubuntu 20.04, you will have to look up the setup process online since these instructions only apply for WSL. 

**NOTE:** This project may work on other versions of Ubuntu but I will heavily suggest version 20.04 since that is the only version I've tested this project. In the future, I will look into using Docker to assist in creating and setting up a Linux environment that has all its dependencies installed such that the local setup process is not so tedious. 

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
# The previous command may produce an output that may be slightly different than the command below. If so, use the command from the terminal output instead
$ sudo cp /var/cuda-repo-wsl-ubuntu-12-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
$ sudo apt-get update
$ sudo apt-get -y install cuda
$ sudo apt install nvidia-cuda-toolkit
```
Again, the above commands are for setting up CUDA on WSL. If you are using a desktop running Ubuntu 20.04 (not through WSL), then you will have to find resources online on how to setup CUDA. 

After setting up CUDA for your machine, run the following commands to get other dependencies set up so that this repository can build. 

```sh
# setting up and upgrading CMake (from https://askubuntu.com/questions/355565/how-do-i-install-the-latest-version-of-cmake-from-the-command-line, answered by Himel)
$ sudo apt purge --auto-remove cmake
$ wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor $ - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
$ sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main' # this is specific to Ubuntu 20.04
$ sudo apt update
$ sudo apt install cmake

# installing Python and dev libraries
$ sudo apt-get install python3
$ sudo apt-get install python-dev

# installing boost libraries 
sudo apt install libboost-all-dev

# installing NumCPP (Note: you can do this in any directory)
$ cd <your_directory>
$ git clone https://github.com/dpilger26/NumCpp.git

$ cd NumCpp
$ mkdir build
$ cd build
$ cmake .. -DNUMCPP_NO_USE_BOOST=ON
$ sudo cmake --build . --target install

# in your preferred directory, you can finally set up this repository
$ cd <your_other_directory>
$ git clone https://github.com/jorgejc2/MusicClassification.git
$ git submodule update --init
```

Go to into the *build/* directory and run:

```sh
$ cmake .. && make clean && make
```

Finally, to get set up with using PyTorch, invoke the following commands. These will take a while since the tar files that get collected are fairly large. 

```sh
$ python3 -m pip install 'pycuda<2021.1'
$ python3 -m pip install torch
```

Afterwards, you can find executables in the *build/example/* directory and run them, or go to the root directory and run the files in *python_testbenches/*. Most of the code in *stft_introduction.ipynb* can run with out this entire set up process but some blocks of code that utilize the CUDA functions will not work since they must be compiled from the previous commands. Only the *stft_introduction.ipynb* notebook is fully updated and goes into explanation on how the Short Time Fourier Transform works and demonstrates its usefulness for extracting frequency content from music files. I will later be cleaning and updating the *music_fft.ipynb.ipynb* notebook detailing what kind of features will be getting fed to the neural network and how the neural network will be designed. 
