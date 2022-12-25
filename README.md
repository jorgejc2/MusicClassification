# MusicClassification
A personal project for creating a deep neural network for music classification

## Description
This project is split into two phases. The first phase is collecting music files in the .wav format and getting their power spectral density information with the help of a
cuda fft kernel. 

The second phase is creating a deep neural network using Pytorch to take the PSD's of all the music files and using them to train a network.

## Progress
Thus far I have only completed a C++ program that gets the data from .wav files. I am utilizing a jupyter notebook to document my progress into further detail and compare 
confirm any results from my C++ programs.

## Setup

Go to into the *build/* directory and run:

```sh
cmake ..
```

Afterwards, go into the *build/example/* directory and run:

```sh
make
./main
```

This executable in the example directory will take a path from the user and extract the information from a .wav file into a .txt file. 
