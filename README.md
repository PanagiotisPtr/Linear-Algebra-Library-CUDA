# Linear Algebra Library CUDA
Simple implementation of a Linear Algebra Library in CUDA

##How to run the code
The code is tested on linux ubuntu 16.04 with CUDA 8.0 and C++11
To run the code you will only need to get CUDA 8.0 installed and setup. And run the compile.sh script from the terminal.

Note: I do not have the nvcc compiler directory at the environment variables on my machine so if your directory is different 
make sure that you change it. Or if you already have nvcc on your environment variables you can remove it and just compile the 
main.cu file.

###About the Project

This is a very simple implementation of vector/matrix operations on the GPU. I am not an expert with CUDA so the code may not be (and probably isn't)
very efficient but it works. The main purpose of this library is to help me with a Neural Network project I am working on.
Also I do not really like using other libraries. I think that if you need something it is better to try make it yourself so that you can understand how it works rather than just including a library.
Lastly the matrix operations include operations such as Determinant, Inverse and Cofactor of a matrix.

I hope that you will find the code helpful and educational.l

Panagiotis Petridis,
High School Student

Greece
