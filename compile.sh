export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
nvcc main.cu -o a.out -std=c++11