# gnoimi

## Introduction.
reference: https://github.com/arbabenko/GNOIMI

The implementation of the (Generalized) Non-Orthogonal Inverted Multi-Index structure from the paper "Efficient Indexing of Billion-Scale Datasets of Deep Descriptors", CVPR 2016.

The code requires Yael library (deps intel mkl) and Intel MKL library.
training_gnoimi.cpp contains the code of the codebooks training.
searching_gnoimi.cpp contains the code of the index searching.

## Installation.

now, this lab is only compiled on wsl-ubuntu 22.04.

### Dependency.
1. install intel-mkl
2. compile yael based on intel-mkl.
3. place headers and libs to proper position.


```
git clone git@github.com:BismarckDD/gnoimi-impl.git
cd gnoimi-impl
mkdir build && cd build && cmake ..
make -j12
```

## Update.
### 2022.06.20 

### 2022.06.19
1. fix downloadDeep1B.py to python3 style.

### 2002.06.18 
1. Use Intel MKL instead of cblas, lapack and OpenBlas to rewrite the code.
2. Extract common headers to common.h.
3. Rename the cpp files, variable and function name according to Google's cpp format.
4. Use marco to rewrite thread work assign and join, thus the code seems to be concise.
WARNING: the correctness is still to be tested further. Don't use it in production environment unless you know what you are doing.


## cblas function.

```
/*
cblas_saxpy(N, alpha, X, incX, Y, incY) 
    => Y = sum(alpha * X[i] + Y[i])
cblas_sdot(N, X, incX, Y, incY)
    => Y =sum(X[i] * Y[i])
*/
```
sift feature: dim = 128
gist feature: dim = 960