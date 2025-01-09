## Parallel state estimation for systems with integrated measurements

This repository contains the code for the paper "Parallel state estimation for system with integrated measurements".
The code leverages JAX and implements the parallel state estimation algorithm for a system with integrated measurements.

This code is written by [Fatemeh Yaghoobi](https://github.com/Fatemeh-Yaghoobi) and [Simo Särkkä](https://github.com/ssarkka). 
ArXiv link: https://arxiv.org/abs/2410.00627

## Installation

1. Create a virtual environment and clone this repository
2. Install JAX (preferably with GPU support) following https://github.com/google/jax#installation

## Examples

Examples (reproducing the experiments from our paper) can be found in the [tests folder](https://github.com/Fatemeh-Yaghoobi/Parallel-integrated-method/tree/master/tests/linear). 

## How to cite

If you find this work useful, please cite us in the following way:

```
@article{yaghoobi2024parallel,
  title={Parallel state estimation for systems with integrated measurements},
  author={Fatemeh Yaghoobi and Simo S\"arkk\"a},
  journal={IEEE Signal Processing Letters},
  year={2024},
  publisher={IEEE}
}
```
