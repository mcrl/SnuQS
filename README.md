# SnuQS

## Prerequisite
- Any distribution of MPI (e.g., OpenMPI, MPICH, MVAPICH)
    * It has been tested with MPICH Version 4.1.2.
    * Latest versions are recommended.
- python >= 3.11.5
    * We recommend to use Python 3.11 not Python 3.12.
- cmake >= 3.26.4
- cuda >= 11.7

For example, if you are using anaconda or miniconda,
```
conda create -n snuqs python=3.11
conda activate snuqs
conda install cmake
...
```


## Installation
```
pip install -r requirements.txt
pip install .
```

## (Optional) QASM Parser Generation
See [antlr4](antlr4).
