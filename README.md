# SnuQS

For we are currently working on updates for AWS cluster supports with Amazon braket.
- Seamless integration with Amazon braket
- AWS support
- Multi-node support
- Futher Optimizations


## Prerequisite
- Any distribution of MPI (e.g., OpenMPI, MPICH, MVAPICH)
    * It has been tested with MPICH Version 4.1.2.
    * Latest versions are recommended.
- python >= 3.11.8
    * We recommend to use Python 3.11 not Python 3.12.
- cuda >= 12.1

For example, if you are using anaconda or miniconda,
```
conda create -n <name> python=3.11.8
conda activate <name>
...
```


## Installation
```
pip install .
```

## (Optional) QASM Parser Generation
See [grammar](grammar).
