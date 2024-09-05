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

## License
Copyright (c) 2022 Seoul National University.
    
All rights reserved.

You are hereby authorized to use, reproduce, and modify the software and documentation, and
to distribute the software and documentation, including modifications, for non-commercial 
purposes only.

Redistribution and use in source and binary forms, with or without modification, are 
permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of 
conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of 
conditions and the following disclaimer in the documentation and/or other materials provided 
with the distribution.

3. Neither the name of Seoul National University nor the names of its contributors may be
used to endorse or promote products derived from this software without specific prior 
written permission.

4. *The grant of the rights will not include and does not grant you the right to sell the 
software, and does not grant to you the right to license the software*. For purposes of the 
foregoing, “sell” means practicing any or all of the rights granted to you to provide a 
product or service whose value derives, entirely or substantially, from the functionality
of the software to third parties for a fee or other consideration (including without 
limitation fees for hosting or consulting/support services related to the software).

THIS SOFTWARE IS PROVIDED BY SEOUL NATIONAL UNIVERSITY “AS IS” AND ANY EXPRESS OR IMPLIED 
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SEOUL NATIONAL UNIVERSITY 
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, 
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING 
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH 
DAMAGE.

Contact information:
Thunder Research Group
Department of Computer Science and Engineering
Seoul National University, Seoul 08826, Korea
https://thunder.snu.ac.kr

Contributors:
Daeyoung Park, Heehoon Kim, Jinpyo Kim and Jaejin Lee
