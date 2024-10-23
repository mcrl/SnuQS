from __future__ import annotations
from dataclasses import dataclass
from braket.snuqs.types import AcceleratorType, PrefetchType, OffloadType
from braket.snuqs._C.operation import GateOperation
from typing import Union


@dataclass
class Subcircuit:
    qubit_count: int
    qubit_count_cpu: int
    qubit_count_cuda: int
    qubit_count_slice: int
    accelerator: AcceleratorType
    prefetch: PrefetchType
    offload: OffloadType
    operations: Union[list[list[GateOperation]], list[GateOperation]]
