from dataclasses import dataclass
from braket.snuqs.types import AcceleratorType, OffloadType
from braket.snuqs._C.operation import GateOperation

from typing import Union

@dataclass
class Subcircuit:
    qubit_count: int
    max_qubit_count: int
    max_qubit_count_cuda: int
    accelerator: AcceleratorType
    offload: OffloadType
    operations: Union[list[GateOperation], list[list[GateOperation]], list[list[list[GateOperation]]]]
