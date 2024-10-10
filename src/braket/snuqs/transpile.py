from braket.snuqs._C.operation import GateOperation
from functools import cmp_to_key
from braket.snuqs.device import DeviceType
from braket.snuqs.offload import OffloadType
from typing import Optional

def pseudo_lt(oppos1, oppos2):
    op1, pos1 = oppos1
    op2, pos2 = oppos2

    if len(op1.targets) == 0 and len(op2.targets) == 0:
        return pos1 < pos2

    if len(op1.targets) == 0:
        return True
    if len(op2.targets) == 0:
        return False
    if len(set(op1.targets) & set(op2.targets)) != 0:
        return pos1 < pos2

    return max(op1.targets) < max(op2.targets)

def pseudo_gt(oppos1, oppos2):
    op1, pos1 = oppos1
    op2, pos2 = oppos2

    if len(op1.targets) == 0 and len(op2.targets) == 0:
        return pos1 < pos2

    if len(op1.targets) == 0:
        return True
    if len(op2.targets) == 0:
        return False
    if len(set(op1.targets) & set(op2.targets)) != 0:
        return pos1 < pos2

    return min(op1.targets) > min(op2.targets)

def pseudo_sort_operations_ascending(operations: list[GateOperation]):
    sorted_operations = operations.copy()
    for i in range(1, len(operations)):
        op1 = sorted_operations[i]

        j = i-1
        while j >= 0:
            op2 = sorted_operations[j]
            if pseudo_lt((op1, i), (op2, j)):
                sorted_operations[j+1] = sorted_operations[j]
            else:
                break
            j -= 1
        if i != j+1:
            sorted_operations[j+1] = op1
    return sorted_operations


def pseudo_sort_operations_descending(operations: list[GateOperation]):
    sorted_operations = operations.copy()
    for i in range(1, len(operations)):
        op1 = sorted_operations[i]

        j = i-1
        while j >= 0:
            op2 = sorted_operations[j]
            if pseudo_gt((op1, i), (op2, j)):
                sorted_operations[j+1] = sorted_operations[j]
            else:
                break
            j -= 1
        if i != j+1:
            sorted_operations[j+1] = op1
    return sorted_operations


def select_new_permutation(operations: list[GateOperation],
                           qubit_count: int,
                           local_qubit_count: int,
                           perm: list[int]) -> list[list[GateOperation]]:
    local_qubits = perm[:local_qubit_count]
    nonlocal_qubits = perm[local_qubit_count:]

    candidates = local_qubits[len(nonlocal_qubits):] + \
        nonlocal_qubits + local_qubits[:len(nonlocal_qubits)]
    return candidates


def transpile_no_offload_cpu(operations: list[GateOperation],
                             qubit_count: int,
                             local_qubit_count: int):
    return operations

def transpile_no_offload_cuda(operations: list[GateOperation],
                              qubit_count: int,
                              local_qubit_count: int):
    return operations

def transpile_no_offload_hybrid(operations: list[GateOperation],
                                qubit_count: int,
                                local_qubit_count: int):
    assert local_qubit_count <= qubit_count
    operations = pseudo_sort_operations_descending(operations)

    accumulating_local = True
    subcircuits = []
    current_operations = []

    for i, op in enumerate(operations):
        is_local_gate = (len(op.targets) == 0 or min(
            op.targets) >= (qubit_count-local_qubit_count))
        if accumulating_local == is_local_gate:
            current_operations.append(op)
        else:
            if len(current_operations) != 0:
                subcircuits.append(current_operations)
            current_operations = [op]
            accumulating_local = not accumulating_local

    if len(current_operations) != 0:
        subcircuits.append(current_operations)

    slice_count = 2**(qubit_count - local_qubit_count)
    return [[subcircuit] * slice_count for subcircuit in subcircuits]

def transpile_no_offload(operations: list[GateOperation],
                         qubit_count: int,
                         local_qubit_count: int,
                         device: DeviceType):
    match device:
        case DeviceType.CPU:
            return transpile_no_offload_cpu(operations,
                                            qubit_count,
                                            local_qubit_count)
        case DeviceType.CUDA:
            return transpile_no_offload_cuda(operations,
                                             qubit_count,
                                             local_qubit_count)
        case DeviceType.HYBRID:
            return transpile_no_offload_hybrid(operations,
                                               qubit_count,
                                               local_qubit_count)
        case _:
            raise NotImplementedError("Not Implemented")

def transpile_cpu_offload_cpu(operations: list[GateOperation],
                              qubit_count: int,
                              local_qubit_count: int):
    return transpile_no_offload_cpu(operations, qubit_count, local_qubit_count)

def transpile_cpu_offload_cuda(operations: list[GateOperation],
                               qubit_count: int,
                               local_qubit_count: int):
    raise NotImplementedError("Not Implemented")

def transpile_cpu_offload_hybrid(operations: list[GateOperation],
                                 qubit_count: int,
                                 local_qubit_count: int):
    return transpile_no_offload_hybrid(operations, qubit_count, local_qubit_count)

def transpile_cpu_offload(operations: list[GateOperation],
                          qubit_count: int,
                          local_qubit_count: int,
                          device: DeviceType):
    match device:
        case DeviceType.CPU:
            return transpile_cpu_offload_cpu(operations, qubit_count, local_qubit_count)
        case DeviceType.CUDA:
            return transpile_cpu_offload_cuda(operations, qubit_count, local_qubit_count)
        case DeviceType.HYBRID:
            return transpile_cpu_offload_hybrid(operations, qubit_count, local_qubit_count)
        case _:
            raise NotImplementedError("Not Implemented")

def transpile_storage_offload(operations: list[GateOperation],
                              qubit_count: int,
                              local_qubit_count: int,
                              device: DeviceType,
                              path: Optional[list[str]] = None):
    raise NotImplementedError("Not Implemented")

def transpile(operations: list[GateOperation],
              qubit_count: int,
              local_qubit_count: int,
              device: DeviceType,
              offload: OffloadType,
              path: Optional[list[str]] = None):
    match offload:
        case OffloadType.NONE:
            return transpile_no_offload(operations,
                                        qubit_count,
                                        local_qubit_count,
                                        device)
        case OffloadType.CPU:
            return transpile_cpu_offload(operations,
                                         qubit_count,
                                         local_qubit_count,
                                         device)
        case OffloadType.STORAGE:
            return transpile_storage_offload(operations,
                                             qubit_count,
                                             local_qubit_count,
                                             device,
                                             path)
        case _:
            raise NotImplementedError("Not Implemented")
