from braket.snuqs._C.operation import GateOperation
import numpy as np
from braket.snuqs.device import DeviceType
from braket.snuqs.offload import OffloadType
from typing import Optional
from braket.snuqs._C.operation.gate_operations import (
    Swap,
)
from braket.snuqs.dag import GateDAG

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


def select_permutation_heuristic(idx: int,
                                 operations: list[GateOperation],
                                 qubit_count: int,
                                 local_qubit_count: int) -> list[int]:
    counts = [0 for i in range(qubit_count)]
    for op in operations[idx+1:]:
        for t in op.targets:
            counts[t] += 1

    local_qubits = [(i, counts[i])
                    for i in range(qubit_count-local_qubit_count, qubit_count)]
    nonlocal_qubits = [(i, counts[i])
                       for i in range(qubit_count-local_qubit_count)]

    local_qubits = [p for p, c in sorted(local_qubits, key=lambda x: x[1])]
    nonlocal_qubits = [p for p, c in sorted(
        nonlocal_qubits, key=lambda x: x[1])]
    new_perm = local_qubits + nonlocal_qubits

    print(f"new permutation -> {new_perm}")
    assert all(t in new_perm[qubit_count-local_qubit_count:]
               for t in operations[idx].targets)
    return new_perm

def select_permutation(idx: int,
                       operations: list[GateOperation],
                       qubit_count: int,
                       local_qubit_count: int) -> list[int]:
    return select_permutation_heuristic(idx,
                                        operations,
                                        qubit_count,
                                        local_qubit_count)

def build_permutation_gates(new_perm: list[int],
                            qubit_count: int,
                            local_qubit_count: int) -> list[GateOperation]:
    perm_gates = []
    perm_map = {i: i for i in range(qubit_count)}
    new_perm_map = {q: i for i, q, in enumerate(new_perm)}

    for q in range(qubit_count):
        i = perm_map[q]
        j = new_perm_map[q]
        if i != j:
            for _q, _i in perm_map.items():
                if _i == j:
                    perm_map[_q] = i
                    break
            perm_map[q] = j
            perm_gates.append(Swap([i, j]))

    slice_qubit_count = qubit_count - local_qubit_count
    reordered_gates = []
    for i in range(len(perm_gates)):
        p = perm_gates[i]
        l, h = sorted(p.targets)
        if l < slice_qubit_count and h >= slice_qubit_count:
            if h != slice_qubit_count:
                reordered_gates.append(Swap([h, slice_qubit_count]))
            p.targets = [l, slice_qubit_count]
            reordered_gates.append(p)
            if h != slice_qubit_count:
                reordered_gates.append(Swap([h, slice_qubit_count]))
        else:
            reordered_gates.append(p)

    return reordered_gates


#    permutation_gates = []
#    slice_qubit_count = qubit_count-local_qubit_count
#    perm_local_qubits = list(range(slice_qubit_count, qubit_count))
#    perm_nonlocal_qubits = list(range(slice_qubit_count))
#    new_perm_local_qubits = list(range(slice_qubit_count, qubit_count))
#    new_perm_nonlocal_qubits = list(range(slice_qubit_count))
#
#    from_local_to_nonlocal = list(set(
#        perm_local_qubits) & set(new_perm_nonlocal_qubits))
#    from_nonlocal_to_local = list(set(
#        perm_nonlocal_qubits) & set(new_perm_local_qubits))
#    assert (len(from_local_to_nonlocal) == len(from_nonlocal_to_local))
#
#    interm_perm = list(range(qubit_count))
#    dist = local_qubit_count-len(from_local_to_nonlocal)
#    for i in range(len(from_local_to_nonlocal)):
#        t0 = from_local_to_nonlocal[i]
#        t1 = dist+i
#        if t0 != t1:
#            permutation_gates.append(Swap([t0, t1]))
#            interm_perm[dist +
#                        i], interm_perm[i] = interm_perm[i], interm_perm[dist+t]
#
#    # FIXME: All-to-all??
#    for i in range(len(from_local_to_nonlocal)):
#        t0 = dist+i
#        t1 = local_qubit_count+i
#        permutation_gates.append(Swap([t0, t1]))
#
#    for i in range(local_qubit_count):
#        permutation_gates.append(Swap([new_perm[i], interm_perm[i]]))
#        for j, q in enumerate(interm_perm):
#            if q == new_perm[i]:
#                interm_perm[i], interm_perm[j] = interm_perm[j], interm_perm[i]
#
#    return permutation_gates


def transpile_no_offload_cpu(operations: list[GateOperation],
                             qubit_count: int,
                             local_qubit_count: int):
    return pseudo_sort_operations_descending(operations)

def transpile_no_offload_cuda(operations: list[GateOperation],
                              qubit_count: int,
                              local_qubit_count: int):
    return pseudo_sort_operations_descending(operations)

def transpile_no_offload_hybrid(operations: list[GateOperation],
                                qubit_count: int,
                                local_qubit_count: int):
    assert local_qubit_count <= qubit_count

    accumulating_local = True
    subcircuits = []
    current_operations = []

    operations = pseudo_sort_operations_descending(operations)
    for i in range(len(operations)):
        op = operations[i]
        is_local_gate = (len(op.targets) == 0 or min(
            op.targets) >= (qubit_count-local_qubit_count))
        if accumulating_local == is_local_gate:
            current_operations.append(op)
        else:
            if len(current_operations) != 0:
                subcircuits.append(current_operations)
            if accumulating_local:
                operations[i +
                           1:] = pseudo_sort_operations_ascending(operations[i+1:])
            else:
                operations[i +
                           1:] = pseudo_sort_operations_descending(operations[i+1:])
            current_operations = [op]
            accumulating_local = not accumulating_local

    if len(current_operations) != 0:
        subcircuits.append(current_operations)

    slice_count = 2**(qubit_count - local_qubit_count)
    return [[subcircuit] * slice_count for subcircuit in subcircuits]

def transpile_cpu_offload_cpu(operations: list[GateOperation],
                              qubit_count: int,
                              local_qubit_count: int):
    return transpile_no_offload_cpu(operations, qubit_count, local_qubit_count)

def transpile_cpu_offload_cuda(operations: list[GateOperation],
                               qubit_count: int,
                               local_qubit_count: int):
    dag = GateDAG(qubit_count, operations)
    dag.topological_sort(
        lambda x: print(f"Before {x}"),
        None,
    )

    assert local_qubit_count <= qubit_count
    slice_qubit_count = qubit_count - local_qubit_count
    subcircuits = []
    current_operations = []

    cleanup_operations = []
    operations = pseudo_sort_operations_descending(operations)

    for i, op in enumerate(operations):
        is_local_gate = (len(op.targets) == 0 or min(
            op.targets) >= (slice_qubit_count))

        if is_local_gate:
            current_operations.append(op)
        else:
            if len(current_operations) != 0:
                subcircuits.append(current_operations)
            new_perm = select_permutation(
                i, operations, qubit_count, local_qubit_count)

            assert all(t in new_perm[qubit_count-local_qubit_count:]
                       for t in op.targets)
            current_operations = build_permutation_gates(
                new_perm, qubit_count, local_qubit_count)
            cleanup_operations += reversed(current_operations)
            new_perm_inverse = {q: i for i, q in enumerate(new_perm)}

            for _op in operations[i:]:
                _op.targets = [new_perm_inverse[t] for t in _op.targets]

            current_operations.append(op)
            operations = pseudo_sort_operations_descending(operations)

    if len(current_operations) != 0:
        subcircuits.append(current_operations)
    if len(cleanup_operations) != 0:
        subcircuits.append(cleanup_operations)

    return [[subcircuit] * (2**slice_qubit_count) for subcircuit in subcircuits]

def transpile_cpu_offload_hybrid(operations: list[GateOperation],
                                 qubit_count: int,
                                 local_qubit_count: int):
    return transpile_no_offload_hybrid(operations,
                                       qubit_count,
                                       local_qubit_count)

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

def transpile_cpu_offload(operations: list[GateOperation],
                          qubit_count: int,
                          local_qubit_count: int,
                          device: DeviceType):
    match device:
        case DeviceType.CPU:
            return transpile_cpu_offload_cpu(operations,
                                             qubit_count,
                                             local_qubit_count)
        case DeviceType.CUDA:
            return transpile_cpu_offload_cuda(operations,
                                              qubit_count,
                                              local_qubit_count)
        case DeviceType.HYBRID:
            return transpile_cpu_offload_hybrid(operations,
                                                qubit_count,
                                                local_qubit_count)
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
