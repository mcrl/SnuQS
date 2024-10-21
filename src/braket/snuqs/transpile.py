from braket.snuqs._C.operation import GateOperation
from braket.snuqs.types import AcceleratorType, PrefetchType, OffloadType
from braket.snuqs._C.operation.gate_operations import (
    Swap,
)
from braket.snuqs._C.core import mem_info
from braket.snuqs._C.core.cuda import mem_info as mem_info_cuda
from braket.snuqs.subcircuit import Subcircuit
import math


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

    if op1.sliceable() and op2.sliceable():
        return pos1 < pos2
    if op1.sliceable():
        return True
    if op2.sliceable():
        return False

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

    if op1.sliceable() and op2.sliceable():
        return pos1 < pos2
    if op1.sliceable():
        return True
    if op2.sliceable():
        return False

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
    slice_qubit_count = qubit_count - local_qubit_count

    accumulating_local = True
    subcircuits = []
    current_operations = []

    operations = pseudo_sort_operations_descending(operations)
    for i in range(len(operations)):
        op = operations[i]
        is_local_gate = (len(op.targets) == 0 or min(
            op.targets) >= slice_qubit_count)
        if accumulating_local == is_local_gate:
            current_operations.append(op)
        else:
            if len(current_operations) != 0:
                subcircuits.append(current_operations)
                current_operations = []
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
        current_operations = []

    slice_count = 2**(qubit_count - local_qubit_count)
    return [[subcircuit] * slice_count for subcircuit in subcircuits]

def transpile_cpu_offload_cpu(operations: list[GateOperation],
                              qubit_count: int,
                              local_qubit_count: int):
    return transpile_no_offload_cpu(operations, qubit_count, local_qubit_count)

def transpile_cpu_offload_cuda(operations: list[GateOperation],
                               qubit_count: int,
                               local_qubit_count: int):
    assert local_qubit_count <= qubit_count
    slice_qubit_count = qubit_count - local_qubit_count
    subcircuits = []
    current_operations = []
    cleanup_operations = []
    operations = pseudo_sort_operations_descending(operations)

    print("AA", operations)
    for i in range(len(operations)):
        op = operations[i]
        is_local_gate = (op.sliceable() or len(op.targets) == 0 or min(
            op.targets) >= slice_qubit_count)

        if is_local_gate:
            current_operations.append(op)
        else:
            new_perm = select_permutation(
                i, operations, qubit_count, local_qubit_count)

            assert all(t in new_perm[qubit_count-local_qubit_count:]
                       for t in op.targets)
            perm_gates = build_permutation_gates(
                new_perm, qubit_count, local_qubit_count)
            cleanup_operations += reversed(perm_gates)

            accumulating_local = True
            for p in perm_gates:
                is_local_gate = min(p.targets) >= slice_qubit_count
                if accumulating_local == is_local_gate:
                    current_operations.append(p)
                else:
                    if len(current_operations) != 0:
                        subcircuits.append(current_operations)
                    current_operations = [p]
                    accumulating_local = not accumulating_local

            new_perm_inverse = {q: i for i, q in enumerate(new_perm)}
            for j in range(i, len(operations)):
                _op = operations[j]
                _op.targets = [new_perm_inverse[t] for t in _op.targets]
                operations[j] = _op

            current_operations.append(operations[i])
            operations[i +
                       1:] = pseudo_sort_operations_descending(operations[i+1:])

            if len(current_operations) != 0:
                subcircuits.append(current_operations)
                current_operations = []

    if len(current_operations) != 0:
        subcircuits.append(current_operations)

    current_operations = []
    accumulating_local = True
    for p in cleanup_operations:
        is_local_gate = min(p.targets) >= slice_qubit_count
        if accumulating_local == is_local_gate:
            current_operations.append(p)
        else:
            if len(current_operations) != 0:
                subcircuits.append(current_operations)
            current_operations = [p]
            accumulating_local = not accumulating_local

    if len(current_operations) != 0:
        subcircuits.append(current_operations)

    print(subcircuits)
    slice_count = 2**(qubit_count - local_qubit_count)
    return [[subcircuit] * slice_count for subcircuit in subcircuits]

def transpile_cpu_offload_hybrid(operations: list[GateOperation],
                                 qubit_count: int,
                                 local_qubit_count: int):
    return transpile_no_offload_hybrid(operations,
                                       qubit_count,
                                       local_qubit_count)

def transpile_no_offload(operations: list[GateOperation],
                         qubit_count: int,
                         max_qubit_count: int,
                         local_qubit_count: int,
                         accelerator: AcceleratorType):
    match accelerator:
        case AcceleratorType.CPU:
            return transpile_no_offload_cpu(operations,
                                            qubit_count,
                                            local_qubit_count)
        case AcceleratorType.CUDA:
            return transpile_no_offload_cuda(operations,
                                             qubit_count,
                                             local_qubit_count)
        case AcceleratorType.HYBRID:
            return transpile_no_offload_hybrid(operations,
                                               qubit_count,
                                               local_qubit_count)
        case _:
            raise NotImplementedError("Not Implemented")

def transpile_cpu_offload(operations: list[GateOperation],
                          qubit_count: int,
                          max_qubit_count: int,
                          local_qubit_count: int,
                          accelerator: AcceleratorType):
    match accelerator:
        case AcceleratorType.CPU:
            return transpile_cpu_offload_cpu(operations,
                                             qubit_count,
                                             local_qubit_count)
        case AcceleratorType.CUDA:
            return transpile_cpu_offload_cuda(operations,
                                              qubit_count,
                                              local_qubit_count)
        case AcceleratorType.HYBRID:
            return transpile_cpu_offload_hybrid(operations,
                                                qubit_count,
                                                local_qubit_count)
        case _:
            raise NotImplementedError("Not Implemented")

def transpile_storage_offload_cpu(operations: list[GateOperation],
                                  qubit_count: int,
                                  max_qubit_count: int,
                                  local_qubit_count: int):
    assert max_qubit_count <= qubit_count
    assert local_qubit_count <= max_qubit_count

    slice_qubit_count = qubit_count - max_qubit_count
    subcircuits = []
    current_operations = []
    cleanup_operations = []
    operations = pseudo_sort_operations_descending(operations)

    print(operations)
    for i in range(len(operations)):
        op = operations[i]
        is_local_gate = (op.sliceable() or len(op.targets) == 0 or min(
            op.targets) >= slice_qubit_count)

        if is_local_gate:
            current_operations.append(op)
        else:
            new_perm = select_permutation(
                i, operations, qubit_count, max_qubit_count)

            assert all(t in new_perm[qubit_count-max_qubit_count:]
                       for t in op.targets)
            perm_gates = build_permutation_gates(
                new_perm, qubit_count, max_qubit_count)
            cleanup_operations += reversed(perm_gates)

            accumulating_local = True
            for p in perm_gates:
                is_local_gate = min(p.targets) >= slice_qubit_count
                if accumulating_local == is_local_gate:
                    current_operations.append(p)
                else:
                    if len(current_operations) != 0:
                        subcircuits.append(current_operations)
                    current_operations = [p]
                    accumulating_local = not accumulating_local

            new_perm_inverse = {q: i for i, q in enumerate(new_perm)}
            for j in range(i, len(operations)):
                _op = operations[j]
                _op.targets = [new_perm_inverse[t] for t in _op.targets]
                operations[j] = _op

            current_operations.append(operations[i])
            operations[i +
                       1:] = pseudo_sort_operations_descending(operations[i+1:])

            if len(current_operations) != 0:
                subcircuits.append(current_operations)
                current_operations = []

    if len(current_operations) != 0:
        subcircuits.append(current_operations)

    current_operations = []
    accumulating_local = True
    for p in cleanup_operations:
        is_local_gate = min(p.targets) >= slice_qubit_count
        if accumulating_local == is_local_gate:
            current_operations.append(p)
        else:
            if len(current_operations) != 0:
                subcircuits.append(current_operations)
            current_operations = [p]
            accumulating_local = not accumulating_local

    if len(current_operations) != 0:
        subcircuits.append(current_operations)

    for subcircuit in subcircuits:
        for op in subcircuit:
            pass

    print(subcircuits)

    slice_count = 2**(qubit_count - max_qubit_count)
    return [[subcircuit] * slice_count for subcircuit in subcircuits]

def transpile_storage_offload_cuda(operations: list[GateOperation],
                                   qubit_count: int,
                                   max_qubit_count: int,
                                   local_qubit_count: int):
    raise NotImplementedError("Not Implemented")

def transpile_storage_offload_hybrid(operations: list[GateOperation],
                                     qubit_count: int,
                                     local_qubit_count: int,
                                     max_qubit_count: int):
    raise NotImplementedError("Not Implemented")

def transpile_storage_offload(operations: list[GateOperation],
                              qubit_count: int,
                              max_qubit_count: int,
                              local_qubit_count: int,
                              accelerator: AcceleratorType):
    match accelerator:
        case AcceleratorType.CPU:
            return transpile_storage_offload_cpu(operations,
                                                 qubit_count,
                                                 max_qubit_count,
                                                 local_qubit_count)
        case AcceleratorType.CUDA:
            return transpile_storage_offload_cuda(operations,
                                                  qubit_count,
                                                  max_qubit_count,
                                                  local_qubit_count)
        case AcceleratorType.HYBRID:
            return transpile_storage_offload_hybrid(operations,
                                                    qubit_count,
                                                    max_qubit_count,
                                                    local_qubit_count)
        case _:
            raise NotImplementedError("Not Implemented")

def compute_max_qubit_count(qubit_count: int, prefetch: PrefetchType):
    free, _ = mem_info()
    max_count = int(math.log(free / 16, 2))
    if prefetch != PrefetchType.NONE:
        max_count //= 2
    return min(qubit_count, max_count)

def compute_max_qubit_count_cuda(qubit_count: int, prefetch: PrefetchType):
    free, _ = mem_info_cuda()
    max_count = int(math.log(free / 16, 2))
    if prefetch != PrefetchType.NONE:
        max_count //= 2
    return min(qubit_count, max_count)


def transpile(
        qubit_count: int,
        operations: list[GateOperation],
        accelerator: AcceleratorType,
        prefetch: PrefetchType,
        offload: OffloadType) -> Subcircuit:
    max_qubit_count = compute_max_qubit_count(qubit_count, prefetch)
    max_qubit_count_cuda = compute_max_qubit_count_cuda(qubit_count, prefetch)

    match offload:
        case OffloadType.NONE:
            operations = transpile_no_offload(operations,
                                              qubit_count,
                                              max_qubit_count,
                                              max_qubit_count_cuda,
                                              accelerator)
        case OffloadType.CPU:
            operations = transpile_cpu_offload(operations,
                                               qubit_count,
                                               max_qubit_count,
                                               max_qubit_count_cuda,
                                               accelerator)
        case OffloadType.STORAGE:
            operations = transpile_storage_offload(operations,
                                                   qubit_count,
                                                   max_qubit_count,
                                                   max_qubit_count_cuda,
                                                   accelerator)
        case _:
            raise NotImplementedError("Not Implemented")

    return Subcircuit(
        qubit_count=qubit_count,
        max_qubit_count=max_qubit_count,
        max_qubit_count_cuda=max_qubit_count_cuda,
        accelerator=accelerator,
        offload=offload,
        operations=operations)
