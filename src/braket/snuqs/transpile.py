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

def pseudo_sort_elevator_descending(operations: list[GateOperation],
                                    slice_qubit_count: int):
    sort_functions = [pseudo_sort_operations_ascending,
                      pseudo_sort_operations_descending]
    sort_index = 0
    operations = pseudo_sort_operations_descending(operations)
    op = operations[0]
    accumulating_local = (len(op.targets) == 0 or min(
        op.targets) >= slice_qubit_count)
    for i in range(len(operations)):
        op = operations[i]
        is_local_gate = (len(op.targets) == 0 or min(
            op.targets) >= slice_qubit_count)
        if accumulating_local != is_local_gate:
            operations[i+1:] = sort_functions[sort_index](operations[i+1:])
            sort_index = (sort_index + 1) % 2
            accumulating_local = not accumulating_local

    return operations


def select_permutation_heuristic(idx: int,
                                 operations: list[GateOperation],
                                 qubit_count: int,
                                 local_qubit_count: int) -> list[int]:
    counts = [0 for i in range(qubit_count)]
    for op in operations[idx:]:
        for t in op.targets:
            counts[t] += 1

    for t in operations[idx].targets:
        counts[t] = len(operations)

    local_qubits = [(i, counts[i])
                    for i in range(qubit_count-local_qubit_count, qubit_count)]
    nonlocal_qubits = [(i, counts[i])
                       for i in range(qubit_count-local_qubit_count)]

    local_qubits = [p for p, c in sorted(local_qubits, key=lambda x: x[1])]
    nonlocal_qubits = [p for p, c in sorted(
        nonlocal_qubits, key=lambda x: x[1])]
    new_perm = local_qubits + nonlocal_qubits

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
                            local_qubit_count: int,
                            slice_qubit_count: int) -> list[GateOperation]:
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


def transpile_no_offload_cpu(subcircuit: Subcircuit):
    subcircuit.operations = pseudo_sort_operations_descending(
        subcircuit.operations)
    return subcircuit

def transpile_no_offload_cuda(subcircuit: Subcircuit):
    subcircuit.operations = pseudo_sort_operations_descending(
        subcircuit.operations)
    return subcircuit

def rearrange_targets(operations: list[GateOperation], new_perm: list[int]):
    new_perm_inverse = {q: i for i, q in enumerate(new_perm)}
    for op in operations:
        op.targets = [new_perm_inverse[t] for t in op.targets]
    return operations

def localize_operations(operations: list[GateOperation],
                        qubit_count: int,
                        local_qubit_count: int,
                        slice_qubit_count: int,
                        ):
    cleanup_operations = []
    slice_qubit_count = qubit_count - local_qubit_count

    localized_operations = []
    operations = pseudo_sort_operations_descending(operations)
    for i in range(len(operations)):
        op = operations[i]
        is_local_gate = (op.sliceable() or len(op.targets) == 0 or min(
            op.targets) >= slice_qubit_count)

        if not is_local_gate:
            new_perm = select_permutation(
                i, operations, qubit_count, local_qubit_count)
            perm_gates = build_permutation_gates(
                new_perm, qubit_count, local_qubit_count, slice_qubit_count)
            localized_operations += perm_gates
            cleanup_operations = list(
                reversed(perm_gates)) + cleanup_operations

            operations[i:] = rearrange_targets(operations[i:], new_perm)
            operations[i +
                       1:] = pseudo_sort_operations_descending(operations[i+1:])

        localized_operations.append(op)

    localized_operations += cleanup_operations
    return localized_operations

def optimize_operations(operations: list[GateOperation]):
    changed = True
    while changed:
        optimized_operations = []
        changed = False

        i = 0
        while i < len(operations):
            if i+1 < len(operations) and operations[i] == operations[i+1]:
                changed = True
                i += 2
            else:
                optimized_operations.append(operations[i])
                i += 1
        operations = optimized_operations

    return operations

def partition_operations(operations: list[GateOperation],
                         qubit_count: int,
                         local_qubit_count: int):
    slice_qubit_count = qubit_count - local_qubit_count
    partitioned_operations = []
    current_operations = []
    op = operations[0]
    accumulating_local = (len(op.targets) == 0 or min(
        op.targets) >= slice_qubit_count)
    for i in range(len(operations)):
        op = operations[i]
        is_local_gate = (len(op.targets) == 0 or min(
            op.targets) >= slice_qubit_count)
        if accumulating_local == is_local_gate:
            current_operations.append(op)
        else:
            if len(current_operations) != 0:
                partitioned_operations.append(current_operations)
                current_operations = []
            current_operations = [op]
            accumulating_local = not accumulating_local

    if len(current_operations) != 0:
        partitioned_operations.append(current_operations)

    sliced_operations = []
    for ops in partitioned_operations:
        is_local_gate = (len(ops[0].targets) == 0 or min(
            ops[0].targets) >= slice_qubit_count)
        if is_local_gate:
            sliced_operations.append([ops] * (2 ** slice_qubit_count))
        else:
            sliced_operations.append(ops)

    return sliced_operations


def transpile_no_offload_hybrid(subcircuit: Subcircuit):
    subcircuit.operations = pseudo_sort_operations_descending(
        subcircuit.operations)
    subcircuit.operations = optimize_operations(subcircuit.operations)
    subcircuit.operations = partition_operations(subcircuit.operations,
                                                 subcircuit.qubit_count,
                                                 subcircuit.max_qubit_count_cuda)
    return subcircuit

def transpile_cpu_offload_cpu(subcircuit: Subcircuit):
    return transpile_no_offload_cpu(subcircuit)

def transpile_cpu_offload_cuda(subcircuit: Subcircuit):
    subcircuit.operations = localize_operations(subcircuit.operations,
                                                subcircuit.qubit_count,
                                                subcircuit.max_qubit_count_cuda,
                                                subcircuit.slice_qubit_count)
    subcircuit.operations = optimize_operations(subcircuit.operations)
    subcircuit.operations = partition_operations(subcircuit.operations,
                                                 subcircuit.qubit_count,
                                                 subcircuit.max_qubit_count_cuda)
    return subcircuit

def transpile_cpu_offload_hybrid(subcircuit: Subcircuit):
    return transpile_no_offload_hybrid(subcircuit)

def transpile_no_offload(subcircuit: Subcircuit):
    match subcircuit.accelerator:
        case AcceleratorType.CPU:
            return transpile_no_offload_cpu(subcircuit)
        case AcceleratorType.CUDA:
            return transpile_no_offload_cuda(subcircuit)
        case AcceleratorType.HYBRID:
            return transpile_no_offload_hybrid(subcircuit)
        case _:
            raise NotImplementedError("Not Implemented")

def transpile_cpu_offload(subcircuit: Subcircuit):
    match subcircuit.accelerator:
        case AcceleratorType.CPU:
            return transpile_cpu_offload_cpu(subcircuit)
        case AcceleratorType.CUDA:
            return transpile_cpu_offload_cuda(subcircuit)
        case AcceleratorType.HYBRID:
            return transpile_cpu_offload_hybrid(subcircuit)
        case _:
            raise NotImplementedError("Not Implemented")

def transpile_storage_offload(subcircuit: Subcircuit):
    match subcircuit.accelerator:
        case AcceleratorType.CPU:
            return transpile_storage_offload_cpu(subcircuit)
        case AcceleratorType.CUDA:
            return transpile_storage_offload_cuda(subcircuit)
        case AcceleratorType.HYBRID:
            return transpile_storage_offload_hybrid(subcircuit)
        case _:
            raise NotImplementedError("Not Implemented")

def transpile_storage_offload_cpu(subcircuit: Subcircuit):
    subcircuit.operations = localize_operations(subcircuit.operations,
                                                subcircuit.qubit_count,
                                                subcircuit.max_qubit_count,
                                                subcircuit.slice_qubit_count)
    subcircuit.operations = optimize_operations(subcircuit.operations)
    subcircuit.operations = partition_operations(subcircuit.operations,
                                                 subcircuit.qubit_count,
                                                 subcircuit.max_qubit_count)
    return subcircuit

def transpile_storage_offload_cuda(subcircuit: Subcircuit):
    subcircuit.operations = localize_operations(subcircuit.operations,
                                                subcircuit.qubit_count,
                                                subcircuit.max_qubit_count_cuda,
                                                subcircuit.slice_qubit_count)
    subcircuit.operations = optimize_operations(subcircuit.operations)
    subcircuit.operations = partition_operations(subcircuit.operations,
                                                 subcircuit.qubit_count,
                                                 subcircuit.max_qubit_count_cuda)
    return subcircuit

def transpile_storage_offload_hybrid(subcircuit: Subcircuit):
    subcircuit.operations = localize_operations(subcircuit.operations,
                                                subcircuit.qubit_count,
                                                subcircuit.max_qubit_count,
                                                subcircuit.slice_qubit_count)
    subcircuit.operations = optimize_operations(subcircuit.operations)
    subcircuit.operations = partition_operations(subcircuit.operations,
                                                 subcircuit.qubit_count,
                                                 subcircuit.max_qubit_count)

    for s in range(len(subcircuit.operations)):
        partitioned_subcircuit = subcircuit.operations[s]
        if isinstance(partitioned_subcircuit[0], list):
            for i in range(len(partitioned_subcircuit)):
                sliced_subcircuit = partitioned_subcircuit[i]
                sliced_subcircuit = pseudo_sort_operations_descending(
                    sliced_subcircuit)
                sliced_subcircuit = optimize_operations(sliced_subcircuit)
                sliced_subcircuit = partition_operations(subcircuit.operations,
                                                         subcircuit.qubit_count,
                                                         subcircuit.max_qubit_count_cuda)
                partitioned_subcircuit[i] = sliced_subcircuit

            subcircuit.operations[s] = partitioned_subcircuit

    print(subcircuit)
    return subcircuit

def compute_max_qubit_count(qubit_count: int, prefetch: PrefetchType):
    free, _ = mem_info()
    max_count = int(math.log(free / 16, 2))
    if prefetch != PrefetchType.NONE:
        max_count -= 1
    return min(qubit_count, max_count)

def compute_max_qubit_count_cuda(qubit_count: int, prefetch: PrefetchType):
    free, _ = mem_info_cuda()
    max_count = int(math.log(free / 16, 2))
    if prefetch != PrefetchType.NONE:
        max_count -= 1
    return min(qubit_count, max_count)


def transpile(
        qubit_count: int,
        operations: list[GateOperation],
        accelerator: AcceleratorType,
        prefetch: PrefetchType,
        offload: OffloadType) -> Subcircuit:
    max_qubit_count = compute_max_qubit_count(qubit_count, prefetch)
    max_qubit_count_cuda = compute_max_qubit_count_cuda(qubit_count, prefetch)
    subcircuit = Subcircuit(
        qubit_count=qubit_count,
        max_qubit_count=max_qubit_count,
        max_qubit_count_cuda=max_qubit_count_cuda,
        slice_qubit_count=max_qubit_count_cuda-1,
        accelerator=accelerator,
        prefetch=prefetch,
        offload=offload,
        operations=operations)

    match offload:
        case OffloadType.NONE:
            return transpile_no_offload(subcircuit)
        case OffloadType.CPU:
            return transpile_cpu_offload(subcircuit)
        case OffloadType.STORAGE:
            return transpile_storage_offload(subcircuit)
        case _:
            raise NotImplementedError("Not Implemented")

    return None
