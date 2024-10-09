from braket.snuqs._C.operation import GateOperation
from functools import cmp_to_key

def compare_oppos_tuple(oppos1, oppos2):
    op1, pos1 = oppos1
    op2, pos2 = oppos1

    if len(op1) == 0 and len(op2) == 0:
        return -1 if pos1 < pos2 else 1

    if len(op1) == 0:
        return -1
    if len(op2) == 0:
        return 1

    for t1 in op1.targets:
        for t2 in op2.targets:
            if t1 == t2:
                return -1 if pos1 < pos2 else 1

def pseudo_sort_operations_ascending(operations: list[GateOperation]):
    oppos_tuples = [(op, i) for i, op in enumerate(operations)]
    return [op for op, pos in sorted(oppos_tuples, key=cmp_to_key(compare_oppos_tuple))]


def pseudo_sort_operations_descending(operations: list[GateOperation]):
    oppos_tuples = [(op, i) for i, op in enumerate(operations)]
    return [op for op, pos in sorted(oppos_tuples, key=cmp_to_key(compare_oppos_tuple), reverse=True)]


def select_new_permutation(operations: list[GateOperation],
                           qubit_count: int,
                           local_qubit_count: int,
                           perm: list[int]) -> list[list[GateOperation]]:
    local_qubits = perm[:local_qubit_count]
    nonlocal_qubits = perm[local_qubit_count:]

    candidates = local_qubits[len(nonlocal_qubits):] + \
        nonlocal_qubits + local_qubits[:len(nonlocal_qubits)]
    return candidates


def subcircuit_partition(operations: list[GateOperation],
                         qubit_count: int,
                         local_qubit_count: int) -> list[list[GateOperation]]:
    assert local_qubit_count <= qubit_count
    slice_count = 2**(qubit_count-local_qubit_count)
    operations = pseudo_sort_operations_descending(operations)

    print(operations)
    perm = list(range(qubit_count))
    local_perm = perm[:local_qubit_count]
    subcircuits = []
    current_operations = []
    for i, op in enumerate(operations):
        is_local_gate = (len(op.targets) == 0 or all(
            t in local_perm for t in op.targets))
        if is_local_gate:
            current_operations.append(op)
        else:
            subcircuits.append(current_operations)
            new_perm = select_new_permutation(
                operations[i:], qubit_count, local_qubit_count, perm)

            print(new_perm)
            assert (len(perm) == len(new_perm))
            qubit_map = {x: y for x, y in zip(perm, new_perm)}
            for k in range(i, len(operations)):
                operations[k].targets = [qubit_map[t]
                                         for t in operations[k].targets]
            local_perm = new_perm[:local_qubit_count]
            current_operations = [op]

    subcircuits.append(current_operations)
    print(subcircuits)
    return subcircuits

def subcircuit_partition_hybrid(operations: list[GateOperation],
                                qubit_count: int,
                                local_qubit_count: int) -> list[list[GateOperation]]:
    assert local_qubit_count <= qubit_count
#    print(operations)
#    operations = pseudo_sort_operations_descending(operations)
#    print(operations)

    accumulating_local = True
    subcircuits = []
    current_operations = []

    for i, op in enumerate(operations):
        is_local_gate = (len(op.targets) == 0 or min(
            op.targets) >= (qubit_count-local_qubit_count))
        if accumulating_local == is_local_gate:
            current_operations.append(op)
        else:
            subcircuits.append(current_operations)
            current_operations = [op]
            accumulating_local = not accumulating_local

    subcircuits.append(current_operations)
    return subcircuits


def transpile_for_hybrid(operations: list[GateOperation], qubit_count: int, local_qubit_count: int) -> list[list[GateOperation]]:
    assert local_qubit_count <= qubit_count
    subcircuits = subcircuit_partition_hybrid(operations,
                                              qubit_count,
                                              local_qubit_count)

    slice_count = 2**(qubit_count - local_qubit_count)
    return [[subcircuit] * slice_count for subcircuit in subcircuits]
