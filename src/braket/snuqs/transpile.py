from braket.snuqs._C.operation import GateOperation
from functools import cmp_to_key

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
            if len(current_operations) != 0:
                subcircuits.append(current_operations)
            current_operations = [op]
            accumulating_local = not accumulating_local

    if len(current_operations) != 0:
        subcircuits.append(current_operations)
    return subcircuits


def transpile_for_hybrid(operations: list[GateOperation],
                         qubit_count: int,
                         local_qubit_count: int) -> list[list[GateOperation]]:
    assert local_qubit_count <= qubit_count
    subcircuits = subcircuit_partition_hybrid(operations,
                                              qubit_count,
                                              local_qubit_count)
    print(subcircuits)
    print(subcircuits[0])

    slice_count = 2**(qubit_count - local_qubit_count)
    return [[subcircuit] * slice_count for subcircuit in subcircuits]
