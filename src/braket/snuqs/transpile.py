from braket.snuqs._C.operation import GateOperation
from functools import cmp_to_key

def sort_operations(operations: list[GateOperation]):
    def compare_gates(x, y):
        if len(x.targets) == 0 and len(y.targets) == 0:
            return 0
        elif len(x.targets) == 0:
            return -1
        elif len(y.targets) == 0:
            return 1

        if len(set(x.targets) & set(y.targets)) != 0:
            return 0

        if max(y.targets) < max(x.targets):
            return 1

        return 0

    return sorted(operations, key=cmp_to_key(compare_gates))


def select_new_permutation(operations: list[GateOperation],
                           qubit_count: int,
                           local_qubit_count: int,
                           perm: list[int]) -> list[list[GateOperation]]:
    local_qubits = perm[:local_qubit_count]
    nonlocal_qubits = perm[local_qubit_count:]

    candidates = local_qubits[len(nonlocal_qubits):] + \
        nonlocal_qubits + local_qubits[:len(nonlocal_qubits)]
    return candidates
    # return list(reversed(candidates))


def subcircuit_partition(operations: list[GateOperation], qubit_count: int, local_qubit_count: int) -> list[list[GateOperation]]:
    assert local_qubit_count <= qubit_count
    slice_count = 2**(qubit_count-local_qubit_count)
    operations = sort_operations(operations)

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

def subcircuit_partition_hybrid(operations: list[GateOperation], qubit_count: int, local_qubit_count: int) -> list[list[GateOperation]]:
    assert local_qubit_count <= qubit_count
    operations = sort_operations(operations)

    accumulating_local = True
    subcircuits = []
    current_operations = []

    for i, op in enumerate(operations):
        is_local_gate = (len(op.targets) == 0 or max(
            op.targets) < local_qubit_count)
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
