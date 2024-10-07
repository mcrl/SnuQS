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


def transpile(operations: list[GateOperation]):
    for operation in operations:
        targets = operation.targets
        print(targets)
    return operations
