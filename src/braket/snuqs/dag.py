from braket.snuqs._C.operation import GateOperation

class GateDAGNode:
    def __init__(self, _id, obj):
        self.obj = obj
        self.out_nodes = []
        self.in_nodes = []
        self.visited = False
        self.id = _id

    def __repr__(self):
        return f"<Node {self.id} {self.obj}>"

    def add_out_node(self, other):
        self.out_nodes.append(other)

    def add_in_node(self, other):
        self.in_nodes.append(other)


class GateDAG:
    def __init__(self, count: int, operations: list[GateOperation]):
        self.count = count
        self.operations = [GateDAGNode(i, op)
                           for i, op in enumerate(operations)]

        op_map = {i: None for i in range(count)}
        for op in self.operations:
            for t in op.obj.targets:
                if op_map[t] is not None:
                    op_map[t].add_out_node(op)
                    op.add_in_node(op_map[t])
                op_map[t] = op

    def _DFS(self, op, ops, before, after):
        if before:
            before(op)

        op.visited = True
        for _op in op.out_nodes:
            if not _op.visited:
                self._DFS(_op, ops, before, after)

        if after:
            after(op)

        ops.append(op)

    def _DFS_reversed(self, op, ops, before, after):
        if before:
            before(op)

        op.visited = True
        for _op in op.in_nodes:
            if not _op.visited:
                self._DFS_reversed(_op, ops, before, after)

        if after:
            after(op)

        ops.append(op)

    def topological_sort(self, before, after):
        for op in self.operations:
            op.visited = False

        ops = []
        for op in self.operations:
            if not op.visited:
                self._DFS(op, ops, before, after)

        return [op.obj for op in reversed(ops)]
