from braket.snuqs._C.operation import GateOperation

class GateDAGNode:
    def __init__(self, _id, obj):
        self.obj = obj
        self.adj = []
        self.adj_reversed = []
        self.visited = False
        self.id = _id
        self.call_id = -1

    def __repr__(self):
        return f"<Node {self.id} {self.obj}> {self.call_id}"

    def add_link(self, other):
        self.adj.append(other)

    def add_link_reversed(self, other):
        self.adj_reversed.append(other)


class GateDAG:
    def __init__(self, count: int, operations: list[GateOperation]):
        self.count = count
        self.operations = [GateDAGNode(i, op)
                           for i, op in enumerate(operations)]
        print(self.operations)
        op_map = {i: None for i in range(count)}
        for op in self.operations:
            for t in op.obj.targets:
                if op_map[t] is not None:
                    op_map[t].add_link(op)
                    op.add_link_reversed(op_map[t])
                op_map[t] = op

    def DFS(self, before, after):
        for op in self.operations:
            op.visited = False

        for op in self.operations:
            if not op.visited:
                self._DFS(op, before, after, 0)

    def _DFS(self, op, before, after, call_id):
        op.visited = True
        op.call_id = call_id
        if before:
            before(op)

        call_id += 1
        for op in op.adj:
            if len(op.adj_reversed) == 0 and not op.visited:
                call_id = self._DFS(op, before, after, call_id)

        if after:
            after(op)
        return call_id

    def _DFS_reversed(self, op, before, after, call_id):
        op.visited = True
        op.call_id = call_id
        if before:
            before(op)

        for op in op.adj_reversed:
            if not op.visited:
                call_id = self._DFS(op, before, after, call_id+1)

        if after:
            after(op)
        return call_id

    def topological_sort(self, before, after):
        for op in self.operations:
            op.visited = False

        for op in self.operations:
            if len(op.adj) == 0 and not op.visited:
                print(f"Exploring {op}")
                self._DFS_reversed(op, before, after, 0)
