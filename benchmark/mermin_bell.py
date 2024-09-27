import numpy as np
import itertools

import stabilizers

from braket.aws import AwsDevice
from braket.devices import LocalSimulator
from braket.circuits import Gate, Circuit

# https://github.com/Infleqtion/client-superstaq/blob/main/supermarq-benchmarks/supermarq/benchmarks/mermin_bell.py

class MerminBell:
    """
    The Mermin-Bell benchmark is a test of a quantum computer's ability to exploit purely quantum
    phenomemna such as superposition and entanglement. It is based on the famous Bell-inequality
    tests of locality. Performance is based on a QPU's ability to prepare a GHZ state and measure
    the Mermin operator.
    """

    def __init__(self, num_qubits: int) -> None:
        """
        Initializes a `MerminBell`.

        Args:
            num_qubits: The number of qubits.
        """
        self.num_qubits = num_qubits
        self.mermin_operator = self._mermin_operator(num_qubits)
        self.stabilizer, self.pauli_basis = stabilizers.construct_stabilizer(
            self.num_qubits, self.mermin_operator
        )

    def mb_circuit(self) -> Circuit:
        """
        The Mermin-Bell circuit, simultaneously measuring Mermin terms in a GHZ circuit.

        Returns:
            The Mermin-Bell `cirq.Circuit`.
        """
        circuit = Circuit()

        circuit.rx(0, -np.pi / 2)
        for i in range(self.num_qubits - 1):
            circuit.cnot(i, i + 1)

        measurement_circuit = self._get_measurement_circuit(circuit).get_circuit()
        
        return circuit + measurement_circuit
    
    def _get_measurement_circuit(self, circuit: Circuit) -> stabilizers.MeasurementCircuit:
        """
        Return a MeasurementCircuit for simultaneous measurement of N operators.

        Each column of self.stabilizer represents a Pauli string that we seek to measure.
        Thus, self.stabilizer should have dimensions of 2 * N rows by N columns. The first N rows
        indicate the presence of a Z in each index of the Pauli String. The last N rows
        indicate X's.

        For instance, simultaneous measurement of YYI, XXY, IYZ would be represented by
        [[1, 0, 0],  ========
         [1, 0, 1],  Z matrix
         [0, 1, 1],  ========
         [1, 1, 0],  ========
         [1, 1, 1],  X matrix
         [0, 1, 0]   ========

        As annotated above, the submatrix of the first (last) N rows is referred to as
        the Z (X) matrix.

        All operators must commute and be independent (i.e. can't express any column as a base-2
        product of the other columns) for this code to work.
        """
        # Validate that the stabilizer matrix is valid
        assert self.stabilizer.shape == (
            2 * self.num_qubits,
            self.num_qubits,
        ), f"{self.num_qubits} qubits, but matrix shape: {self.stabilizer.shape}"

        # i, j will always denote row, column index
        for i in range(2 * self.num_qubits):
            for j in range(self.num_qubits):
                value = self.stabilizer[i, j]
                assert value in [0, 1], f"[{i}, {j}] index is {value}"

        measurement_circuit = stabilizers.MeasurementCircuit(
            Circuit(), self.stabilizer, self.num_qubits
        )

        stabilizers.prepare_X_matrix(measurement_circuit)
        stabilizers.row_reduce_X_matrix(measurement_circuit)
        stabilizers.patch_Z_matrix(measurement_circuit)
        stabilizers.change_X_to_Z_basis(measurement_circuit)
        
        # terminate with measurements
        for i in range(self.num_qubits):
            measurement_circuit.get_circuit().measure(i)

        return measurement_circuit

    def score(self, counts: dict[str, float]) -> float:
        """Compute the score for the N-qubit Mermin-Bell benchmark.

        This function assumes the regular big endian ordering of bitstring results.

        Args:
            counts: A dictionary containing the measurement counts from circuit execution.

        Returns:
            The score for the Mermin-Bell benchmark score.
        """

        # Store the conjugation rules for H, S, CX, CZ, SWAP in dictionaries. The keys are
        # the pauli strings to be conjugated and the values are the resulting pauli strings
        # after conjugation.
        # The typing here was added to satisfy mypy. Declaring this dict without the explicit
        # typing gets created as Dict[EigenGate, Dict[str, str]], but iterating through a
        # cirq.Circuit and passing op.gate as the key yields type Optional[Gate].
        conjugation_rules: dict[Circuit.Gate | None, dict[str, str]] = {
            "H": {"I": "I", "X": "Z", "Y": "-Y", "Z": "X"},
            "S": {"I": "I", "X": "Y", "Y": "-X", "Z": "Z"},
            "CNot": {
                "II": "II",
                "IX": "IX",
                "XI": "XX",
                "XX": "XI",
                "IY": "ZY",
                "YI": "YX",
                "YY": "-XZ",
                "IZ": "ZZ",
                "ZI": "ZI",
                "ZZ": "IZ",
                "XY": "YZ",
                "YX": "YI",
                "XZ": "-YY",
                "ZX": "ZX",
                "YZ": "XY",
                "ZY": "IY",
            },
            "CZ": {
                "II": "II",
                "IX": "ZX",
                "XI": "XZ",
                "XX": "YY",
                "IY": "ZY",
                "YI": "YZ",
                "YY": "XX",
                "IZ": "IZ",
                "ZI": "ZI",
                "ZZ": "ZZ",
                "XY": "-YX",
                "YX": "-XY",
                "XZ": "XI",
                "ZX": "IX",
                "YZ": "YI",
                "ZY": "IY",
            },
            "Swap": {
                "II": "II",
                "IX": "XI",
                "XI": "IX",
                "XX": "XX",
                "IY": "YI",
                "YI": "IY",
                "YY": "YY",
                "IZ": "ZI",
                "ZI": "IZ",
                "ZZ": "ZZ",
                "XY": "YX",
                "YX": "XY",
                "XZ": "ZX",
                "ZX": "XZ",
                "YZ": "ZY",
                "ZY": "YZ",
            },
        }

        measurement_circuit = self._get_measurement_circuit().get_circuit()

        expect_val = 0.0
        for mermin_coef, mermin_pauli in self.mermin_operator:
            # Iterate through the operations in the measurement circuit and conjugate with the
            # current Pauli to determine the correct measurement qubits and coefficient.
            measure_pauli = [p for p in mermin_pauli]
            parity = 1

            for instruction in measurement_circuit.instructions:
                gate_name = instruction.operator.name
                if gate_name == "Measure":
                    break

                qubits = [qubit for qubit in instruction.target]
                substr = [measure_pauli[int(qubit)] for qubit in qubits]
                conjugated_substr = conjugation_rules[gate_name]["".join(substr)]

                if conjugated_substr[0] == "-":
                    parity = -1 * parity
                    conjugated_substr = conjugated_substr[1:]

                for qubit, pauli in zip(qubits, conjugated_substr):
                    measure_pauli[int(qubit)] = pauli

            measurement_qubits = [i for i, pauli in enumerate(measure_pauli) if pauli == "Z"]
            measurement_coef = parity

            numerator = 0.0
            for bitstr, count in counts.items():
                parity = 1
                for qb in measurement_qubits:
                    if bitstr[qb] == "1":  # Qubit order is big endian
                        parity = -1 * parity

                numerator += mermin_coef * measurement_coef * parity * count

            expect_val += numerator / sum(list(counts.values()))

        return (expect_val + 2 ** (self.num_qubits - 1)) / 2**self.num_qubits

    def _mermin_operator(self, num_qubits):
        """
        Generate the Mermin operator
        (https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.65.1838), or M_n
        (Eq. 2.8) in https://arxiv.org/pdf/2005.11271.pdf
        """
        mermin_op = []
        for num_y in range(1, num_qubits + 1, 2):
            coef = (-1.0) ** (num_y // 2)

            for x_indices in itertools.combinations(range(num_qubits), num_qubits - num_y):
                pauli = np.array(["Y"] * num_qubits)
                pauli.put(x_indices, "X")
                mermin_op.append((coef, "".join(pauli)))

        return mermin_op

    def run(self, shots: int, backend=None):
        if not backend:
            sim = LocalSimulator()
        else:
            sim = LocalSimulator(backend=backend)

        circ = self.mb_circuit()
        task = sim.run(circ, shots=shots)

        result = task.result().measurement_counts
        score = self.score(result)
        
        print(f"Result: {result}")
        print(f"Mermin-Bell test score: {score}")
        return score