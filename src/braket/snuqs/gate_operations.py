import cmath
import math
import numpy as np
from braket.snuqs.snuqs_operation import GateOperation

from collections.abc import Sequence
import braket.ir.jaqcd as braket_instruction
from braket.default_simulator.operation_helpers import (
    _from_braket_instruction,
    check_matrix_dimensions,
    check_unitary,
    ir_matrix_to_ndarray,
)


class Identity(GateOperation):
    """Identity gate"""

    def __init__(self, targets, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )

    @property
    def _base_matrix(self) -> np.ndarray:
        return np.eye(2)


class Hadamard(GateOperation):
    """Hadamard gate"""

    def __init__(self, targets, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )

    @property
    def _base_matrix(self) -> np.ndarray:
        return np.array([[1, 1], [1, -1]]) / math.sqrt(2)


class PauliX(GateOperation):
    """Pauli-X gate"""

    def __init__(self, targets, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )

    @property
    def _base_matrix(self) -> np.ndarray:
        return np.array([[0, 1], [1, 0]])


class PauliY(GateOperation):
    """Pauli-Y gate"""

    def __init__(self, targets, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )

    @property
    def _base_matrix(self) -> np.ndarray:
        return np.array([[0, -1j], [1j, 0]])


class PauliZ(GateOperation):
    """Pauli-Z gate"""

    def __init__(self, targets, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )

    @property
    def _base_matrix(self) -> np.ndarray:
        return np.array([[1, 0], [0, -1]])


class CV(GateOperation):
    """Controlled-Sqrt(NOT) gate"""

    def __init__(self, targets, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )

    @property
    def _base_matrix(self) -> np.ndarray:
        return np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0.5 + 0.5j, 0.5 - 0.5j],
                [0, 0, 0.5 - 0.5j, 0.5 + 0.5j],
            ]
        )


class CX(GateOperation):
    """Controlled Pauli-X gate"""

    def __init__(self, targets, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )

    @property
    def _base_matrix(self) -> np.ndarray:
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])


class CY(GateOperation):
    """Controlled Pauli-Y gate"""

    def __init__(self, targets, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )

    @property
    def _base_matrix(self):
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]])


class CZ(GateOperation):
    """Controlled Pauli-Z gate"""

    def __init__(self, targets, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )

    @property
    def _base_matrix(self) -> np.ndarray:
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])


class ECR(GateOperation):
    """ECR gate"""

    def __init__(self, targets, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )

    @property
    def _base_matrix(self) -> np.ndarray:
        return (
            1
            / np.sqrt(2)
            * np.array(
                [[0, 0, 1, 1.0j], [0, 0, 1.0j, 1], [1, -1.0j, 0, 0], [-1.0j, 1, 0, 0]],
                dtype=complex,
            )
        )


class S(GateOperation):
    """S gate"""

    def __init__(self, targets, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )

    @property
    def _base_matrix(self) -> np.ndarray:
        return np.array([[1, 0], [0, 1j]], dtype=complex)


class Si(GateOperation):
    r"""The adjoint :math:`S^{\dagger}` of the S gate"""

    def __init__(self, targets, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )

    @property
    def _base_matrix(self) -> np.ndarray:
        return np.array([[1, 0], [0, -1j]], dtype=complex)


class T(GateOperation):
    """T gate"""

    def __init__(self, targets, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )

    @property
    def _base_matrix(self) -> np.ndarray:
        return np.array([[1, 0], [0, cmath.exp(1j * math.pi / 4)]], dtype=complex)


class Ti(GateOperation):
    r"""The adjoint :math:`T^{\dagger}` of the T gate"""

    def __init__(self, targets, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )

    @property
    def _base_matrix(self) -> np.ndarray:
        return np.array([[1, 0], [0, cmath.exp(-1j * math.pi / 4)]], dtype=complex)


class V(GateOperation):
    """Square root of the X (not) gate"""

    def __init__(self, targets, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )

    @property
    def _base_matrix(self) -> np.ndarray:
        return np.array([[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]], dtype=complex)


class Vi(GateOperation):
    r"""The adjoint :math:`V^{\dagger}` of the square root of the X (not) gate"""

    def __init__(self, targets, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )

    @property
    def _base_matrix(self) -> np.ndarray:
        return np.array(([[0.5 - 0.5j, 0.5 + 0.5j], [0.5 + 0.5j, 0.5 - 0.5j]]), dtype=complex)


class PhaseShift(GateOperation):
    """Phase shift gate"""

    def __init__(self, targets, angle, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )
        self._angle = angle

    @property
    def _base_matrix(self) -> np.ndarray:
        return np.array([[1, 0], [0, cmath.exp(1j * self._angle)]])


class CPhaseShift(GateOperation):
    """Controlled phase shift gate"""

    def __init__(self, targets, angle, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )
        self._angle = angle

    @property
    def _base_matrix(self) -> np.ndarray:
        return np.diag([1.0, 1.0, 1.0, cmath.exp(1j * self._angle)])


class CPhaseShift00(GateOperation):
    r"""Controlled phase shift gate phasing the :math:`\ket{00}` state"""

    def __init__(self, targets, angle, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )
        self._angle = angle

    @property
    def _base_matrix(self) -> np.ndarray:
        return np.diag([cmath.exp(1j * self._angle), 1.0, 1.0, 1.0])


class CPhaseShift01(GateOperation):
    r"""Controlled phase shift gate phasing the :math:`\ket{01}` state"""

    def __init__(self, targets, angle, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )
        self._angle = angle

    @property
    def _base_matrix(self) -> np.ndarray:
        return np.diag([1.0, cmath.exp(1j * self._angle), 1.0, 1.0])


class CPhaseShift10(GateOperation):
    r"""Controlled phase shift gate phasing the :math:`\ket{10}` state"""

    def __init__(self, targets, angle, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )
        self._angle = angle

    @property
    def _base_matrix(self) -> np.ndarray:
        return np.diag([1.0, 1.0, cmath.exp(1j * self._angle), 1.0])


class RotX(GateOperation):
    """X-axis rotation gate"""

    def __init__(self, targets, angle, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )
        self._angle = angle

    @property
    def _base_matrix(self) -> np.ndarray:
        cos_half_angle = math.cos(self._angle / 2)
        i_sin_half_angle = 1j * math.sin(self._angle / 2)
        return np.array([[cos_half_angle, -i_sin_half_angle], [-i_sin_half_angle, cos_half_angle]])


class RotY(GateOperation):
    """Y-axis rotation gate"""

    def __init__(self, targets, angle, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )
        self._angle = angle

    @property
    def _base_matrix(self) -> np.ndarray:
        cos_half_angle = math.cos(self._angle / 2)
        sin_half_angle = math.sin(self._angle / 2)
        return np.array([[cos_half_angle, -sin_half_angle], [sin_half_angle, cos_half_angle]])


class RotZ(GateOperation):
    """Z-axis rotation gate"""

    def __init__(self, targets, angle, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )
        self._angle = angle

    @property
    def _base_matrix(self) -> np.ndarray:
        positive_phase = cmath.exp(1j * self._angle / 2)
        negative_phase = cmath.exp(-1j * self._angle / 2)
        return np.array([[negative_phase, 0], [0, positive_phase]])


class Swap(GateOperation):
    """Swap gate"""

    def __init__(self, targets, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )

    @property
    def _base_matrix(self) -> np.ndarray:
        return np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])


class ISwap(GateOperation):
    """iSwap gate"""

    def __init__(self, targets, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )

    @property
    def _base_matrix(self) -> np.ndarray:
        return np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0j, 0.0],
                [0.0, 1.0j, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=complex,
        )


class PSwap(GateOperation):
    """Parametrized Swap gate"""

    def __init__(self, targets, angle, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )
        self._angle = angle

    @property
    def _base_matrix(self) -> np.ndarray:
        return np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, cmath.exp(1j * self._angle), 0.0],
                [0.0, cmath.exp(1j * self._angle), 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=complex,
        )


class XY(GateOperation):
    """XY gate

    Reference: https://arxiv.org/abs/1912.04424v1
    """

    def __init__(self, targets, angle, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )
        self._angle = angle

    @property
    def _base_matrix(self) -> np.ndarray:
        cos = math.cos(self._angle / 2)
        sin = math.sin(self._angle / 2)
        return np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, cos, 1.0j * sin, 0.0],
                [0.0, 1.0j * sin, cos, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=complex,
        )


class XX(GateOperation):
    """Ising XX gate

    Reference: https://arxiv.org/abs/1707.06356
    """

    def __init__(self, targets, angle, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )
        self._angle = angle

    @property
    def _base_matrix(self) -> np.ndarray:
        cos_angle = math.cos(self._angle / 2)
        i_sin_angle = 1j * math.sin(self._angle / 2)
        return np.array(
            [
                [cos_angle, 0, 0, -i_sin_angle],
                [0, cos_angle, -i_sin_angle, 0],
                [0, -i_sin_angle, cos_angle, 0],
                [-i_sin_angle, 0, 0, cos_angle],
            ]
        )


class YY(GateOperation):
    """Ising YY gate

    Reference: https://arxiv.org/abs/1707.06356
    """

    def __init__(self, targets, angle, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )
        self._angle = angle

    @property
    def _base_matrix(self) -> np.ndarray:
        cos_angle = math.cos(self._angle / 2)
        i_sin_angle = 1j * math.sin(self._angle / 2)
        return np.array(
            [
                [cos_angle, 0, 0, i_sin_angle],
                [0, cos_angle, -i_sin_angle, 0],
                [0, -i_sin_angle, cos_angle, 0],
                [i_sin_angle, 0, 0, cos_angle],
            ]
        )


class ZZ(GateOperation):
    """Ising ZZ gate

    Reference: https://arxiv.org/abs/1707.06356
    """

    def __init__(self, targets, angle, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )
        self._angle = angle

    @property
    def _base_matrix(self) -> np.ndarray:
        positive_phase = cmath.exp(1j * self._angle / 2)
        negative_phase = cmath.exp(-1j * self._angle / 2)
        return np.array(
            [
                [negative_phase, 0, 0, 0],
                [0, positive_phase, 0, 0],
                [0, 0, positive_phase, 0],
                [0, 0, 0, negative_phase],
            ]
        )


class CCNot(GateOperation):
    """Controlled CNot or Toffoli gate"""

    def __init__(self, targets, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )

    @property
    def _base_matrix(self) -> np.ndarray:
        return np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
            ],
            dtype=complex,
        )


class CSwap(GateOperation):
    """Controlled Swap gate"""

    def __init__(self, targets, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )

    @property
    def _base_matrix(self) -> np.ndarray:
        return np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=complex,
        )


class PRx(GateOperation):
    """
    PhaseRx gate.
    """

    def __init__(self, targets, angle_1, angle_2, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )
        self._angle_1 = angle_1
        self._angle_2 = angle_2

    @property
    def _base_matrix(self) -> np.ndarray:
        theta = self._angle_1
        phi = self._angle_2
        return np.array(
            [
                [
                    np.cos(theta / 2),
                    -1j * np.exp(-1j * phi) * np.sin(theta / 2),
                ],
                [
                    -1j * np.exp(1j * phi) * np.sin(theta / 2),
                    np.cos(theta / 2),
                ],
            ]
        )


class Unitary(GateOperation):
    """Arbitrary unitary gate"""

    def __init__(self, targets, matrix, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )
        clone = np.array(matrix, dtype=complex)
        check_matrix_dimensions(clone, self._targets)
        check_unitary(clone)
        self._matrix = clone

    @property
    def _base_matrix(self) -> np.ndarray:
        return np.array(self._matrix)


"""
OpenQASM gate operations
"""


class U(GateOperation):
    """
    Parameterized primitive gate for OpenQASM simulator
    """

    def __init__(
        self,
        targets: Sequence[int],
        theta: float,
        phi: float,
        lambda_: float,
        ctrl_modifiers: Sequence[int] = (),
        power: float = 1,
    ):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )
        self._theta = theta
        self._phi = phi
        self._lambda = lambda_

    @property
    def _base_matrix(self) -> np.ndarray:
        """
        Generate parameterized Unitary matrix.
        https://openqasm.com/language/gates.html#built-in-gates

        Returns:
            np.ndarray: U Matrix
        """
        return np.array(
            [
                [
                    math.cos(self._theta / 2),
                    -cmath.exp(1j * self._lambda) * math.sin(self._theta / 2),
                ],
                [
                    cmath.exp(1j * self._phi) * math.sin(self._theta / 2),
                    cmath.exp(1j * (self._phi + self._lambda)) * math.cos(self._theta / 2),
                ],
            ]
        )


class GPhase(GateOperation):
    """
    Global phase operation for OpenQASM simulator
    """

    def __init__(self, targets: Sequence[int], angle: float):
        super().__init__(
            targets=targets,
        )
        self._angle = angle

    @property
    def _base_matrix(self) -> np.ndarray:
        return cmath.exp(self._angle * 1j) * np.eye(2 ** len(self._targets))


@_from_braket_instruction.register(braket_instruction.I)
def _i(instruction) -> Identity:
    return Identity([instruction.target])


@_from_braket_instruction.register(braket_instruction.H)
def _hadamard(instruction) -> Hadamard:
    return Hadamard([instruction.target])


@_from_braket_instruction.register(braket_instruction.X)
def _pauli_x(instruction) -> PauliX:
    return PauliX([instruction.target])


@_from_braket_instruction.register(braket_instruction.Y)
def _pauli_y(instruction) -> PauliY:
    return PauliY([instruction.target])


@_from_braket_instruction.register(braket_instruction.Z)
def _pauli_z(instruction) -> PauliZ:
    return PauliZ([instruction.target])


@_from_braket_instruction.register(braket_instruction.CV)
def _cv(instruction) -> CV:
    return CV([instruction.control, instruction.target])


@_from_braket_instruction.register(braket_instruction.CNot)
def _cx(instruction) -> CX:
    return CX([instruction.control, instruction.target])


@_from_braket_instruction.register(braket_instruction.CY)
def _cy(instruction) -> CY:
    return CY([instruction.control, instruction.target])


@_from_braket_instruction.register(braket_instruction.CZ)
def _cz(instruction) -> CZ:
    return CZ([instruction.control, instruction.target])


@_from_braket_instruction.register(braket_instruction.ECR)
def _ecr(instruction) -> ECR:
    return ECR(instruction.targets)


@_from_braket_instruction.register(braket_instruction.S)
def _s(instruction) -> S:
    return S([instruction.target])


@_from_braket_instruction.register(braket_instruction.Si)
def _si(instruction) -> Si:
    return Si([instruction.target])


@_from_braket_instruction.register(braket_instruction.T)
def _t(instruction) -> T:
    return T([instruction.target])


@_from_braket_instruction.register(braket_instruction.Ti)
def _ti(instruction) -> Ti:
    return Ti([instruction.target])


@_from_braket_instruction.register(braket_instruction.V)
def _v(instruction) -> V:
    return V([instruction.target])


@_from_braket_instruction.register(braket_instruction.Vi)
def _vi(instruction) -> Vi:
    return Vi([instruction.target])


@_from_braket_instruction.register(braket_instruction.PhaseShift)
def _phase_shift(instruction) -> PhaseShift:
    return PhaseShift([instruction.target], instruction.angle)


@_from_braket_instruction.register(braket_instruction.CPhaseShift)
def _c_phase_shift(instruction) -> CPhaseShift:
    return CPhaseShift([instruction.control, instruction.target], instruction.angle)


@_from_braket_instruction.register(braket_instruction.CPhaseShift00)
def _c_phase_shift_00(instruction) -> CPhaseShift00:
    return CPhaseShift00([instruction.control, instruction.target], instruction.angle)


@_from_braket_instruction.register(braket_instruction.CPhaseShift01)
def _c_phase_shift_01(instruction) -> CPhaseShift01:
    return CPhaseShift01([instruction.control, instruction.target], instruction.angle)


@_from_braket_instruction.register(braket_instruction.CPhaseShift10)
def _c_phase_shift_10(instruction) -> CPhaseShift10:
    return CPhaseShift10([instruction.control, instruction.target], instruction.angle)


@_from_braket_instruction.register(braket_instruction.Rx)
def _rot_x(instruction) -> RotX:
    return RotX([instruction.target], instruction.angle)


@_from_braket_instruction.register(braket_instruction.Ry)
def _rot_y(instruction) -> RotY:
    return RotY([instruction.target], instruction.angle)


@_from_braket_instruction.register(braket_instruction.Rz)
def _rot_z(instruction) -> RotZ:
    return RotZ([instruction.target], instruction.angle)


@_from_braket_instruction.register(braket_instruction.Swap)
def _swap(instruction) -> Swap:
    return Swap(instruction.targets)


@_from_braket_instruction.register(braket_instruction.ISwap)
def _iswap(instruction) -> ISwap:
    return ISwap(instruction.targets)


@_from_braket_instruction.register(braket_instruction.PSwap)
def _pswap(instruction) -> PSwap:
    return PSwap(instruction.targets, instruction.angle)


@_from_braket_instruction.register(braket_instruction.XY)
def _xy(instruction) -> XY:
    return XY(instruction.targets, instruction.angle)


@_from_braket_instruction.register(braket_instruction.XX)
def _xx(instruction) -> XX:
    return XX(instruction.targets, instruction.angle)


@_from_braket_instruction.register(braket_instruction.YY)
def _yy(instruction) -> YY:
    return YY(instruction.targets, instruction.angle)


@_from_braket_instruction.register(braket_instruction.ZZ)
def _zz(instruction) -> ZZ:
    return ZZ(instruction.targets, instruction.angle)


@_from_braket_instruction.register(braket_instruction.CCNot)
def _ccnot(instruction) -> CCNot:
    return CCNot([*instruction.controls, instruction.target])


@_from_braket_instruction.register(braket_instruction.CSwap)
def _cswap(instruction) -> CSwap:
    return CSwap([instruction.control, *instruction.targets])


@_from_braket_instruction.register(braket_instruction.Unitary)
def _unitary(instruction) -> Unitary:
    return Unitary(instruction.targets, ir_matrix_to_ndarray(instruction.matrix))


BRAKET_GATES = {
    "i": Identity,
    "h": Hadamard,
    "x": PauliX,
    "y": PauliY,
    "z": PauliZ,
    "cv": CV,
    "cnot": CX,
    "cy": CY,
    "cz": CZ,
    "ecr": ECR,
    "s": S,
    "si": Si,
    "t": T,
    "ti": Ti,
    "v": V,
    "vi": Vi,
    "phaseshift": PhaseShift,
    "cphaseshift": CPhaseShift,
    "cphaseshift00": CPhaseShift00,
    "cphaseshift01": CPhaseShift01,
    "cphaseshift10": CPhaseShift10,
    "rx": RotX,
    "ry": RotY,
    "rz": RotZ,
    "swap": Swap,
    "iswap": ISwap,
    "pswap": PSwap,
    "xy": XY,
    "xx": XX,
    "yy": YY,
    "zz": ZZ,
    "ccnot": CCNot,
    "cswap": CSwap,
    "prx": PRx,
    "unitary": Unitary,
    "U": U,
    "gphase": GPhase,
}
