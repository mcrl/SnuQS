from braket.snuqs.operation import GateOperation
import braket.snuqs.quantumpy as qp

import numpy as np
from collections.abc import Sequence
import braket.ir.jaqcd as braket_instruction
from braket.snuqs.operation_helpers import (
    _from_braket_instruction,
    check_matrix_dimensions,
    check_unitary,
    ir_matrix_to_ndarray,
)
from braket.snuqs._C import (
    Identity,
    Hadamard, PauliX, PauliY, PauliZ,
    CX, CY, CZ, S, Si, T, Ti, V, Vi,
    PhaseShift, CPhaseShift, CPhaseShift00, CPhaseShift01, CPhaseShift10,
    RotX, RotY, RotZ,
    Swap, ISwap, PSwap, XY, XX, YY, ZZ,
    CCNot, CSwap,
    U, GPhase,
)


# class Identity(GateOperation):
#    """Identity gate"""
#
#    def __init__(self, targets, ctrl_modifiers=(), power=1):
#        super().__init__(
#            targets=targets,
#            ctrl_modifiers=ctrl_modifiers,
#            power=power,
#        )
#
#    @property
#    def _base_matrix(self) -> qp.ndarray:
#        return qp.identity()
#
#
# class Hadamard(GateOperation):
#    """Hadamard gate"""
#
#    def __init__(self, targets, ctrl_modifiers=(), power=1):
#        super().__init__(
#            targets=targets,
#            ctrl_modifiers=ctrl_modifiers,
#            power=power,
#        )
#
#    @property
#    def _base_matrix(self) -> qp.ndarray:
#        return qp.hadamard()
#
#
# class PauliX(GateOperation):
#    """Pauli-X gate"""
#
#    def __init__(self, targets, ctrl_modifiers=(), power=1):
#        super().__init__(
#            targets=targets,
#            ctrl_modifiers=ctrl_modifiers,
#            power=power,
#        )
#
#    @property
#    def _base_matrix(self) -> qp.ndarray:
#        return qp.paulix()
#
#
# class PauliY(GateOperation):
#    """Pauli-Y gate"""
#
#    def __init__(self, targets, ctrl_modifiers=(), power=1):
#        super().__init__(
#            targets=targets,
#            ctrl_modifiers=ctrl_modifiers,
#            power=power,
#        )
#
#    @property
#    def _base_matrix(self) -> qp.ndarray:
#        return qp.pauliy()
#
#
# class PauliZ(GateOperation):
#    """Pauli-Z gate"""
#
#    def __init__(self, targets, ctrl_modifiers=(), power=1):
#        super().__init__(
#            targets=targets,
#            ctrl_modifiers=ctrl_modifiers,
#            power=power,
#        )
#
#    @property
#    def _base_matrix(self) -> qp.ndarray:
#        return qp.pauliz()
#
#
# class CX(GateOperation):
#    """Controlled Pauli-X gate"""
#
#    def __init__(self, targets, ctrl_modifiers=(), power=1):
#        super().__init__(
#            targets=targets,
#            ctrl_modifiers=ctrl_modifiers,
#            power=power,
#        )
#
#    @property
#    def _base_matrix(self) -> qp.ndarray:
#        return qp.cx()
#
#
# class CY(GateOperation):
#    """Controlled Pauli-Y gate"""
#
#    def __init__(self, targets, ctrl_modifiers=(), power=1):
#        super().__init__(
#            targets=targets,
#            ctrl_modifiers=ctrl_modifiers,
#            power=power,
#        )
#
#    @property
#    def _base_matrix(self):
#        return qp.cy()
#
#
# class CZ(GateOperation):
#    """Controlled Pauli-Z gate"""
#
#    def __init__(self, targets, ctrl_modifiers=(), power=1):
#        super().__init__(
#            targets=targets,
#            ctrl_modifiers=ctrl_modifiers,
#            power=power,
#        )
#
#    @property
#    def _base_matrix(self) -> qp.ndarray:
#        return qp.cz()
#
#
# class S(GateOperation):
#    """S gate"""
#
#    def __init__(self, targets, ctrl_modifiers=(), power=1):
#        super().__init__(
#            targets=targets,
#            ctrl_modifiers=ctrl_modifiers,
#            power=power,
#        )
#
#    @property
#    def _base_matrix(self) -> qp.ndarray:
#        return qp.s()
#
#
# class Si(GateOperation):
#    r"""The adjoint :math:`S^{\dagger}` of the S gate"""
#
#    def __init__(self, targets, ctrl_modifiers=(), power=1):
#        super().__init__(
#            targets=targets,
#            ctrl_modifiers=ctrl_modifiers,
#            power=power,
#        )
#
#    @property
#    def _base_matrix(self) -> qp.ndarray:
#        return qp.si()
#
#
# class T(GateOperation):
#    """T gate"""
#
#    def __init__(self, targets, ctrl_modifiers=(), power=1):
#        super().__init__(
#            targets=targets,
#            ctrl_modifiers=ctrl_modifiers,
#            power=power,
#        )
#
#    @property
#    def _base_matrix(self) -> qp.ndarray:
#        return qp.t()
#
#
# class Ti(GateOperation):
#    r"""The adjoint :math:`T^{\dagger}` of the T gate"""
#
#    def __init__(self, targets, ctrl_modifiers=(), power=1):
#        super().__init__(
#            targets=targets,
#            ctrl_modifiers=ctrl_modifiers,
#            power=power,
#        )
#
#    @property
#    def _base_matrix(self) -> qp.ndarray:
#        return qp.ti()
#
#
# class V(GateOperation):
#    """Square root of the X (not) gate"""
#
#    def __init__(self, targets, ctrl_modifiers=(), power=1):
#        super().__init__(
#            targets=targets,
#            ctrl_modifiers=ctrl_modifiers,
#            power=power,
#        )
#
#    @property
#    def _base_matrix(self) -> qp.ndarray:
#        return qp.v()
#
#
# class Vi(GateOperation):
#    r"""The adjoint :math:`V^{\dagger}` of the square root of the X (not) gate"""
#
#    def __init__(self, targets, ctrl_modifiers=(), power=1):
#        super().__init__(
#            targets=targets,
#            ctrl_modifiers=ctrl_modifiers,
#            power=power,
#        )
#
#    @property
#    def _base_matrix(self) -> qp.ndarray:
#        return qp.vi()
#
#
# class PhaseShift(GateOperation):
#    """Phase shift gate"""
#
#    def __init__(self, targets, angle, ctrl_modifiers=(), power=1):
#        super().__init__(
#            targets=targets,
#            ctrl_modifiers=ctrl_modifiers,
#            power=power,
#        )
#        self._angle = angle
#
#    @property
#    def _base_matrix(self) -> qp.ndarray:
#        return qp.phase_shift(self._angle)
#
#
# class CPhaseShift(GateOperation):
#    """Controlled phase shift gate"""
#
#    def __init__(self, targets, angle, ctrl_modifiers=(), power=1):
#        super().__init__(
#            targets=targets,
#            ctrl_modifiers=ctrl_modifiers,
#            power=power,
#        )
#        self._angle = angle
#
#    @property
#    def _base_matrix(self) -> qp.ndarray:
#        return qp.cphase_shift(self._angle)
#
#
# class CPhaseShift00(GateOperation):
#    r"""Controlled phase shift gate phasing the :math:`\ket{00}` state"""
#
#    def __init__(self, targets, angle, ctrl_modifiers=(), power=1):
#        super().__init__(
#            targets=targets,
#            ctrl_modifiers=ctrl_modifiers,
#            power=power,
#        )
#        self._angle = angle
#
#    @property
#    def _base_matrix(self) -> qp.ndarray:
#        return qp.cphase_shift00(self._angle)
#
#
# class CPhaseShift01(GateOperation):
#    r"""Controlled phase shift gate phasing the :math:`\ket{01}` state"""
#
#    def __init__(self, targets, angle, ctrl_modifiers=(), power=1):
#        super().__init__(
#            targets=targets,
#            ctrl_modifiers=ctrl_modifiers,
#            power=power,
#        )
#        self._angle = angle
#
#    @property
#    def _base_matrix(self) -> qp.ndarray:
#        return qp.cphase_shift01(self._angle)
#
#
# class CPhaseShift10(GateOperation):
#    r"""Controlled phase shift gate phasing the :math:`\ket{10}` state"""
#
#    def __init__(self, targets, angle, ctrl_modifiers=(), power=1):
#        super().__init__(
#            targets=targets,
#            ctrl_modifiers=ctrl_modifiers,
#            power=power,
#        )
#        self._angle = angle
#
#    @property
#    def _base_matrix(self) -> qp.ndarray:
#        return qp.cphase_shift10(self._angle)
#
#
# class RotX(GateOperation):
#    """X-axis rotation gate"""
#
#    def __init__(self, targets, angle, ctrl_modifiers=(), power=1):
#        super().__init__(
#            targets=targets,
#            ctrl_modifiers=ctrl_modifiers,
#            power=power,
#        )
#        self._angle = angle
#
#    @property
#    def _base_matrix(self) -> qp.ndarray:
#        return qp.rot_x(self._angle)
#
#
# class RotY(GateOperation):
#    """Y-axis rotation gate"""
#
#    def __init__(self, targets, angle, ctrl_modifiers=(), power=1):
#        super().__init__(
#            targets=targets,
#            ctrl_modifiers=ctrl_modifiers,
#            power=power,
#        )
#        self._angle = angle
#
#    @property
#    def _base_matrix(self) -> qp.ndarray:
#        return qp.rot_y(self._angle)
#
#
# class RotZ(GateOperation):
#    """Z-axis rotation gate"""
#
#    def __init__(self, targets, angle, ctrl_modifiers=(), power=1):
#        super().__init__(
#            targets=targets,
#            ctrl_modifiers=ctrl_modifiers,
#            power=power,
#        )
#        self._angle = angle
#
#    @property
#    def _base_matrix(self) -> qp.ndarray:
#        return qp.rot_z(self._angle)
#
#
# class Swap(GateOperation):
#    """Swap gate"""
#
#    def __init__(self, targets, ctrl_modifiers=(), power=1):
#        super().__init__(
#            targets=targets,
#            ctrl_modifiers=ctrl_modifiers,
#            power=power,
#        )
#
#    @property
#    def _base_matrix(self) -> qp.ndarray:
#        return qp.swap()
#
#
# class ISwap(GateOperation):
#    """iSwap gate"""
#
#    def __init__(self, targets, ctrl_modifiers=(), power=1):
#        super().__init__(
#            targets=targets,
#            ctrl_modifiers=ctrl_modifiers,
#            power=power,
#        )
#
#    @property
#    def _base_matrix(self) -> qp.ndarray:
#        return qp.iswap()
#
#
# class PSwap(GateOperation):
#    """Parametrized Swap gate"""
#
#    def __init__(self, targets, angle, ctrl_modifiers=(), power=1):
#        super().__init__(
#            targets=targets,
#            ctrl_modifiers=ctrl_modifiers,
#            power=power,
#        )
#        self._angle = angle
#
#    @property
#    def _base_matrix(self) -> qp.ndarray:
#        return qp.pswap(self._angle)
#
#
# class XY(GateOperation):
#    """XY gate
#
#    Reference: https://arxiv.org/abs/1912.04424v1
#    """
#
#    def __init__(self, targets, angle, ctrl_modifiers=(), power=1):
#        super().__init__(
#            targets=targets,
#            ctrl_modifiers=ctrl_modifiers,
#            power=power,
#        )
#        self._angle = angle
#
#    @property
#    def _base_matrix(self) -> qp.ndarray:
#        return qp.xy(self._angle)
#
#
# class XX(GateOperation):
#    """Ising XX gate
#
#    Reference: https://arxiv.org/abs/1707.06356
#    """
#
#    def __init__(self, targets, angle, ctrl_modifiers=(), power=1):
#        super().__init__(
#            targets=targets,
#            ctrl_modifiers=ctrl_modifiers,
#            power=power,
#        )
#        self._angle = angle
#
#    @property
#    def _base_matrix(self) -> qp.ndarray:
#        return qp.xx(self._angle)
#
#
# class YY(GateOperation):
#    """Ising YY gate
#
#    Reference: https://arxiv.org/abs/1707.06356
#    """
#
#    def __init__(self, targets, angle, ctrl_modifiers=(), power=1):
#        super().__init__(
#            targets=targets,
#            ctrl_modifiers=ctrl_modifiers,
#            power=power,
#        )
#        self._angle = angle
#
#    @property
#    def _base_matrix(self) -> qp.ndarray:
#        return qp.yy(self._angle)
#
#
# class ZZ(GateOperation):
#    """Ising ZZ gate
#
#    Reference: https://arxiv.org/abs/1707.06356
#    """
#
#    def __init__(self, targets, angle, ctrl_modifiers=(), power=1):
#        super().__init__(
#            targets=targets,
#            ctrl_modifiers=ctrl_modifiers,
#            power=power,
#        )
#        self._angle = angle
#
#    @property
#    def _base_matrix(self) -> qp.ndarray:
#        return qp.zz(self._angle)
#
#
# class CCNot(GateOperation):
#    """Controlled CNot or Toffoli gate"""
#
#    def __init__(self, targets, ctrl_modifiers=(), power=1):
#        super().__init__(
#            targets=targets,
#            ctrl_modifiers=ctrl_modifiers,
#            power=power,
#        )
#
#    @property
#    def _base_matrix(self) -> qp.ndarray:
#        return qp.ccnot()
#
#
# class CSwap(GateOperation):
#    """Controlled Swap gate"""
#
#    def __init__(self, targets, ctrl_modifiers=(), power=1):
#        super().__init__(
#            targets=targets,
#            ctrl_modifiers=ctrl_modifiers,
#            power=power,
#        )
#
#    @property
#    def _base_matrix(self) -> qp.ndarray:
#        return qp.cswap()
#
#
# """
# OpenQASM gate operations
# """
#
#
# class U(GateOperation):
#    """
#    Parameterized primitive gate for OpenQASM simulator
#    """
#
#    def __init__(
#        self,
#        targets: Sequence[int],
#        theta: float,
#        phi: float,
#        lambda_: float,
#        ctrl_modifiers: Sequence[int] = (),
#        power: float = 1,
#    ):
#        super().__init__(
#            targets=targets,
#            ctrl_modifiers=ctrl_modifiers,
#            power=power,
#        )
#        self._theta = theta
#        self._phi = phi
#        self._lambda = lambda_
#
#    @property
#    def _base_matrix(self) -> qp.ndarray:
#        """
#        Generate parameterized Unitary matrix.
#        https://openqasm.com/language/gates.html#built-in-gates
#
#        Returns:
#            qp.ndarray: U Matrix
#        """
#        return qp.u(self._theta, self._phi, self._lambda)
#
#
# class GPhase(GateOperation):
#    """
#    Global phase operation for OpenQASM simulator
#    """
#
#    def __init__(self, targets: Sequence[int], angle: float):
#        super().__init__(
#            targets=targets,
#        )
#        self._angle = angle
#
#    @property
#    def _base_matrix(self) -> qp.ndarray:
#        return qp.gphase(self._angle)

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
    def _base_matrix(self) -> qp.ndarray:
        return qp.array(self._matrix)


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


@_from_braket_instruction.register(braket_instruction.CNot)
def _cx(instruction) -> CX:
    return CX([instruction.control, instruction.target])


@_from_braket_instruction.register(braket_instruction.CY)
def _cy(instruction) -> CY:
    return CY([instruction.control, instruction.target])


@_from_braket_instruction.register(braket_instruction.CZ)
def _cz(instruction) -> CZ:
    return CZ([instruction.control, instruction.target])


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
    "cnot": CX,
    "cy": CY,
    "cz": CZ,
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
    "U": U,
    "gphase": GPhase,
    "unitary": Unitary,

    # Removed: CV, ECR, PRx
}
