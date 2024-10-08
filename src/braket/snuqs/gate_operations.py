from braket.snuqs._C.operation import GateOperation

import numpy as np
import braket.ir.jaqcd as braket_instruction
from braket.snuqs.operation_helpers import (
    _from_braket_instruction,
    check_matrix_dimensions,
    check_unitary,
    ir_matrix_to_ndarray,
)
from braket.snuqs._C.operation.gate_operations import (
    Identity,
    Hadamard, PauliX, PauliY, PauliZ,
    CX, CY, CZ, S, Si, T, Ti, V, Vi,
    PhaseShift, CPhaseShift, CPhaseShift00, CPhaseShift01, CPhaseShift10,
    RotX, RotY, RotZ,
    Swap, ISwap, PSwap, XY, XX, YY, ZZ,
    CCNot, CSwap,
    U, GPhase,
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
