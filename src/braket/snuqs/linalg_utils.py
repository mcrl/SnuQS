from typing import Optional
import numpy as np


_SLICES = (
    _NEG_CONTROL_SLICE := slice(None, 1),
    _CONTROL_SLICE := slice(1, None),
    _NO_CONTROL_SLICE := slice(None, None),
)


def multiply_matrix(
    state: np.ndarray,
    matrix: np.ndarray,
    targets: tuple[int, ...],
    controls: Optional[tuple[int, ...]] = (),
    control_state: Optional[tuple[int, ...]] = (),
) -> np.ndarray:
    """Multiplies the given matrix by the given state, applying the matrix on the target qubits,
    controlling the operation as specified.

    Args:
        state (np.ndarray): The state to multiply the matrix by.
        matrix (np.ndarray): The matrix to apply to the state.
        targets (tuple[int]): The qubits to apply the state on.
        controls (Optional[tuple[int]]): The qubits to control the operation on. Default ().
        control_state (Optional[tuple[int]]): A tuple of same length as `controls` with either
            a 0 or 1 in each index, corresponding to whether to control on the `|0⟩` or `|1⟩` state.
            Default (1,) * len(controls).

    Returns:
        np.ndarray: The state after the matrix has been applied.
    """
    if not controls:
        return _multiply_matrix(state, matrix, targets)

    control_state = control_state or (1,) * len(controls)
    num_qubits = len(state.shape)
    control_slices = {i: _SLICES[state] for i, state in zip(controls, control_state)}
    ctrl_index = tuple(
        control_slices[i] if i in controls else _NO_CONTROL_SLICE for i in range(num_qubits)
    )
    state[ctrl_index] = _multiply_matrix(state[ctrl_index], matrix, targets)
    return state


def _multiply_matrix(
    state: np.ndarray,
    matrix: np.ndarray,
    targets: tuple[int, ...],
) -> np.ndarray:
    """Multiplies the given matrix by the given state, applying the matrix on the target qubits.

    Args:
        state (np.ndarray): The state to multiply the matrix by.
        matrix (np.ndarray): The matrix to apply to the state.
        targets (tuple[int]): The qubits to apply the state on.

    Returns:
        np.ndarray: The state after the matrix has been applied.
    """
    gate_matrix = np.reshape(matrix, [2] * len(targets) * 2)
    axes = (
        np.arange(len(targets), 2 * len(targets)),
        targets,
    )
    product = np.tensordot(gate_matrix, state, axes=axes)

    # Axes given in `operation.targets` are in the first positions.
    unused_idxs = [idx for idx in range(len(state.shape)) if idx not in targets]
    permutation = list(targets) + unused_idxs
    # Invert the permutation to put the indices in the correct place
    inverse_permutation = np.argsort(permutation)
    return np.transpose(product, inverse_permutation)
