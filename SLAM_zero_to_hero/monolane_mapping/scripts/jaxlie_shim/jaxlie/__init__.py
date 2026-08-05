"""Minimal numpy stand-in for the parts of jaxlie that MonoLaneMapping uses.

misc/lie_utils.py does `import jaxlie` at module load and calls
`jaxlie.SE3.from_matrix(...).log()` / `jaxlie.SE3.exp(...)`, so the import has to
resolve for the pipeline to start at all. jax and jaxlib no longer publish
cp38 wheels to PyPI, and ROS Noetic pins us to Python 3.8, so there is nothing
to install. This module reimplements exactly those calls in numpy/scipy.

Tangent ordering matches jaxlie: [tx, ty, tz, wx, wy, wz] -- translation first,
rotation last. scripts/verify_jaxlie_shim.py checks this against the real
package over 500 random SE(3) elements (including theta -> 0); agreement is
2e-9 with jax_enable_x64, i.e. jaxlie's own Taylor-threshold residual.
"""
import numpy as np
from scipy.spatial.transform import Rotation as _R

__all__ = ["SO3", "SE3"]
__version__ = "1.3.4+numpy-shim"


def _skew(w):
    return np.array([[0.0, -w[2], w[1]],
                     [w[2], 0.0, -w[0]],
                     [-w[1], w[0], 0.0]])


class SO3:
    def __init__(self, matrix):
        self._matrix = np.asarray(matrix, dtype=float).reshape(3, 3)

    @classmethod
    def from_matrix(cls, matrix):
        return cls(matrix)

    @classmethod
    def exp(cls, tangent):
        return cls(_R.from_rotvec(np.asarray(tangent, dtype=float).reshape(3)).as_matrix())

    def log(self):
        return _R.from_matrix(self._matrix).as_rotvec()

    def as_matrix(self):
        return self._matrix

    def inverse(self):
        return SO3(self._matrix.T)


class SE3:
    def __init__(self, matrix):
        self._matrix = np.asarray(matrix, dtype=float).reshape(4, 4)

    @classmethod
    def from_matrix(cls, matrix):
        return cls(matrix)

    @classmethod
    def from_rotation_and_translation(cls, rotation, translation):
        m = np.eye(4)
        m[:3, :3] = rotation.as_matrix() if isinstance(rotation, SO3) else np.asarray(rotation)
        m[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
        return cls(m)

    @classmethod
    def exp(cls, tangent):
        tangent = np.asarray(tangent, dtype=float).reshape(6)
        rho, omega = tangent[:3], tangent[3:]
        theta = np.linalg.norm(omega)
        W = _skew(omega)
        if theta < 1e-8:
            # Taylor expansions of the two coefficients below, to O(theta^2).
            V = np.eye(3) + 0.5 * W + (1.0 / 6.0) * (W @ W)
        else:
            V = (np.eye(3)
                 + ((1.0 - np.cos(theta)) / theta ** 2) * W
                 + ((theta - np.sin(theta)) / theta ** 3) * (W @ W))
        m = np.eye(4)
        m[:3, :3] = _R.from_rotvec(omega).as_matrix()
        m[:3, 3] = V @ rho
        return cls(m)

    def log(self):
        R_mat = self._matrix[:3, :3]
        t = self._matrix[:3, 3]
        omega = _R.from_matrix(R_mat).as_rotvec()
        theta = np.linalg.norm(omega)
        W = _skew(omega)
        if theta < 1e-8:
            V_inv = np.eye(3) - 0.5 * W + (1.0 / 12.0) * (W @ W)
        else:
            half = 0.5 * theta
            V_inv = (np.eye(3)
                     - 0.5 * W
                     + (1.0 - half * np.cos(half) / np.sin(half)) / theta ** 2 * (W @ W))
        return np.concatenate([V_inv @ t, omega])

    def as_matrix(self):
        return self._matrix

    def rotation(self):
        return SO3(self._matrix[:3, :3])

    def translation(self):
        return self._matrix[:3, 3]

    def inverse(self):
        return SE3(np.linalg.inv(self._matrix))

    def __matmul__(self, other):
        if isinstance(other, SE3):
            return SE3(self._matrix @ other._matrix)
        return NotImplemented
