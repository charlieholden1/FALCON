"""
F.A.L.C.O.N. – Mock Radar & Coordinate Projection Module
==========================================================

Provides two classes for sensor-fusion prototyping:

MockRadar
    Simulates a single 3-D radar target that oscillates along
    the X axis (person walking back and forth) while Y (depth)
    and Z (height) remain fixed.

CameraProjection
    Implements the pinhole camera projection equation

    .. math::
        p_c = K [R | t] p_r

    using dummy intrinsic and extrinsic matrices suitable for a
    standard 720p / 1080p webcam with the radar mounted ≈ 10 cm
    below the camera.
"""

from __future__ import annotations

import math
import time
from typing import Tuple

import numpy as np


# ── MockRadar ────────────────────────────────────────────────────────

class MockRadar:
    """
    Simulate a 3-D radar return for a single person.

    The target oscillates in *X* (cross-range) via ``sin(t)`` while
    *Y* (depth / range) stays at a fixed distance and *Z* (height)
    remains at ground level.

    Parameters
    ----------
    amplitude_x : float
        Peak lateral displacement (metres).  Default 1.0 m.
    frequency : float
        Oscillation frequency (Hz).  Default 0.25 Hz (one full cycle
        every 4 s – a comfortable walking pace).
    depth_y : float
        Fixed depth from the sensor (metres).  Default 2.0 m.
    height_z : float
        Fixed height offset (metres).  Default 0.0 m.
    """

    def __init__(
        self,
        amplitude_x: float = 1.0,
        frequency: float = 0.25,
        depth_y: float = 2.0,
        height_z: float = 0.0,
    ):
        self.amplitude_x = amplitude_x
        self.frequency = frequency
        self.depth_y = depth_y
        self.height_z = height_z
        self._t0 = time.time()

    def get_target_3d(self) -> np.ndarray:
        """
        Return the current mock radar target position as a NumPy array
        ``[X, Y, Z]`` in metres.

        X oscillates between ``-amplitude_x`` and ``+amplitude_x``.
        """
        elapsed = time.time() - self._t0
        x = self.amplitude_x * math.sin(2.0 * math.pi * self.frequency * elapsed)
        return np.array([x, self.depth_y, self.height_z], dtype=np.float64)


# ── CameraProjection ────────────────────────────────────────────────

class CameraProjection:
    """
    Pinhole camera model:  ``p_pixel = K @ [R | t] @ p_world``

    Parameters
    ----------
    K : np.ndarray | None
        3×3 intrinsic matrix.  If *None* a reasonable 720p webcam
        default is used (focal length ≈ 800 px, principal point at
        frame centre).
    RT : np.ndarray | None
        4×4 extrinsic matrix (world-to-camera rigid transform).
        If *None* a default is used that places the radar 10 cm
        below the camera (translation along the camera Y axis).
    """

    def __init__(
        self,
        K: np.ndarray | None = None,
        RT: np.ndarray | None = None,
    ):
        # ── Intrinsic matrix (approximate 720p webcam) ──────────────
        if K is not None:
            self.K = np.asarray(K, dtype=np.float64)
        else:
            fx = fy = 800.0          # focal length in pixels
            cx, cy = 640.0, 360.0    # principal point (1280×720 / 2)
            self.K = np.array([
                [fx,  0.0, cx],
                [0.0, fy,  cy],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)

        # ── Extrinsic matrix (radar → camera) ──────────────────────
        #    Radar coord-system :  X = lateral,  Y = depth (range),
        #                          Z = height (up)
        #    Camera coord-system:  X = right,    Y = down,
        #                          Z = forward (optical axis)
        #
        #    Rotation maps:
        #        radar X  →  camera X   (lateral stays lateral)
        #        radar Y  →  camera Z   (depth → optical axis)
        #        radar Z  →  camera -Y  (up → down-inverted)
        #
        #    Translation: radar is 10 cm below the camera, so in
        #    camera Y the radar origin sits at +0.10 m.
        if RT is not None:
            self.RT = np.asarray(RT, dtype=np.float64)
        else:
            self.RT = np.array([
                [1.0,  0.0,  0.0, 0.00],
                [0.0,  0.0, -1.0, 0.10],
                [0.0,  1.0,  0.0, 0.00],
                [0.0,  0.0,  0.0, 1.00],
            ], dtype=np.float64)

        # Pre-compute 3×4 projection matrix  P = K @ [R|t]
        Rt_3x4 = self.RT[:3, :]          # drop last row → 3×4
        self.P = self.K @ Rt_3x4         # 3×4

    # ── public API ───────────────────────────────────────────────────

    def project_3d_to_2d(self, point_3d: np.ndarray) -> Tuple[int, int]:
        """
        Project a 3-D world point ``[X, Y, Z]`` to a 2-D pixel
        coordinate ``(u, v)``.

        Parameters
        ----------
        point_3d : np.ndarray
            Shape ``(3,)`` – coordinates in metres.

        Returns
        -------
        tuple[int, int]
            Pixel location ``(u, v)`` (column, row).  Values may be
            negative or exceed the frame dimensions if the point falls
            outside the field of view.
        """
        # Homogeneous world coordinate  [X, Y, Z, 1]
        p_h = np.array([*point_3d, 1.0], dtype=np.float64)

        # Project:  s·[u, v, 1]^T = P @ [X, Y, Z, 1]^T
        projected = self.P @ p_h         # shape (3,)

        # Perspective divide
        if abs(projected[2]) < 1e-8:
            # Point is at or behind the camera – return off-screen
            return (-1, -1)

        u = projected[0] / projected[2]
        v = projected[1] / projected[2]
        return int(round(u)), int(round(v))
