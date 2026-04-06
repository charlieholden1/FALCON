# uncompyle6 version 3.9.3
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.8.10 (default, Mar 18 2025, 20:04:55) 
# [GCC 9.4.0]
# Embedded file name: /home/ionat/FALCON/radar.py
# Compiled at: 2026-03-23 12:43:30
# Size of source mod 2**32: 6144 bytes
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
import math, time
from typing import Tuple
import numpy as np

class MockRadar:
    __doc__ = "\n    Simulate a 3-D radar return for a single person.\n\n    The target oscillates in *X* (cross-range) via ``sin(t)`` while\n    *Y* (depth / range) stays at a fixed distance and *Z* (height)\n    remains at ground level.\n\n    Parameters\n    ----------\n    amplitude_x : float\n        Peak lateral displacement (metres).  Default 1.0 m.\n    frequency : float\n        Oscillation frequency (Hz).  Default 0.25 Hz (one full cycle\n        every 4 s – a comfortable walking pace).\n    depth_y : float\n        Fixed depth from the sensor (metres).  Default 2.0 m.\n    height_z : float\n        Fixed height offset (metres).  Default 0.0 m.\n    "

    def __init__(self, amplitude_x=1.0, frequency=0.25, depth_y=2.0, height_z=0.0):
        self.amplitude_x = amplitude_x
        self.frequency = frequency
        self.depth_y = depth_y
        self.height_z = height_z
        self._t0 = time.time()

    def get_target_3d(self) -> "np.ndarray":
        """
        Return the current mock radar target position as a NumPy array
        ``[X, Y, Z]`` in metres.

        X oscillates between ``-amplitude_x`` and ``+amplitude_x``.
        """
        elapsed = time.time() - self._t0
        x = self.amplitude_x * math.sin(2.0 * math.pi * self.frequency * elapsed)
        return np.array([x, self.depth_y, self.height_z], dtype=(np.float64))


class CameraProjection:
    __doc__ = "\n    Pinhole camera model:  ``p_pixel = K @ [R | t] @ p_world``\n\n    Parameters\n    ----------\n    K : np.ndarray | None\n        3×3 intrinsic matrix.  If *None* a reasonable 720p webcam\n        default is used (focal length ≈ 800 px, principal point at\n        frame centre).\n    RT : np.ndarray | None\n        4×4 extrinsic matrix (world-to-camera rigid transform).\n        If *None* a default is used that places the radar 10 cm\n        below the camera (translation along the camera Y axis).\n    "

    def __init__(self, K: "np.ndarray | None"=None, RT: "np.ndarray | None"=None):
        if K is not None:
            self.K = np.asarray(K, dtype=(np.float64))
        else:
            fx = fy = 800.0
            cx, cy = (640.0, 360.0)
            self.K = np.array([
             [
              fx, 0.0, cx],
             [
              0.0, fy, cy],
             [
              0.0, 0.0, 1.0]],
              dtype=(np.float64))
        if RT is not None:
            self.RT = np.asarray(RT, dtype=(np.float64))
        else:
            self.RT = np.array([
             [
              1.0, 0.0, 0.0, 0.0],
             [
              0.0, 0.0, -1.0, 0.1],
             [
              0.0, 1.0, 0.0, 0.0],
             [
              0.0, 0.0, 0.0, 1.0]],
              dtype=(np.float64))
        Rt_3x4 = self.RT[(None[:3], None[:None])]
        self.P = self.K @ Rt_3x4

    def project_3d_to_2d(self, point_3d: "np.ndarray") -> "Tuple[int, int]":
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
        p_h = np.array([*point_3d, 1.0], dtype=(np.float64))
        projected = self.P @ p_h
        if abs(projected[2]) < 1e-08:
            return (-1, -1)
        u = projected[0] / projected[2]
        v = projected[1] / projected[2]
        return (int(round(u)), int(round(v)))

# okay decompiling __pycache__/radar.cpython-38.pyc
