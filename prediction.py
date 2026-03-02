"""
F.A.L.C.O.N. – Position Prediction Module
==========================================

Provides a lightweight Kalman filter for 2D bounding-box-center tracking
and a linear-extrapolation fallback.  Both produce position estimates and
uncertainty metrics that drive the visualization of predicted boxes.

State vector: [x, y, vx, vy]   (centre position + velocity)
Measurement :  [x, y]          (detected centre)

The Kalman filter is instantiated per-person via :class:`KalmanPredictor`.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


# ── KalmanPredictor ─────────────────────────────────────────────────

class KalmanPredictor:
    """
    Constant-velocity Kalman filter for 2D point tracking.

    Parameters
    ----------
    dt : float
        Time step between frames (1/FPS).
    process_noise : float
        Scalar multiplier for the process-noise matrix **Q**.
    measurement_noise : float
        Scalar multiplier for the measurement-noise matrix **R**.
    """

    def __init__(
        self,
        dt: float = 1.0 / 30.0,
        process_noise: float = 5.0,
        measurement_noise: float = 2.0,
    ):
        self.dt = dt
        self._initialized = False

        # State transition  (constant velocity)
        #   x'  = x + vx*dt
        #   y'  = y + vy*dt
        #   vx' = vx
        #   vy' = vy
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=np.float64)

        # Measurement matrix  (observe position only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        # Process noise
        G = np.array([
            [0.5 * dt**2, 0],
            [0, 0.5 * dt**2],
            [dt, 0],
            [0, dt],
        ], dtype=np.float64)
        self.Q = process_noise * (G @ G.T)

        # Measurement noise
        self.R = measurement_noise * np.eye(2, dtype=np.float64)

        # State & covariance (populated on first init_state)
        self.x: np.ndarray = np.zeros(4, dtype=np.float64)
        self.P: np.ndarray = np.eye(4, dtype=np.float64) * 100.0

    # ── public API ───────────────────────────────────────────────────

    def init_state(self, center: np.ndarray) -> None:
        """Initialise the filter with the first measurement [cx, cy]."""
        self.x = np.array([center[0], center[1], 0.0, 0.0], dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 100.0
        self._initialized = True

    def predict(self) -> np.ndarray:
        """
        Propagate state one time-step forward.

        Returns the predicted position [x, y].
        """
        if not self._initialized:
            return self.x[:2].copy()
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].copy()

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        Correct the state with a new measurement [cx, cy].

        Returns the corrected position [x, y].
        """
        if not self._initialized:
            self.init_state(measurement)
            return self.x[:2].copy()

        z = measurement.astype(np.float64)
        y = z - self.H @ self.x                     # innovation
        S = self.H @ self.P @ self.H.T + self.R     # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)    # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x[:2].copy()

    def get_state_position(self) -> Optional[np.ndarray]:
        """Return predicted centre [x, y] or None if uninitialised."""
        if not self._initialized:
            return None
        return self.x[:2].copy()

    def get_velocity(self) -> np.ndarray:
        """Return estimated velocity [vx, vy]."""
        return self.x[2:4].copy()

    def get_uncertainty(self) -> Tuple[float, float]:
        """
        Return 1-sigma position uncertainty (σ_x, σ_y) from the
        state covariance.  Useful for drawing confidence ellipses.
        """
        return float(np.sqrt(self.P[0, 0])), float(np.sqrt(self.P[1, 1]))


# ── LinearExtrapolator (fallback) ───────────────────────────────────

class LinearExtrapolator:
    """
    Dead-simple fallback: next_pos = last_pos + velocity · dt.

    Used when the Kalman filter is not yet initialised or for
    ultra-low-latency scenarios.
    """

    def __init__(self, dt: float = 1.0 / 30.0):
        self.dt = dt
        self._last_pos: Optional[np.ndarray] = None
        self._velocity: np.ndarray = np.zeros(2, dtype=np.float64)

    def update(self, position: np.ndarray) -> None:
        if self._last_pos is not None:
            self._velocity = (position - self._last_pos) / self.dt
        self._last_pos = position.copy()

    def predict(self) -> Optional[np.ndarray]:
        if self._last_pos is None:
            return None
        return self._last_pos + self._velocity * self.dt
