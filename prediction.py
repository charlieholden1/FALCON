"""
F.A.L.C.O.N. – Position Prediction Module  (7D Kalman)
=======================================================

Provides a 7-dimensional Kalman filter for bounding-box tracking with
dynamic scale estimation, plus a linear-extrapolation fallback.

State vector : [x, y, s, r, vx, vy, vs]
    x, y   – bounding-box centre coordinates
    s      – bounding-box scale (area in pixels²)
    r      – bounding-box aspect ratio (w / h)  — treated as constant
    vx, vy – centre velocity
    vs     – scale velocity (growth / shrinkage rate)

Measurement  : [x, y, s, r]
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


# ── KalmanPredictor ─────────────────────────────────────────────────

class KalmanPredictor:
    """
    Constant-velocity Kalman filter with 7D state for bounding-box
    centre + scale tracking.

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

        # State transition  (constant velocity for x, y, s; r is constant)
        #   x'  = x  + vx*dt
        #   y'  = y  + vy*dt
        #   s'  = s  + vs*dt
        #   r'  = r
        #   vx' = vx
        #   vy' = vy
        #   vs' = vs
        self.F = np.array([
            [1, 0, 0, 0, dt,  0,  0],
            [0, 1, 0, 0,  0, dt,  0],
            [0, 0, 1, 0,  0,  0, dt],
            [0, 0, 0, 1,  0,  0,  0],
            [0, 0, 0, 0,  1,  0,  0],
            [0, 0, 0, 0,  0,  1,  0],
            [0, 0, 0, 0,  0,  0,  1],
        ], dtype=np.float64)

        # Measurement matrix  (observe x, y, s, r)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ], dtype=np.float64)

        # Process noise  (G maps process-noise accelerations → state)
        G = np.array([
            [0.5 * dt**2, 0,            0],
            [0,           0.5 * dt**2,   0],
            [0,           0,             0.5 * dt**2],
            [0,           0,             0],
            [dt,          0,             0],
            [0,           dt,            0],
            [0,           0,             dt],
        ], dtype=np.float64)
        self.Q = process_noise * (G @ G.T)

        # Measurement noise
        self.R = measurement_noise * np.eye(4, dtype=np.float64)

        # State & covariance (populated on first init_state)
        self.x: np.ndarray = np.zeros(7, dtype=np.float64)
        self.P: np.ndarray = np.eye(7, dtype=np.float64) * 100.0

    # ── public API ───────────────────────────────────────────────────

    def init_state(self, center: np.ndarray, scale: float = 1.0,
                   aspect: float = 1.0) -> None:
        """Initialise the filter with [cx, cy] and optionally bbox scale & aspect."""
        self.x = np.array(
            [center[0], center[1], scale, aspect, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )
        self.P = np.eye(7, dtype=np.float64) * 100.0
        self._initialized = True

    def predict(self) -> np.ndarray:
        """
        Propagate state one time-step forward.

        Returns the predicted position [x, y].
        """
        if not self._initialized:
            return self.x[:2].copy()
        self.x = self.F @ self.x
        # Ensure scale stays positive
        if self.x[2] < 1.0:
            self.x[2] = 1.0
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].copy()

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        Correct the state with a new measurement [cx, cy, s, r].

        For backward compatibility a 2-element [cx, cy] measurement is
        accepted; scale and aspect ratio are then taken from the current
        state (i.e. no correction on those dimensions).

        Returns the corrected position [x, y].
        """
        if not self._initialized:
            if len(measurement) >= 4:
                self.init_state(measurement[:2], measurement[2], measurement[3])
            else:
                self.init_state(measurement[:2])
            return self.x[:2].copy()

        if len(measurement) < 4:
            z = np.array([measurement[0], measurement[1],
                          self.x[2], self.x[3]], dtype=np.float64)
        else:
            z = measurement[:4].astype(np.float64)

        y = z - self.H @ self.x                      # innovation
        S = self.H @ self.P @ self.H.T + self.R      # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)     # Kalman gain
        self.x = self.x + K @ y
        if self.x[2] < 1.0:
            self.x[2] = 1.0
        self.P = (np.eye(7) - K @ self.H) @ self.P
        return self.x[:2].copy()

    def get_state_position(self) -> Optional[np.ndarray]:
        """Return predicted centre [x, y] or None if uninitialised."""
        if not self._initialized:
            return None
        return self.x[:2].copy()

    def get_state_bbox(self) -> Optional[np.ndarray]:
        """Return predicted [x1, y1, x2, y2] derived from (x, y, s, r)."""
        if not self._initialized:
            return None
        cx, cy, s, r = self.x[0], self.x[1], self.x[2], self.x[3]
        s = max(s, 1.0)
        r = max(r, 0.1)
        h = np.sqrt(s / r)
        w = r * h
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    def get_velocity(self) -> np.ndarray:
        """Return estimated velocity [vx, vy]."""
        return self.x[4:6].copy()

    def get_scale_velocity(self) -> float:
        """Return estimated scale velocity vs."""
        return float(self.x[6])

    def get_uncertainty(self) -> Tuple[float, float]:
        """
        Return 1-sigma position uncertainty (σ_x, σ_y) from the
        state covariance.
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
