"""
F.A.L.C.O.N. Vision System
Capstone Project – Human Pose Tracking with YOLO26n-Pose

Features
--------
- IoU-based person tracking with persistent IDs
- Keypoint-driven occlusion detection (VISIBLE / PARTIAL / HEAVY / LOST)
- Kalman-filter position prediction for occluded persons
- Real-time visualization with colour-coded boxes, trajectories, and info overlay
"""

import time

import cv2
import numpy as np
from ultralytics import YOLO

from occlusion import OcclusionState
from tracking import TrackingManager

# ── Configuration ───────────────────────────────────────────────────
MODEL_PATH = "yolo26n-pose.pt"  # relative to project root
CONFIDENCE_THRESHOLD = 0.45   # kept low so tracker gets more candidates
WEBCAM_INDEX = 0
TRAIL_LENGTH = 10             # last N centres drawn as trajectory

# ── Visualisation helpers ───────────────────────────────────────────


def _draw_dashed_rect(
    img: np.ndarray,
    pt1: tuple,
    pt2: tuple,
    colour: tuple,
    thickness: int = 2,
    dash_len: int = 10,
) -> None:
    """Draw a dashed rectangle on *img* (OpenCV has no native dashed rect)."""
    x1, y1 = pt1
    x2, y2 = pt2
    edges = [
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    ]
    for (sx, sy), (ex, ey) in edges:
        dist = int(np.hypot(ex - sx, ey - sy))
        pts_x = np.linspace(sx, ex, max(dist // dash_len, 2)).astype(int)
        pts_y = np.linspace(sy, ey, max(dist // dash_len, 2)).astype(int)
        for i in range(0, len(pts_x) - 1, 2):
            cv2.line(img, (pts_x[i], pts_y[i]),
                     (pts_x[i + 1], pts_y[i + 1]), colour, thickness)


def _draw_uncertainty_ellipse(
    img: np.ndarray,
    center: tuple,
    sigma_x: float,
    sigma_y: float,
    colour: tuple,
) -> None:
    """Draw 2-sigma confidence ellipse around predicted centre."""
    axes = (max(int(2 * sigma_x), 2), max(int(2 * sigma_y), 2))
    cv2.ellipse(img, center, axes, 0, 0, 360, colour, 1, cv2.LINE_AA)


def _draw_trail(
    img: np.ndarray,
    history: list,
    colour: tuple,
    max_len: int = TRAIL_LENGTH,
) -> None:
    """Draw a trajectory polyline from last N centre points."""
    pts = history[-max_len:]
    if len(pts) < 2:
        return
    for i in range(1, len(pts)):
        cx1 = int((pts[i - 1][0] + pts[i - 1][2]) / 2)
        cy1 = int((pts[i - 1][1] + pts[i - 1][3]) / 2)
        cx2 = int((pts[i][0] + pts[i][2]) / 2)
        cy2 = int((pts[i][1] + pts[i][3]) / 2)
        # Fade older segments
        alpha = (i / len(pts))
        fade = tuple(int(c * alpha) for c in colour)
        cv2.line(img, (cx1, cy1), (cx2, cy2), fade, 2, cv2.LINE_AA)


# ── Main loop ───────────────────────────────────────────────────────


def main():
    # ── Load model ──────────────────────────────────────────────────
    print("[FALCON] Loading YOLO26n-Pose model …")
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[FALCON] Using device: {device.upper()}")
    model = YOLO(MODEL_PATH)
    model.to(device)

    # ── Tracking & occlusion subsystems ─────────────────────────────
    tracker = TrackingManager(
        iou_threshold=0.3,
        max_frames_lost=90,
        prediction_decay=0.90,
        recovery_duration=90,
    )

    # ── Open webcam ─────────────────────────────────────────────────
    print("[FALCON] Initialising webcam …")
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print("[FALCON] ERROR: Cannot open webcam.")
        return

    print("[FALCON] Vision system active. Press 'q' to quit.")

    cv2.namedWindow("F.A.L.C.O.N. Vision", cv2.WINDOW_NORMAL)

    fps_timer = time.time()
    fps_value = 0.0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[FALCON] Failed to grab frame – exiting.")
            break

        # ── Run YOLO26 Pose inference ───────────────────────────────
        results = model(frame, verbose=False)
        result = results[0]

        # ── Extract detections ──────────────────────────────────────
        det_boxes = np.empty((0, 4))
        det_confs = np.empty(0)
        det_keypoints = None

        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()                  # (N, 4)
            confs = result.boxes.conf.cpu().numpy()                 # (N,)

            # Filter by confidence threshold
            mask = confs >= CONFIDENCE_THRESHOLD
            det_boxes = xyxy[mask]
            det_confs = confs[mask]

            if result.keypoints is not None and len(result.keypoints) > 0:
                kp_data = result.keypoints.data.cpu().numpy()       # (N, 17, 3)
                det_keypoints = kp_data[mask]

        # ── Update tracker ──────────────────────────────────────────
        tracks = tracker.update(det_boxes, det_keypoints, det_confs, frame=frame)

        # ── Start with the raw frame (we draw our own annotations) ──
        vis = frame.copy()

        # ── Draw per-person annotations ─────────────────────────────
        n_occluded = 0
        n_predicted = 0
        n_recovered = 0

        for t in tracks:
            colour = t.occlusion_state.colour_bgr
            x1, y1, x2, y2 = t.bbox.astype(int)

            # Trajectory trail
            _draw_trail(vis, t.bbox_history, colour)

            if t.is_predicted:
                # Dashed box + uncertainty ellipse for predicted persons
                _draw_dashed_rect(vis, (x1, y1), (x2, y2), colour, 2)
                sig_x, sig_y = t.predictor.get_uncertainty()
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                _draw_uncertainty_ellipse(vis, (cx, cy), sig_x, sig_y, colour)
                label = f"Person #{t.track_id} (PRED {t.prediction_confidence:.0%})"
                n_predicted += 1
            elif t.is_recovered:
                # Recovered person: solid box with thicker outline + recovery tag
                cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 3)
                secs_left = t.recovery_frames_remaining / 30.0
                label = f"Person #{t.track_id} RECOVERED ({secs_left:.1f}s)"
                n_recovered += 1

                # Draw keypoints if available
                if t.keypoints is not None:
                    for kp_idx in range(len(t.keypoints)):
                        kx, ky, kc = t.keypoints[kp_idx]
                        if kc > 0.4:
                            cv2.circle(vis, (int(kx), int(ky)), 3, colour, -1)
            else:
                # Solid box for detected persons
                cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 2)
                label = f"Person #{t.track_id}"

                # Draw keypoints if available
                if t.keypoints is not None:
                    for kp_idx in range(len(t.keypoints)):
                        kx, ky, kc = t.keypoints[kp_idx]
                        if kc > 0.4:
                            cv2.circle(vis, (int(kx), int(ky)), 3, colour, -1)

            # Occlusion state label
            occ_label = t.occlusion_state.label
            cv2.putText(vis, label, (x1, y1 - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)
            cv2.putText(vis, occ_label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)

            if t.occlusion_state in (
                OcclusionState.PARTIALLY_OCCLUDED,
                OcclusionState.HEAVILY_OCCLUDED,
                OcclusionState.LOST,
            ):
                n_occluded += 1

        # ── FPS calculation ─────────────────────────────────────────
        frame_count += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 0.5:
            fps_value = frame_count / elapsed
            frame_count = 0
            fps_timer = time.time()

        # ── Info overlay (top-left) ─────────────────────────────────
        overlay_lines = [
            f"FPS: {fps_value:.1f}",
            f"Tracked: {len(tracks)}",
            f"Occluded: {n_occluded}",
            f"Predicted: {n_predicted}",
            f"Recovered: {n_recovered}",
        ]
        for i, line in enumerate(overlay_lines):
            cv2.putText(
                vis, line, (10, 25 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
            )

        # ── Status banner ─────────────────────────────────────────
        if n_occluded == 0:
            cv2.putText(vis, "STATE: ALL VISIBLE", (10, vis.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2, cv2.LINE_AA)

        # ── Display ────────────────────────────────────────────────
        cv2.imshow("F.A.L.C.O.N. Vision", vis)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[FALCON] Quit signal received.")
            break

    # ── Cleanup ─────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    print("[FALCON] Vision system shut down.")


if __name__ == "__main__":
    main()