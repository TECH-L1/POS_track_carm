"""REF-only, EST-only, and XY trajectory into one synchronized review video."""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "03_결과" / "XY추정_v1"
DATA = OUT / "data"
VISUALS = OUT / "visuals"
REF_VIDEO = VISUALS / "overlay_ref.mp4"
EST_VIDEO = VISUALS / "overlay_est.mp4"
CSV = DATA / "trajectory.csv"
OUTPUT = VISUALS / "ref_est_trajectory_dashboard.mp4"

OUT_W, OUT_H = 1280, 720
TOP_W, TOP_H = 640, 360
BOTTOM_Y, BOTTOM_H = TOP_H, OUT_H - TOP_H
PLAYBACK_FPS = 30.0
FRAME_STEP = 2  # Source overlays are 30 fps but represent a 59.4 fps experiment.


def load_rows():
    with open(CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ref = np.array([[float(r["ref_X_mm"]), float(r["ref_Y_mm"])] for r in rows], np.float64)
    est = np.array([[float(r["est_X_mm"]), float(r["est_Y_mm"])] for r in rows], np.float64)
    err = np.array([float(r["err_norm_mm"]) for r in rows], np.float64)
    return rows, ref, est, err


def make_mapper(ref: np.ndarray, est: np.ndarray):
    pts = np.vstack([ref, est])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    pad = np.maximum((hi - lo) * .06, 12.0)
    lo, hi = lo - pad, hi + pad
    # Keep the plot area square in its coordinate mapping; this avoids visually changing
    # the relative X/Y trajectory geometry.
    span = max(*(hi - lo))
    center = (lo + hi) / 2
    lo, hi = center - span / 2, center + span / 2
    x0, y0, size = 24, BOTTOM_Y + 36, 300

    def map_xy(p):
        q = (p - lo) / (hi - lo)
        return (int(x0 + q[0] * size), int(y0 + (1.0 - q[1]) * size))

    return map_xy, (x0, y0, size), lo, hi


def put_text(image, text, xy, scale=.55, color=(230, 230, 230), thickness=1):
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def main():
    rows, ref, est, err = load_rows()
    cap_ref, cap_est = cv2.VideoCapture(str(REF_VIDEO)), cv2.VideoCapture(str(EST_VIDEO))
    if not cap_ref.isOpened() or not cap_est.isOpened():
        raise RuntimeError("Missing overlay_ref.mp4 or overlay_est.mp4")
    writer = cv2.VideoWriter(str(OUTPUT), cv2.VideoWriter_fourcc(*"mp4v"), PLAYBACK_FPS, (OUT_W, OUT_H))
    mapper, (x0, y0, size), lo, hi = make_mapper(ref, est)
    rmse = np.sqrt(np.cumsum(err * err) / np.arange(1, len(err) + 1))

    for i, row in enumerate(rows):
        ok_r, frame_r = cap_ref.read()
        ok_e, frame_e = cap_est.read()
        if not ok_r or not ok_e:
            raise RuntimeError(f"Source overlay ended at frame {i}")
        if i % FRAME_STEP:
            continue

        canvas = np.zeros((OUT_H, OUT_W, 3), np.uint8)
        canvas[:TOP_H, :TOP_W] = cv2.resize(frame_r, (TOP_W, TOP_H))
        canvas[:TOP_H, TOP_W:] = cv2.resize(frame_e, (TOP_W, TOP_H))
        cv2.line(canvas, (TOP_W, 0), (TOP_W, TOP_H), (100, 100, 100), 1)
        cv2.line(canvas, (0, TOP_H), (OUT_W, TOP_H), (100, 100, 100), 1)

        # Bottom-left: equal-axis XY trajectory plot.
        put_text(canvas, "XY trajectory [mm]", (x0, BOTTOM_Y + 24), .55)
        cv2.rectangle(canvas, (x0, y0), (x0 + size, y0 + size), (140, 140, 140), 1)
        for frac in (.25, .50, .75):
            cv2.line(canvas, (x0 + int(size * frac), y0), (x0 + int(size * frac), y0 + size), (55, 55, 55), 1)
            cv2.line(canvas, (x0, y0 + int(size * frac)), (x0 + size, y0 + int(size * frac)), (55, 55, 55), 1)
        k = i + 1
        for track, color in ((ref[:k], (0, 220, 0)), (est[:k], (255, 140, 40))):
            q = np.array([mapper(p) for p in track], np.int32)
            if len(q) > 1:
                cv2.polylines(canvas, [q], False, color, 2, cv2.LINE_AA)
            cv2.circle(canvas, tuple(q[-1]), 5, color, -1, cv2.LINE_AA)
        put_text(canvas, "REF", (x0 + size + 12, y0 + 20), .50, (0, 220, 0), 2)
        put_text(canvas, "EST", (x0 + size + 12, y0 + 44), .50, (255, 140, 40), 2)
        put_text(canvas, f"X range: {lo[0]:.0f} to {hi[0]:.0f} mm", (x0, y0 + size + 23), .40)
        put_text(canvas, f"Y range: {lo[1]:.0f} to {hi[1]:.0f} mm", (x0, y0 + size + 42), .40)

        # Bottom-right: synchronized numerical status and a compact running-error plot.
        bx, by = 470, BOTTOM_Y + 34
        put_text(canvas, "Synchronized status", (bx, by), .62, (255, 255, 255), 2)
        put_text(canvas, f"Frame: {i:04d}   Time: {float(row['t_s']):.2f} s", (bx, by + 34), .52)
        put_text(canvas, f"REF: ({ref[i,0]:+.2f}, {ref[i,1]:+.2f}) mm", (bx, by + 66), .52, (0, 220, 0))
        put_text(canvas, f"EST: ({est[i,0]:+.2f}, {est[i,1]:+.2f}) mm", (bx, by + 98), .52, (255, 140, 40))
        put_text(canvas, f"Instant error: {err[i]:.3f} mm", (bx, by + 130), .52, (80, 180, 255))
        put_text(canvas, f"Running RMSE: {rmse[i]:.3f} mm", (bx, by + 162), .52, (80, 180, 255))
        put_text(canvas, f"KLT points: {row['tracked_points']}   Inlier ratio: {float(row['inlier_ratio']):.3f}", (bx, by + 194), .52)

        ex0, ey0, ew, eh = 470, BOTTOM_Y + 245, 760, 86
        cv2.rectangle(canvas, (ex0, ey0), (ex0 + ew, ey0 + eh), (140, 140, 140), 1)
        mx = max(7.0, float(err.max()) * 1.08)
        count = i + 1
        series = err[:count]
        if count > 1:
            q = np.column_stack([
                ex0 + np.linspace(0, ew, count),
                ey0 + eh - series / mx * eh,
            ]).astype(np.int32)
            cv2.polylines(canvas, [q], False, (80, 180, 255), 1, cv2.LINE_AA)
        put_text(canvas, "Instant error [mm]", (ex0, ey0 - 8), .42, (80, 180, 255))
        put_text(canvas, f"0", (ex0 - 12, ey0 + eh), .35)
        put_text(canvas, f"{mx:.1f}", (ex0 - 20, ey0 + 10), .35)

        writer.write(canvas)

    cap_ref.release()
    cap_est.release()
    writer.release()
    print(f"saved: {OUTPUT}")


if __name__ == "__main__":
    main()
