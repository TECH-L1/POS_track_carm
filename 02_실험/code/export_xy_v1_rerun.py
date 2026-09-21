"""XY추정_v1의 REF/EST 분리 영상을 Rerun .rrd로 묶는다.

선행 조건:
    report.overlay_video(ref, fixed, 'ref')
    report.overlay_video(ref, fixed, 'est')

실행:
    python export_xy_v1_rerun.py
    rerun ../../03_결과/XY추정_v1/xy_v1_ref_est.rrd
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import rerun as rr


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "03_결과" / "XY추정_v1"
DATA = OUT / "data"
VISUALS = OUT / "visuals"
RERUN = OUT / "rerun"
CSV = DATA / "trajectory.csv"
DASHBOARD_VIDEO = VISUALS / "ref_est_trajectory_dashboard.mp4"
RRD = RERUN / "xy_v1_ref_est.rrd"


def value(row: dict[str, str], key: str) -> float:
    x = row.get(key, "")
    return float(x) if x else float("nan")


def point(row: dict[str, str], x: str, y: str) -> np.ndarray | None:
    p = np.array([value(row, x), value(row, y)], np.float32)
    return p if np.isfinite(p).all() else None


def jpeg_image(bgr: np.ndarray) -> rr.EncodedImage:
    """RRD가 원시 RGB 프레임으로 비대해지지 않도록 JPEG 스트림으로 기록한다."""
    ok, data = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("Rerun JPEG 인코딩 실패")
    return rr.EncodedImage(contents=data.tobytes(), media_type="image/jpeg")


def main() -> None:
    with open(CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cap = cv2.VideoCapture(str(DASHBOARD_VIDEO))
    if not cap.isOpened():
        raise RuntimeError("Dashboard video is missing. Run compose_xy_v1_dashboard.py first.")

    rr.init("xy_v1_ref_est", recording_id="xy-v1-ref-est")
    RERUN.mkdir(parents=True, exist_ok=True)
    rr.save(RRD)
    rr.log("recording", rr.TextDocument(
        "# XY v1 - REF and EST Dashboard Validation\n\n"
        "- `video/dashboard`: synchronized REF-only, EST-only, XY trajectory, error, and RMSE panels.\n"
        "- `xy/trajectory`: green = REF, blue = IR-only EST.\n"
        "- `metrics/cumulative_rmse_mm`: running RMSE through the current frame.\n"
        "- `metrics/error_norm_mm`: per-frame |EST - REF|."
    ), static=True)

    ref_hist: list[np.ndarray] = []
    est_hist: list[np.ndarray] = []
    squared_errors: list[float] = []
    for row in rows:
        frame = int(row["frame"])
        # Dashboard frames are sampled every two source frames: its kth frame represents 2k.
        if frame % 2:
            continue
        ok, bgr = cap.read()
        if not ok:
            raise RuntimeError(f"Dashboard video ended at f{row['frame']}")
        rr.set_time("frame", sequence=frame)
        rr.set_time("time", duration=float(row["t_s"]))
        rr.log("video/dashboard", jpeg_image(bgr))

        ref = point(row, "ref_X_mm", "ref_Y_mm")
        est = point(row, "est_X_mm", "est_Y_mm")
        if ref is not None:
            ref_hist.append(ref)
            rr.log("xy/current/ref", rr.Points2D(ref, colors=[0, 220, 0], radii=3.0, labels=["REF"]))
        if est is not None:
            est_hist.append(est)
            rr.log("xy/current/est", rr.Points2D(est, colors=[40, 120, 255], radii=3.0, labels=["EST"]))
        if len(ref_hist) > 1:
            rr.log("xy/trajectory/ref", rr.LineStrips2D([np.asarray(ref_hist)], colors=[0, 220, 0], radii=1.5))
        if len(est_hist) > 1:
            rr.log("xy/trajectory/est", rr.LineStrips2D([np.asarray(est_hist)], colors=[40, 120, 255], radii=1.5))

        error = value(row, "err_norm_mm")
        if np.isfinite(error):
            squared_errors.append(error * error)
        rr.log("metrics/error_norm_mm", rr.Scalars(error))
        rr.log("metrics/cumulative_rmse_mm", rr.Scalars(np.sqrt(np.mean(squared_errors))))
        rr.log("metrics/tracked_points", rr.Scalars(value(row, "tracked_points")))
        rr.log("metrics/inlier_ratio", rr.Scalars(value(row, "inlier_ratio")))
        rr.log("metrics/ref_xcheck_mm", rr.Scalars(value(row, "xcheck_mm")))

    cap.release()
    print(f"saved: {RRD}")


if __name__ == "__main__":
    main()
