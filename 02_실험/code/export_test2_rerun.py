"""test2 결과를 Rerun .rrd 레코딩으로 내보낸다.

실행:
    python export_test2_rerun.py
    rerun ../../03_결과/test2_IR_only/test2_ir_only.rrd
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import rerun as rr


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "03_결과" / "test2_IR_only"
CSV = OUT / "trajectory.csv"
OVERLAY = OUT / "overlay.mp4"
RRD = OUT / "test2_ir_only.rrd"


def num(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value else float("nan")


def finite_xy(row: dict[str, str], x: str, y: str) -> np.ndarray | None:
    p = np.array([num(row, x), num(row, y)], np.float32)
    return p if np.isfinite(p).all() else None


def main() -> None:
    with open(CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cap = cv2.VideoCapture(str(OVERLAY))
    if not cap.isOpened():
        raise RuntimeError(f"오버레이 영상을 열 수 없다: {OVERLAY}")

    rr.init("test2_ir_only_camera_xy", recording_id="test2-ir-only")
    rr.save(RRD)
    rr.log("recording", rr.TextDocument(
        "# test2: IR-only camera XY estimation\n\n"
        "- Blue: central-IR-only KLT estimate\n"
        "- Green: 10 mm LEFT-checkerboard reference\n"
        "- Orange: IMU double-integration, similarity-aligned auxiliary curve\n"
        "- Image stream is the verification overlay: yellow = checkerboard exclusion mask, red = IR ROI.\n"
        "- `mask_violation_points` is expected to remain zero."
    ), static=True)

    ref_history: list[np.ndarray] = []
    ir_history: list[np.ndarray] = []
    imu_history: list[np.ndarray] = []
    for i, row in enumerate(rows):
        ok, bgr = cap.read()
        if not ok:
            raise RuntimeError(f"overlay.mp4 프레임 부족: CSV f{i}를 읽을 수 없다")
        frame = int(row["frame"])
        rr.set_time("frame", sequence=frame)
        rr.set_time("time", duration=float(row["t_s"]))

        # Rerun Image expects RGB; the source is BGR OpenCV.
        rr.log("video/verification_overlay", rr.Image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))

        ref = finite_xy(row, "ref_X_mm", "ref_Y_mm")
        ir = finite_xy(row, "ir_X_mm", "ir_Y_mm")
        imu = finite_xy(row, "sensor_X_aligned_mm", "sensor_Y_aligned_mm")
        if ref is not None:
            ref_history.append(ref)
            rr.log("xy/current/reference", rr.Points2D(ref, colors=[0, 220, 0], radii=3.0,
                                                        labels=["checkerboard REF"]))
        if ir is not None:
            ir_history.append(ir)
            rr.log("xy/current/ir_only", rr.Points2D(ir, colors=[40, 120, 255], radii=3.0,
                                                      labels=["IR-only"] ))
        if imu is not None:
            imu_history.append(imu)
            rr.log("xy/current/imu_auxiliary", rr.Points2D(imu, colors=[255, 150, 40], radii=2.0,
                                                            labels=["IMU aligned (auxiliary)"]))

        if len(ref_history) > 1:
            rr.log("xy/trajectory/reference", rr.LineStrips2D([np.asarray(ref_history)], colors=[0, 220, 0], radii=1.5))
        if len(ir_history) > 1:
            rr.log("xy/trajectory/ir_only", rr.LineStrips2D([np.asarray(ir_history)], colors=[40, 120, 255], radii=1.5))
        if len(imu_history) > 1:
            rr.log("xy/trajectory/imu_auxiliary", rr.LineStrips2D([np.asarray(imu_history)], colors=[255, 150, 40], radii=1.0))

        rr.log("metrics/error_norm_mm", rr.Scalars(num(row, "err_norm_mm")))
        rr.log("metrics/tracked_points", rr.Scalars(num(row, "tracked_points")))
        rr.log("metrics/inlier_ratio", rr.Scalars(num(row, "inlier_ratio")))
        rr.log("metrics/board_xcheck_mm", rr.Scalars(num(row, "board_xcheck_mm")))
        rr.log("metrics/mask_violation_points", rr.Scalars(num(row, "mask_violation_points")))
    cap.release()
    print(f"saved: {RRD}")


if __name__ == "__main__":
    main()
