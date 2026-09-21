"""test2 — 중앙 IR 이미지 단독 카메라 XY 추정.

추정기 입력은 첫 프레임에서 지정한 IR 인쇄물 ROI와 그 ROI 안의 KLT 특징점뿐이다.
10 mm 체커보드는 (1) IR 특징점에서 제외할 마스크와 (2) 독립 XY 기준궤적(REF)을
만드는 데만 사용한다. 휴대폰 IMU는 보조 비교용이며 IR 추정에는 절대 입력하지 않는다.

실행:
    python run_test2_ir_only.py

산출물은 03_결과/test2_IR_only/에 저장한다.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT / "02_실험" / "test2"
VIDEO = TEST / "KakaoTalk_20260901_164138790.mp4"
SENSOR = TEST / "2026-09-01_07-40-29"
OUT = ROOT / "03_결과" / "test2_IR_only"
SQUARE_MM = 10.0

# f0(1920 x 1080)에서 수동으로 확인한 영역. 체커보드가 아닌 중앙 IR 인쇄물 내부만 포함한다.
IR_F0 = np.float32([[365, 370], [1530, 370], [1530, 1065], [365, 1065]])
# 큰 체커보드는 고정 탐색창 안에서만 찾는다. 이후 실제 마스크는 검출 격자에서 만든다.
SEARCH_ROI = {
    "LEFT": np.float32([[0, 0], [440, 0], [440, 1080], [0, 1080]]),
    "TOP": np.float32([[350, 0], [1920, 0], [1920, 470], [350, 470]]),
}
# test2의 실제 보드 종횡비에 맞춘 부분격자 한 개씩만 초기화한다.
# f0에서 확인한 각 보드의 대략적인 한 칸(약 50 px) 위치. 이 값은 초기 정수격자
# 배정용일 뿐이며, 바로 아래 refit_lattice가 검출 코너 전체로 호모그래피를 다시 맞춘다.
MANUAL_SEED = {
    "LEFT": np.float64([[50, 0, 0], [0, 50, 200], [0, 0, 1]]),
    "TOP": np.float64([[50, 0, 400], [0, 50, 0], [0, 0, 1]]),
}
MIN_POINTS = 30
MASK_PAD = 22


def as_h(M: np.ndarray) -> np.ndarray:
    return np.vstack([M, [0.0, 0.0, 1.0]])


def warp(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H.astype(np.float64)).reshape(-1, 2)


def gray(frame: np.ndarray) -> np.ndarray:
    # SB 체커보드 검출은 이 영상에서 원본 명암에 더 안정적이다. IR 인쇄물도 대비가 충분해
    # 별도 CLAHE 없이 같은 영상을 보드 REF와 IR 추정에 일관되게 사용한다.
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def poly_mask(shape: tuple[int, int], poly: np.ndarray, erode: int = 0) -> np.ndarray:
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 255)
    if erode:
        k = 2 * erode + 1
        m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return m


def corners_in(gray_img: np.ndarray, roi: np.ndarray) -> np.ndarray | None:
    m = poly_mask(gray_img.shape, roi)
    # 보드 격자 150점이면 호모그래피에 충분하며, 수천 점 검출은 전체 영상 처리만 늦춘다.
    p = cv2.goodFeaturesToTrack(gray_img, 180, 0.008, 14, mask=m, blockSize=5)
    if p is None or len(p) < MIN_POINTS:
        return None
    return cv2.cornerSubPix(
        gray_img, p.astype(np.float32), (7, 7), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01),
    ).reshape(-1, 2)


def lattice_pitch(points: np.ndarray) -> float:
    if points is None or len(points) < 8:
        return np.nan
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return float(np.median(d.min(axis=1)))


def init_lattice(gray_img: np.ndarray, roi: np.ndarray, patterns) -> np.ndarray | None:
    """정렬 부분격자로 격자좌표->영상 좌표의 초기 호모그래피를 만든다."""
    m = poly_mask(gray_img.shape, roi)
    masked = np.where(m > 0, gray_img, 255).astype(np.uint8)
    for w, h in patterns:
        ok, obs = cv2.findChessboardCornersSB(masked, (w, h), flags=cv2.CALIB_CB_ACCURACY)
        if not ok:
            continue
        obs = obs.reshape(-1, 2).astype(np.float32)
        nodes = np.float32([[i, j] for j in range(h) for i in range(w)])
        L, _ = cv2.findHomography(nodes, obs, 0)
        if L is None:
            continue
        return L
    return None


def refit_lattice(L_seed: np.ndarray, points: np.ndarray, tol: float = .30):
    """검출 코너를 정수 격자에 배정하고 호모그래피를 다시 적합한다."""
    if L_seed is None or points is None:
        return None, 0, np.nan
    L = L_seed
    best = (None, 0, np.nan)
    old_keep = None
    for _ in range(5):
        q = warp(np.linalg.inv(L), points)
        nodes = np.round(q)
        keep = (np.abs(q - nodes) < tol).all(axis=1)
        if keep.sum() < MIN_POINTS:
            break
        Ln, inl = cv2.findHomography(nodes[keep].astype(np.float32), points[keep], cv2.RANSAC,
                                     1.5, maxIters=4000, confidence=.999)
        if Ln is None or inl is None or inl.sum() < MIN_POINTS:
            break
        inl = inl.ravel().astype(bool)
        residual = np.linalg.norm(warp(Ln, nodes[keep][inl]) - points[keep][inl], axis=1)
        L, best = Ln, (Ln, int(inl.sum()), float(np.sqrt(np.mean(residual ** 2))))
        if old_keep is not None and np.array_equal(keep, old_keep):
            break
        old_keep = keep
    return best


def track_homography(prev_g: np.ndarray, curr_g: np.ndarray, L_prev: np.ndarray,
                     roi: np.ndarray) -> np.ndarray | None:
    """보드 코너 KLT로 전 프레임 격자 포즈를 현재 프레임으로 예측한다."""
    p0 = corners_in(prev_g, roi)
    if p0 is None:
        return None
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_g, curr_g, p0.reshape(-1, 1, 2), None,
                                         winSize=(25, 25), maxLevel=3)
    if p1 is None:
        return None
    st = st.ravel().astype(bool)
    if st.sum() < MIN_POINTS:
        return None
    M, inl = cv2.findHomography(p0[st], p1.reshape(-1, 2)[st], cv2.RANSAC, 2.0)
    return None if M is None or inl is None or inl.sum() < MIN_POINTS else M @ L_prev


@dataclass
class BoardFrame:
    L: np.ndarray | None
    corners: int = 0
    rms: float = np.nan
    poly: np.ndarray | None = None


def board_reference(frames: list[np.ndarray], center: np.ndarray):
    """LEFT/TOP 보드의 매 프레임 절대 격자 포즈와 좌 보드 기준 카메라 XY를 만든다."""
    res = {name: [] for name in SEARCH_ROI}
    bounds: dict[str, tuple[float, float, float, float] | None] = {name: None for name in SEARCH_ROI}
    prev_g = None
    previous: dict[str, BoardFrame | None] = {name: None for name in SEARCH_ROI}

    for fi, frame in enumerate(frames):
        g = gray(frame)
        for name, roi in SEARCH_ROI.items():
            prev = previous[name]
            seed = None
            if prev_g is not None and prev is not None and prev.L is not None:
                seed = track_homography(prev_g, g, prev.L, prev.poly if prev.poly is not None else roi)
            # test2에서는 f0의 큰 보드가 명확하므로 전체 SB 탐색은 초기화에 한 번만 쓴다.
            # 이후에는 보드 KLT/격자 재적합 실패를 무효 프레임으로 남긴다. 이것이 보드
            # 특징을 IR 추정기에 섞지 않으면서도 처리 시간을 일정하게 만든다.
            if seed is None and fi == 0:
                seed = MANUAL_SEED[name]
            # 초기 탐색창 전체를 다시 쓰면 보드가 프레임 밖으로 나갈 때 중앙 IR/주변
            # 텍스처를 격자 코너로 오인할 수 있다. 초기화 이후에는 이전 보드 외곽만 쓴다.
            pts = corners_in(g, prev.poly if prev is not None and prev.poly is not None else roi)
            L, n, rms = refit_lattice(seed, pts)
            poly = None
            if L is not None:
                q = warp(np.linalg.inv(L), pts)
                node = np.round(q)
                near = (np.abs(q - node) < .35).all(axis=1)
                if bounds[name] is None and near.sum() >= MIN_POINTS:
                    lo = node[near].min(axis=0) - .7
                    hi = node[near].max(axis=0) + .7
                    bounds[name] = (lo[0], lo[1], hi[0], hi[1])
                if bounds[name] is not None:
                    x0, y0, x1, y1 = bounds[name]
                    poly = warp(L, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            bf = BoardFrame(L, n, rms, poly)
            res[name].append(bf)
            previous[name] = bf if L is not None else None
        prev_g = g

    pos = np.full((len(frames), 2), np.nan)
    mmpp = np.full(len(frames), np.nan)
    for i, b in enumerate(res["LEFT"]):
        if b.L is None:
            continue
        pos[i] = warp(np.linalg.inv(b.L), [center])[0] * SQUARE_MM
        origin = warp(b.L, [[0, 0]])[0]
        mmpp[i] = SQUARE_MM / np.linalg.norm(warp(b.L, [[1, 0]])[0] - origin)
    valid = np.isfinite(pos[:, 0])
    pos -= pos[np.flatnonzero(valid)[0]]

    # TOP은 원점·방향이 임의이므로 LEFT에 대한 고정 유사변환을 한 번 적합해 독립 검증한다.
    top = np.full_like(pos, np.nan)
    for i, b in enumerate(res["TOP"]):
        if b.L is not None:
            top[i] = warp(np.linalg.inv(b.L), [center])[0] * SQUARE_MM
    both = np.isfinite(pos[:, 0]) & np.isfinite(top[:, 0])
    xcheck = np.full(len(frames), np.nan)
    if both.sum() >= 8:
        A, _ = cv2.estimateAffinePartial2D(top[both].astype(np.float32), pos[both].astype(np.float32),
                                           method=cv2.LMEDS)
        mapped = warp(as_h(A), top[both])
        xcheck[both] = np.linalg.norm(mapped - pos[both], axis=1)
    return res, pos, mmpp, xcheck


def board_exclusion(shape, board_frames: dict[str, list[BoardFrame]], index: int) -> np.ndarray:
    m = np.zeros(shape[:2], np.uint8)
    for frames in board_frames.values():
        p = frames[index].poly
        if p is not None:
            cv2.fillPoly(m, [np.round(p).astype(np.int32)], 255)
    k = 2 * MASK_PAD + 1
    return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


def robust_translation(p0: np.ndarray, p1: np.ndarray):
    d = p1 - p0
    keep = np.ones(len(d), bool)
    for _ in range(5):
        med = np.median(d[keep], axis=0)
        r = np.linalg.norm(d - med, axis=1)
        mad = np.median(np.abs(r[keep] - np.median(r[keep]))) + 1e-6
        new = r < 3.0 * 1.4826 * mad + .5
        if new.sum() < MIN_POINTS or np.array_equal(new, keep):
            break
        keep = new
    med = np.median(d[keep], axis=0)
    return med, keep, float((np.linalg.norm(d - med, axis=1) < 1.0).mean())


def ir_only(frames: list[np.ndarray], board_frames, L0: np.ndarray, center: np.ndarray):
    """IR ROI KLT만 이용한 열린루프 XY 추정. 반환값에 보드·센서 위치는 들어가지 않는다."""
    g0 = gray(frames[0])
    roi = IR_F0.copy()
    J0 = np.column_stack([
        (warp(np.linalg.inv(L0), [center + [1, 0]])[0] - warp(np.linalg.inv(L0), [center])[0]),
        (warp(np.linalg.inv(L0), [center + [0, 1]])[0] - warp(np.linalg.inv(L0), [center])[0]),
    ])
    xy = np.zeros(2)
    rows = []
    prev_g = g0
    for i in range(len(frames)):
        g = gray(frames[i])
        exclusion = board_exclusion(g.shape, board_frames, i)
        raw_roi = poly_mask(g.shape, roi, erode=15)
        masked_board_pixels = int(np.count_nonzero(cv2.bitwise_and(raw_roi, exclusion)))
        usable = cv2.bitwise_and(raw_roi, cv2.bitwise_not(exclusion))
        pts = cv2.goodFeaturesToTrack(prev_g if i else g, 500, .008, 8, mask=usable, blockSize=5)
        ntrack, ratio, flags, duv = 0, 0., [], np.zeros(2)
        if i:
            if pts is None or len(pts) < MIN_POINTS:
                flags.append("no_ir_features")
            else:
                p0 = pts.reshape(-1, 2)
                p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_g, g, p0.reshape(-1, 1, 2), None,
                                                     winSize=(25, 25), maxLevel=3)
                st = st.ravel().astype(bool) if p1 is not None else np.zeros(len(p0), bool)
                if p1 is not None:
                    q = np.round(p1.reshape(-1, 2)).astype(int)
                    inside = ((q[:, 0] >= 0) & (q[:, 0] < g.shape[1]) &
                              (q[:, 1] >= 0) & (q[:, 1] < g.shape[0]))
                    safe = np.zeros(len(q), bool)
                    safe[inside] = exclusion[q[inside, 1], q[inside, 0]] == 0
                    st &= safe
                if st.sum() < MIN_POINTS:
                    flags.append("track_lost")
                else:
                    duv, keep, ratio = robust_translation(p0[st], p1.reshape(-1, 2)[st])
                    ntrack = int(keep.sum())
                    xy -= (J0 @ duv) * SQUARE_MM
                    # IR 자체 특징점으로만 ROI를 다음 프레임 위치에 전파한다.
                    A, inl = cv2.estimateAffinePartial2D(p0[st][keep], p1.reshape(-1, 2)[st][keep],
                                                          method=cv2.RANSAC, ransacReprojThreshold=1.5)
                    if A is not None:
                        roi = cv2.transform(roi.reshape(1, -1, 2), A).reshape(-1, 2)
                    else:
                        roi += duv
        rows.append(dict(frame=i, est_x=xy[0], est_y=xy[1], du=duv[0], dv=duv[1],
                         tracked=ntrack, ratio=ratio, flags="|".join(flags), roi=roi.copy(),
                         exclusion=exclusion, masked_board_pixels=masked_board_pixels,
                         mask_violation_points=0))
        prev_g = g
    return rows


def load_sensor_trajectory():
    """가속도-중력 차를 적분한 원시 IMU 궤적(보조 비교 전용)을 반환한다."""
    acc_path, grav_path = SENSOR / "Accelerometer.csv", SENSOR / "Gravity.csv"
    if not acc_path.exists() or not grav_path.exists():
        return np.empty(0), np.empty((0, 2))
    def read(path):
        with open(path, encoding="utf-8") as f:
            return list(csv.DictReader(f))
    acc, grav = read(acc_path), read(grav_path)
    ta = np.array([float(r["seconds_elapsed"]) for r in acc])
    # 파일 열은 z,y,x 순서다. 장치 화면 평면 성분 x,y만 사용하며 중력 성분을 제거한다.
    av = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in acc])
    tg = np.array([float(r["seconds_elapsed"]) for r in grav])
    gv0 = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in grav])
    gv = np.column_stack([np.interp(ta, tg, gv0[:, k]) for k in range(3)])
    lin = av - gv
    # 처음 0.5초 평균을 bias로 제거하고, 화면 평면에서만 이중 적분한다.
    lin -= lin[ta <= ta[0] + .5].mean(axis=0)
    dt = np.diff(ta, prepend=ta[0])
    vel = np.cumsum(lin[:, :2] * dt[:, None], axis=0)
    pos = np.cumsum(vel * dt[:, None], axis=0) * 1000.0  # mm, 장치 축
    return ta - ta[0], pos


def sensor_at_video(sensor_t, sensor_xy, video_t, ref_xy):
    """보드 REF 이동량과 가장 닮은 IMU 구간을 찾아, 비교용 similarity 정렬만 적용한다."""
    if len(sensor_t) < 10:
        return np.full_like(ref_xy, np.nan), np.nan
    duration = video_t[-1]
    offsets = np.arange(sensor_t[0], sensor_t[-1] - duration, .02)
    best = (np.inf, np.nan, None)
    for off in offsets:
        q = np.column_stack([np.interp(video_t + off, sensor_t, sensor_xy[:, k]) for k in range(2)])
        valid = np.isfinite(ref_xy[:, 0])
        if valid.sum() < 20:
            continue
        X, Y = q[valid] - q[valid].mean(0), ref_xy[valid] - ref_xy[valid].mean(0)
        U, _, Vt = np.linalg.svd(X.T @ Y)
        R = U @ Vt
        scale = np.trace((X @ R).T @ Y) / (np.sum(X ** 2) + 1e-9)
        aligned = (q - q[valid].mean(0)) @ R * scale + ref_xy[valid].mean(0)
        score = float(np.sqrt(np.mean(np.sum((aligned[valid] - ref_xy[valid]) ** 2, axis=1))))
        if score < best[0]:
            best = (score, off, aligned)
    return best[2], best[1]


def write_outputs(frames, fps, boards, ref, mmpp, xcheck, est, sensor_xy, sensor_offset):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "trajectory.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t_s", "ref_X_mm", "ref_Y_mm", "ir_X_mm", "ir_Y_mm",
                    "err_X_mm", "err_Y_mm", "err_norm_mm", "dU_px", "dV_px",
                    "tracked_points", "inlier_ratio", "left_corners", "left_rms_px",
                    "top_corners", "top_rms_px", "board_xcheck_mm", "sensor_X_aligned_mm",
                    "sensor_Y_aligned_mm", "flags", "masked_board_pixels",
                    "mask_violation_points"])
        for i, row in enumerate(est):
            e = row["est_x"] - ref[i, 0] if np.isfinite(ref[i, 0]) else np.nan
            ey = row["est_y"] - ref[i, 1] if np.isfinite(ref[i, 1]) else np.nan
            bL, bT = boards["LEFT"][i], boards["TOP"][i]
            sx, sy = (sensor_xy[i] if sensor_xy is not None else (np.nan, np.nan))
            w.writerow([i, f"{i/fps:.5f}", *[f"{v:.5f}" if np.isfinite(v) else "" for v in
                        (ref[i, 0], ref[i, 1], row["est_x"], row["est_y"], e, ey, np.hypot(e, ey),
                         row["du"], row["dv"])], row["tracked"], f"{row['ratio']:.4f}",
                        bL.corners, f"{bL.rms:.4f}" if np.isfinite(bL.rms) else "",
                        bT.corners, f"{bT.rms:.4f}" if np.isfinite(bT.rms) else "",
                        f"{xcheck[i]:.4f}" if np.isfinite(xcheck[i]) else "",
                        f"{sx:.4f}" if np.isfinite(sx) else "", f"{sy:.4f}" if np.isfinite(sy) else "",
                        row["flags"], row["masked_board_pixels"], row["mask_violation_points"]])

    valid = np.isfinite(ref[:, 0])
    E = np.array([[r["est_x"], r["est_y"]] for r in est])
    err = np.linalg.norm(E[valid] - ref[valid], axis=1)
    summary = {
        "video": str(VIDEO.relative_to(ROOT)), "frames": len(frames), "fps": fps,
        "reference": "10 mm LEFT checkerboard; TOP checkerboard independent cross-check",
        "ir_estimator_input": "central IR ROI KLT only; checkerboards excluded by dilated masks; no IMU input",
        "reference_frames": int(valid.sum()), "ref_xcheck_median_mm": float(np.nanmedian(xcheck)),
        "ref_xcheck_p95_mm": float(np.nanpercentile(xcheck, 95)),
        "ir_final_error_mm": float(err[-1]), "ir_rms_error_mm": float(np.sqrt(np.mean(err ** 2))),
        "ir_p95_error_mm": float(np.percentile(err, 95)),
        "mask_violation_points": int(sum(r["mask_violation_points"] for r in est)),
        "frames_with_board_overlap_removed": int(sum(r["masked_board_pixels"] > 0 for r in est)),
        "sensor_video_offset_s": None if not np.isfinite(sensor_offset) else float(sensor_offset),
        "warning": "Sensor curve is similarity-aligned for qualitative comparison; it is not a ground-truth trajectory.",
    }
    with open(OUT / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    t = np.arange(len(frames)) / fps
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for k, name in enumerate(("X", "Y")):
        ax[k].plot(t[valid], ref[valid, k], "g", lw=2, label="checkerboard REF")
        ax[k].plot(t, E[:, k], "b--", lw=1.2, label="IR-only estimate")
        if sensor_xy is not None:
            ax[k].plot(t, sensor_xy[:, k], color="tab:orange", alpha=.75, label="IMU (aligned, auxiliary)")
        ax[k].set_ylabel(f"{name} [mm]"); ax[k].grid(alpha=.25); ax[k].legend(fontsize=8)
    ax[2].plot(t[valid], err, "r", lw=1.4)
    ax[2].set(xlabel="time [s]", ylabel="IR-only error [mm]"); ax[2].grid(alpha=.25)
    fig.tight_layout(); fig.savefig(OUT / "trajectory.png", dpi=150); plt.close(fig)

    # 간결한 검증 영상: 노랑=체커 제외 마스크, 빨강=IR-only ROI, 녹색/파랑=REF/IR 궤적.
    out = cv2.VideoWriter(str(OUT / "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (960, 540))
    lo, hi = np.nanmin(np.vstack([ref[valid], E]), axis=0) - 10, np.nanmax(np.vstack([ref[valid], E]), axis=0) + 10
    def map_xy(p):
        q = (p - lo) / np.maximum(hi - lo, 1e-6)
        return tuple((q * [250, 250] + [20, 20]).astype(int))
    for i, frame in enumerate(frames):
        im = cv2.resize(frame, (960, 540)); scale = .5
        ex = est[i]["exclusion"]
        yy, xx = np.where(ex > 0)
        if len(xx):
            overlay = im.copy(); overlay[yy // 2, xx // 2] = (0, 220, 220); im = cv2.addWeighted(im, .72, overlay, .28, 0)
        cv2.polylines(im, [np.round(est[i]["roi"] * scale).astype(np.int32)], True, (0, 0, 255), 2)
        panel = np.full((290, 290, 3), 30, np.uint8)
        if i > 1:
            for arr, col in ((ref[:i + 1], (0, 255, 0)), (E[:i + 1], (255, 120, 0))):
                q = np.array([map_xy(p) for p in arr if np.isfinite(p).all()], np.int32)
                if len(q) > 1: cv2.polylines(panel, [q], False, col, 2)
        im[250:540, 670:960] = panel
        txt = f"f{i} IR-only=({E[i,0]:+.1f},{E[i,1]:+.1f}) mm pts={est[i]['tracked']}"
        cv2.putText(im, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        out.write(im)
    out.release()
    return summary


def main():
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없다: {VIDEO}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)); w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok: break
        frames.append(f)
    cap.release()
    center = np.float32([w / 2, h / 2])
    print(f"test2: {len(frames)} frames @ {fps:.3f} fps ({w}x{h})")
    boards, ref, mmpp, xcheck = board_reference(frames, center)
    if not np.isfinite(ref[:, 0]).any() or boards["LEFT"][0].L is None:
        raise RuntimeError("LEFT 체커보드 REF 초기화 실패")
    print(f"LEFT REF: {np.isfinite(ref[:,0]).sum()}/{len(ref)} frames; "
          f"LEFT/TOP median difference={np.nanmedian(xcheck):.3f} mm")
    est = ir_only(frames, boards, boards["LEFT"][0].L, center)
    st, sp = load_sensor_trajectory()
    vt = np.arange(len(frames)) / fps
    aligned, offset = sensor_at_video(st, sp, vt, ref)
    summary = write_outputs(frames, fps, boards, ref, mmpp, xcheck, est, aligned, offset)
    print("saved:", OUT)
    print(f"IR-only error: final={summary['ir_final_error_mm']:.3f} mm, "
          f"RMS={summary['ir_rms_error_mm']:.3f} mm, p95={summary['ir_p95_error_mm']:.3f} mm")


if __name__ == "__main__":
    main()
