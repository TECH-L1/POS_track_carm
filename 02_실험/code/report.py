"""산출물 생성 — trajectory.csv, trajectory.png, sweep_heatmap.png, overlay.mp4."""
import csv

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import common as C
from validate_ref import load_ref

FPS_OUT = 30
VID_W, VID_H = 960, 540
# 우하단 REF/EST 궤적 패널. 원본 960×540 검증 영상의 가시성을 우선해 축소한다.
PW = PH = 200


def load_est(name):
    with open(C.DATA / name, encoding="utf-8") as fh:
        return {int(r["frame"]): r for r in csv.DictReader(fh)}


def merged_csv(ref, fixed, oracle):
    C.DATA.mkdir(parents=True, exist_ok=True)
    path = C.DATA / "trajectory.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "t_s", "ref_X_mm", "ref_Y_mm", "est_X_mm", "est_Y_mm",
                    "err_X_mm", "err_Y_mm", "err_norm_mm",
                    "est_oracle_X_mm", "est_oracle_Y_mm", "err_oracle_mm",
                    "tracked_points", "inlier_ratio", "roi_visible",
                    "ref_corners", "ref_rms_px", "mm_per_px", "xcheck_mm", "flags"])
        for i in sorted(fixed):
            e = fixed[i]
            o = oracle.get(i)
            if i not in ref:
                w.writerow([i, e["t_s"]] + [""] * 10 +
                           [e["tracked_points"], e["inlier_ratio"], e["roi_visible"],
                            "", "", "", "", e["flags"]])
                continue
            r = ref[i]
            ex = float(e["est_X_mm"]) - r["x"]
            ey = float(e["est_Y_mm"]) - r["y"]
            eo = np.hypot(float(o["est_X_mm"]) - r["x"], float(o["est_Y_mm"]) - r["y"])
            w.writerow([i, e["t_s"], f"{r['x']:.4f}", f"{r['y']:.4f}",
                        e["est_X_mm"], e["est_Y_mm"], f"{ex:.4f}", f"{ey:.4f}",
                        f"{np.hypot(ex, ey):.4f}",
                        o["est_X_mm"], o["est_Y_mm"], f"{eo:.4f}",
                        e["tracked_points"], e["inlier_ratio"], e["roi_visible"],
                        r["n"], f"{r['rms']:.4f}", f"{r['mmpp']:.6f}",
                        "" if np.isnan(r["xchk"]) else f"{r['xchk']:.4f}", e["flags"]])
    return path


def trajectory_plot(ref, fixed, oracle):
    idx = sorted(ref)
    t = np.array([ref[i]["t"] for i in idx])
    R = np.array([[ref[i]["x"], ref[i]["y"]] for i in idx])
    E = np.array([[float(fixed[i]["est_X_mm"]), float(fixed[i]["est_Y_mm"])] for i in idx])
    O = np.array([[float(oracle[i]["est_X_mm"]), float(oracle[i]["est_Y_mm"])] for i in idx])
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for k, lab in ((0, "X"), (1, "Y")):
        ax[k].plot(t, R[:, k], "g-", lw=2, label="REF (checkerboard lattice)")
        ax[k].plot(t, E[:, k], "b--", lw=1.2, label="EST fixed scale")
        ax[k].plot(t, O[:, k], "r:", lw=1.4, label="EST oracle scale")
        ax[k].set_ylabel(lab + " [mm]")
        ax[k].grid(alpha=.3)
    ax[0].legend(fontsize=8)
    ax[2].plot(t, np.linalg.norm(E - R, axis=1), "b-", lw=1.2, label="fixed")
    ax[2].plot(t, np.linalg.norm(O - R, axis=1), "r-", lw=1.2, label="oracle")
    ax[2].set_ylabel("error [mm]")
    ax[2].set_xlabel("t [s]")
    ax[2].grid(alpha=.3)
    ax[2].legend(fontsize=8)
    for a in ax:
        for s0, s1 in C.STILL_SEGMENTS:
            a.axvspan(s0 / 59.4, s1 / 59.4, color="k", alpha=.06)
    fig.tight_layout()
    C.VISUALS.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.VISUALS / "trajectory.png", dpi=140)
    plt.close(fig)
    return C.VISUALS / "trajectory.png"


def heatmap():
    with open(C.DATA / "sweep.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    hs = sorted({int(r["height"]) for r in rows}, reverse=True)
    sg = sorted({float(r["sigma"]) for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    specs = ((axes[0], "pct_travel", "cumulative error / travel", "%", "{:.2f}"),
             (axes[1], "still_drift_mm", "drift over 7.6 s at rest", "mm", "{:.3f}"))
    for ax, key, title, unit, fmt in specs:
        M = np.array([[float(next(r[key] for r in rows if int(r["height"]) == h
                                  and float(r["sigma"]) == s)) for s in sg] for h in hs])
        im = ax.imshow(M, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(sg)), [f"{s:.0f}" for s in sg])
        ax.set_yticks(range(len(hs)), [f"{h}p" for h in hs])
        ax.set_xlabel("noise sigma (8-bit gray level)")
        ax.set_ylabel("vertical resolution (FOV fixed)")
        ax.set_title(title)
        for a in range(len(hs)):
            for b in range(len(sg)):
                ax.text(b, a, fmt.format(M[a, b]), ha="center", va="center", fontsize=8,
                        color="w" if M[a, b] < M.mean() else "k")
        fig.colorbar(im, ax=ax, label=unit)
    fig.tight_layout()
    C.VISUALS.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.VISUALS / "sweep_heatmap.png", dpi=140)
    plt.close(fig)
    return C.VISUALS / "sweep_heatmap.png"


def overlay_video(ref, fixed, mode="both"):
    """검증 영상을 만든다.

    mode='ref' 는 체커보드 REF만, mode='est' 는 중앙 IR KLT만 보여준다.
    분리 영상에는 궤적 패널을 넣지 않는다. 궤적·RMSE 비교는 Rerun에서 수행한다.
    """
    if mode not in ("both", "ref", "est"):
        raise ValueError(f"unknown overlay mode: {mode}")
    Ls = np.load(C.DATA / "ref_L.npy")
    # regions_f0.json의 폴리곤은 첫 영상 프레임 좌표계다. 격자포즈 L이 아니라
    # f0->현재 프레임 부트스트랩 변환 H로 보드 표시 영역을 전파해야 한다.
    Hs = np.load(C.EXP / "config" / "bootstrap_H.npy")
    regions = C.load_regions()
    poly_lat = C.warp_pts(np.linalg.inv(Ls[0]), regions["PRINT"])
    idx = sorted(ref)
    R = np.array([[ref[i]["x"], ref[i]["y"]] for i in idx])
    E = np.array([[float(fixed[i]["est_X_mm"]), float(fixed[i]["est_Y_mm"])] for i in idx])
    at = {i: k for k, i in enumerate(idx)}
    lo = np.minimum(R.min(0), E.min(0)) - 12
    hi = np.maximum(R.max(0), E.max(0)) + 12

    def xy(p):
        return (20 + int(PW * (p[0] - lo[0]) / (hi[0] - lo[0])),
                20 + int(PH * (p[1] - lo[1]) / (hi[1] - lo[1])))

    name = {"both": "overlay.mp4", "ref": "overlay_ref.mp4", "est": "overlay_est.mp4"}[mode]
    C.VISUALS.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(C.VISUALS / name),
                          cv2.VideoWriter_fourcc(*"mp4v"), FPS_OUT, (VID_W, VID_H))
    sc = VID_W / 1920.0
    prev_ir_g, prev_ir_pts = None, None
    for i, f in C.frames():
        vis_src = f.copy()
        L = Ls[i]
        ref_mask = np.zeros(f.shape[:2], np.uint8)
        ir_mask = None
        if np.isfinite(L).all():
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            for nm, col in (("LEFT", (0, 255, 255)), ("TOP", (255, 160, 0))):
                m = C.polygon_mask(regions[nm], Hs[i], g.shape, erode_px=25)
                ref_mask = cv2.bitwise_or(ref_mask, m)
                if m.mean() < 3:
                    continue
                if mode in ("both", "ref"):
                    q = cv2.goodFeaturesToTrack(g, 400, 0.01, 18, mask=m)
                    if q is not None:
                        for pt in np.round(q.reshape(-1, 2)).astype(int):
                            cv2.circle(vis_src, tuple(pt), 4, col, -1)
            p = C.warp_pts(L, poly_lat) * sc
            # estimate_xy.py 와 동일한 중앙 IR 마스크(20 px 침식).
            ir_poly = C.warp_pts(L, poly_lat)
            ir_mask = np.zeros(g.shape, np.uint8)
            cv2.fillPoly(ir_mask, [np.round(ir_poly).astype(np.int32)], 255)
            ir_mask = cv2.erode(ir_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
            if mode in ("both", "est"):
                cv2.polylines(vis_src, [np.round(p / sc).astype(np.int32)], True, (0, 0, 255), 4)

        if mode == "ref":
            # REF는 체커보드만 남겨 실제 관측점을 명확히 한다.
            vis_src = cv2.bitwise_and(vis_src, vis_src, mask=ref_mask)
        elif mode == "est":
            # EST의 입력에는 체커보드/주변 텍스처가 없음을 영상에서 명확히 보인다.
            vis_src = cv2.bitwise_and(vis_src, vis_src, mask=ir_mask)
            g_ir = C.prep_gray(f)
            if prev_ir_pts is not None and len(prev_ir_pts) >= 30:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(
                    prev_ir_g, g_ir, prev_ir_pts.reshape(-1, 1, 2), None,
                    winSize=(21, 21), maxLevel=3)
                if p1 is not None:
                    for a, b in zip(prev_ir_pts[st.ravel().astype(bool)], p1.reshape(-1, 2)[st.ravel().astype(bool)]):
                        cv2.line(vis_src, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)),
                                 (255, 220, 0), 2)
                        cv2.circle(vis_src, tuple(np.round(b).astype(int)), 3, (255, 120, 0), -1)
            # 추정기는 2,000점을 사용한다. 영상은 가독성과 렌더링 시간을 위해 그중
            # 대표 500개 KLT 트랙만 보이며, 저장된 EST 궤적 자체는 바꾸지 않는다.
            q = cv2.goodFeaturesToTrack(g_ir, 500, 0.01, 8, mask=ir_mask)
            prev_ir_pts = None if q is None else q.reshape(-1, 2)
            prev_ir_g = g_ir

        vis = cv2.resize(vis_src, (VID_W, VID_H))

        k = at.get(i)
        if mode == "both":
            panel = np.full((PH + 40, PW + 40, 3), 32, np.uint8)
            if k is not None and k > 1:
                for arr, col in ((R[:k + 1], (0, 255, 0)), (E[:k + 1], (60, 120, 255))):
                    cv2.polylines(panel, [np.array([xy(p) for p in arr], np.int32)], False, col, 2)
                cv2.circle(panel, xy(R[k]), 5, (0, 255, 0), -1)
                cv2.circle(panel, xy(E[k]), 5, (60, 120, 255), -1)
            cv2.putText(panel, "REF", (22, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
            cv2.putText(panel, "EST", (72, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (60, 120, 255), 1)
            vis[VID_H - PH - 40:, VID_W - PW - 40:] = panel

        if mode == "ref":
            txt = f"f{i}  REF only - checkerboard corners / lattice fit"
        elif mode == "est":
            txt = f"f{i}  EST only - central IR KLT display (max 500), est pts={fixed[i]['tracked_points']}"
        elif k is not None:
            txt = (f"f{i}  t={i/59.4:.2f}s  REF({R[k,0]:+.1f},{R[k,1]:+.1f})  "
                   f"EST({E[k,0]:+.1f},{E[k,1]:+.1f})  "
                   f"err={np.linalg.norm(E[k]-R[k]):.2f}mm  pts={fixed[i]['tracked_points']}")
        else:
            txt = f"f{i}"
        for col, th in (((0, 0, 0), 4), ((255, 255, 255), 1)):
            cv2.putText(vis, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .5, col, th)
        out.write(vis)
    out.release()
    return C.VISUALS / name


if __name__ == "__main__":
    ref = load_ref()
    fixed = load_est("est_print_fixed.csv")
    oracle = load_est("est_print_oracle.csv")
    print("trajectory.csv    ->", merged_csv(ref, fixed, oracle))
    print("trajectory.png    ->", trajectory_plot(ref, fixed, oracle))
    print("sweep_heatmap.png ->", heatmap())
    print("overlay.mp4 생성 중 (1206 프레임)...")
    print("overlay.mp4       ->", overlay_video(ref, fixed))
    print("overlay_ref.mp4   ->", overlay_video(ref, fixed, "ref"))
    print("overlay_est.mp4   ->", overlay_video(ref, fixed, "est"))
