"""이 영상만으로 반경방향 왜곡(k1,k2)을 추정한다.

원리 (plumb-line / lattice self-calibration)
--------------------------------------------
체커보드 격자는 평면 위의 정규 격자이므로, 왜곡 없는 카메라라면 검출된 코너 배열이
이상 정수격자와 **호모그래피 하나로** 정확히 대응해야 한다. 왜곡이 있으면 잔차가 남는다.
따라서 여러 프레임·여러 화면위치에서 정렬격자를 모아 잔차 RMS를 최소화하는 (k1,k2)를 찾는다.
별도 캘리브레이션 촬영이 필요 없다.

주의: 좌·상 보드를 **각각 따로** 검출해야 한다. 합쳐서 돌리면 검출기가 두 시트를 가로지르는
가짜 격자를 만들고 잔차가 20~40 px(≈ 반 칸)로 뛴다.
"""
import json

import cv2
import numpy as np
from scipy.optimize import minimize

import common as C

SIZES = [(9, 6), (8, 6), (9, 5), (8, 5), (7, 5), (6, 5), (7, 4), (6, 4), (5, 4), (5, 3), (4, 3)]
STEP = 12
MAX_GRID_RMS = 5.0          # 인덱스 점프 오검출(20~40 px)만 배제. 왜곡 성분은 남긴다


def ideal(sz):
    w, h = sz
    return np.float32([[i, j] for j in range(h) for i in range(w)])


def lattice_rms(sz, corners, K=None, D=None):
    obs = corners.reshape(-1, 2).astype(np.float32)
    if D is not None and np.any(D):
        obs = cv2.undistortPoints(obs.reshape(-1, 1, 2), K, D, P=K).reshape(-1, 2)
    idl = ideal(sz)
    H, _ = cv2.findHomography(idl, obs, 0)
    if H is None:
        return 1e9
    pr = cv2.perspectiveTransform(idl.reshape(-1, 1, 2), H).reshape(-1, 2)
    return float(np.sqrt((np.linalg.norm(pr - obs, axis=1) ** 2).mean()))


def harvest():
    Hs = np.load(C.EXP / "config" / "bootstrap_H.npy")
    regions = C.load_regions()
    grids = []
    for fi in range(0, len(Hs), STEP):
        g = cv2.cvtColor(C.read_frame(fi), cv2.COLOR_BGR2GRAY)
        for name in ("LEFT", "TOP"):
            m = C.polygon_mask(regions[name], Hs[fi], g.shape, erode_px=6)
            if m.mean() < 8:
                continue
            gm = np.where(m > 0, g, 255).astype(np.uint8)
            for sz in SIZES:
                ok, c = cv2.findChessboardCornersSB(gm, sz, flags=cv2.CALIB_CB_ACCURACY)
                if not ok:
                    continue
                pts = c.reshape(-1, 2)
                ii = np.round(pts).astype(int)
                ii[:, 0] = np.clip(ii[:, 0], 0, g.shape[1] - 1)
                ii[:, 1] = np.clip(ii[:, 1], 0, g.shape[0] - 1)
                if not (m[ii[:, 1], ii[:, 0]] > 0).all():
                    continue
                if lattice_rms(sz, c) < MAX_GRID_RMS:
                    grids.append((fi, name, sz, c.reshape(-1, 2).astype(np.float32)))
                break
    return grids


def main():
    grids = harvest()
    if len(grids) < 15:
        raise SystemExit(f"정렬격자가 {len(grids)}개뿐이다 — 왜곡 추정 불가")
    pts = np.vstack([c for *_, c in grids])
    rr = np.hypot(pts[:, 0] - C.IMG_CENTER[0], pts[:, 1] - C.IMG_CENTER[1])
    print(f"정렬격자 {len(grids)}개, 코너 {len(pts)}개")
    print(f"  화면 커버리지 x {pts[:,0].min():.0f}..{pts[:,0].max():.0f}"
          f"  y {pts[:,1].min():.0f}..{pts[:,1].max():.0f}")
    print(f"  중심거리 p50={np.percentile(rr,50):.0f} p95={np.percentile(rr,95):.0f}"
          f" max={rr.max():.0f} px")

    K = C.camera_matrix()

    def cost(p):
        D = np.float64([p[0], p[1], 0.0, 0.0])
        e = [lattice_rms(sz, c, K, D) for _, _, sz, c in grids]
        return float(np.mean(e))

    base = cost([0.0, 0.0])
    res = minimize(cost, [0.0, 0.0], method="Nelder-Mead",
                   options=dict(xatol=1e-6, fatol=1e-9, maxiter=600))
    k1, k2 = float(res.x[0]), float(res.x[1])
    print(f"\n왜곡 없음      평균 격자 RMS = {base:.4f} px")
    print(f"k1={k1:+.5f} k2={k2:+.5f}  평균 격자 RMS = {res.fun:.4f} px"
          f"   ({100*(1-res.fun/base):.1f}% 감소)")

    C.DIST_JSON.write_text(json.dumps(
        dict(focal_px=C.FOCAL_PX, cx=C.IMG_CENTER[0], cy=C.IMG_CENTER[1],
             k1=k1, k2=k2, n_grids=len(grids), n_corners=int(len(pts)),
             rms_before_px=base, rms_after_px=float(res.fun)),
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {C.DIST_JSON}")


if __name__ == "__main__":
    main()
