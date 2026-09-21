"""REF 궤적 — 체커보드를 '이상 격자'로 모델링해 매 프레임 절대 포즈를 푼다.

왜 체인이 아니라 격자인가
--------------------------
프레임 간 변환을 누적하면 프레임당 잔차 ~0.3 px 가 1000 프레임에 걸쳐 랜덤워크로 쌓여
왕복 후 14~20 px(≈3~4 mm) 드리프트가 남는다(실측). 반면 보드는 피치 10 mm 정규 격자이므로
격자 좌표 (i,j) -> 영상 좌표의 호모그래피 L_i 를 **매 프레임 새로 적합**하면 누적이 원천적으로 없다.
체인은 격자 인덱스 배정을 예측하는 데만 쓰이고(반 칸 = 25 px 이내면 충분, 실제 프레임간 최대 19 px),
위치는 절대 측정이 된다.

world 좌표계 = LEFT 시트 격자 (단위 1 = 한 칸 = 10 mm). 전 구간 가시라 기준으로 적합하다.
TOP 시트는 LEFT 격자에 한 번만 앵커해 인덱스 규약을 잡은 뒤, 이후 매 프레임 **독립으로** 적합한다.
따라서 두 시트가 주는 카메라 위치는 고정 강체변환을 제외하면 서로 독립 측정이고,
그 차이가 REF 오차의 경험적 상한이 된다.
"""
import cv2
import numpy as np

import common as C

SUBPIX_WIN = (7, 7)
SUBPIX_CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01)
ASSIGN_TOL = 0.28          # 격자 단위. 정수 노드에서 이만큼 이내여야 채택
MIN_CORNERS = 20
MIN_MASK_FRAC = 0.03
INIT_SIZES = [(9, 6), (8, 6), (9, 5), (8, 5), (7, 5), (6, 5), (7, 4), (6, 4), (5, 4)]


def detect_corners(gray, mask):
    p = cv2.goodFeaturesToTrack(gray, 3000, 0.01, 18, mask=mask, blockSize=5)
    if p is None or len(p) < MIN_CORNERS:
        return None
    return cv2.cornerSubPix(gray, p.astype(np.float32), SUBPIX_WIN,
                            (-1, -1), SUBPIX_CRIT).reshape(-1, 2)


def refit(L_pred, corners, tol=ASSIGN_TOL, iters=1):
    """예측 L_pred 로 코너를 정수격자에 배정하고 L 을 적합한다.

    iters=1(기본)이 단일 패스다. 배정<->적합을 반복 수렴시켜 예측 의존성을 없애는 것도
    시도했으나(iters=4) 모든 궤적 지표가 악화됐다: V2 0.23 -> 1.06 mm, V5 5.5 -> 12.7 mm,
    프레임간 최대 이동 4.87 -> 5.36 mm(반 칸 초과). 격자가 주기적이라 '가장 잘 맞는 배정'이
    유일하지 않고, 반복하면 통째로 한 칸 미끄러지기 때문이다. 적합 잔차는 좋아지지만
    배정은 틀린다. 예측에 앵커하는 단일 패스가 옳다.
    """
    if corners is None or len(corners) < MIN_CORNERS:
        return None, 0, np.nan
    L = L_pred
    best = (None, 0, np.nan)
    prev_keep = None
    for _ in range(iters):
        uv = C.warp_pts(np.linalg.inv(L), corners)
        node = np.round(uv)
        keep = (np.abs(uv - node) < tol).all(axis=1)
        if keep.sum() < MIN_CORNERS:
            return best
        src = node[keep].astype(np.float32)
        dst = corners[keep].astype(np.float32)
        Ln, inl = cv2.findHomography(src, dst, cv2.RANSAC, 1.5, maxIters=3000,
                                     confidence=0.999)
        if Ln is None:
            return best
        inl = inl.ravel().astype(bool)
        if inl.sum() < MIN_CORNERS:
            return best
        pr = C.warp_pts(Ln, src[inl])
        rms = float(np.sqrt((np.linalg.norm(pr - dst[inl], axis=1) ** 2).mean()))
        L, best = Ln, (Ln, int(inl.sum()), rms)
        if prev_keep is not None and (keep == prev_keep).all():
            break
        prev_keep = keep
    return best


def _basis_ok(L, pitch_nn):
    """격자 기저벡터가 실제 최근접이웃 피치와 맞는지 (인덱스 점프 검출). -> 보정행렬 또는 None"""
    o = C.warp_pts(L, [[0, 0]])[0]
    fix = np.eye(3)
    for ax, e in ((0, np.linalg.norm(C.warp_pts(L, [[1, 0]])[0] - o)),
                  (1, np.linalg.norm(C.warp_pts(L, [[0, 1]])[0] - o))):
        k = e / pitch_nn
        kr = int(round(k))
        if kr < 1 or abs(k - kr) > 0.2:
            return None
        fix[ax, ax] = 1.0 / kr
    return fix


def init_lattice_ordered(gray, mask):
    """정렬 부분격자(findChessboardCornersSB)로 L을 초기화. 실패 시 None."""
    corners = detect_corners(gray, mask)
    if corners is None:
        return None
    pitch = C.lattice_pitch(corners)
    gm = np.where(mask > 0, gray, 255).astype(np.uint8)
    for sz in INIT_SIZES:
        ok, c = cv2.findChessboardCornersSB(gm, sz, flags=cv2.CALIB_CB_ACCURACY)
        if not ok:
            continue
        w, h = sz
        obs = c.reshape(-1, 2).astype(np.float32)
        ii = np.clip(np.round(obs).astype(int), [0, 0],
                     [gray.shape[1] - 1, gray.shape[0] - 1])
        if not (mask[ii[:, 1], ii[:, 0]] > 0).all():
            continue
        idl = np.float32([[i, j] for j in range(h) for i in range(w)])
        L, _ = cv2.findHomography(idl, obs, 0)
        if L is None:
            continue
        pr = C.warp_pts(L, idl)
        if np.sqrt((np.linalg.norm(pr - obs, axis=1) ** 2).mean()) > 1.0:
            continue
        fix = _basis_ok(L, pitch)
        if fix is None:
            continue
        return refit(L @ np.linalg.inv(fix), corners)[0]
    return None


def init_lattice_anchored(corners, L_ref):
    """다른 시트의 격자 L_ref 에 앵커해 초기화한다.

    두 시트 모두 10 mm 정규 격자이므로, 이 시트의 코너를 L_ref 격자좌표로 보내면
    '단위 간격 + 미지 회전 theta + 미지 오프셋'의 격자가 된다. theta 를 1차원 탐색해
    정수격자에 가장 잘 맞는 각도를 찾는다.
    """
    P = C.warp_pts(np.linalg.inv(L_ref), corners)
    best = None
    for th in np.deg2rad(np.arange(0.0, 90.0, 0.05)):
        R = np.float64([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        Q = P @ R.T
        # 최적 오프셋 = 소수부의 원형평균
        off = np.array([np.angle(np.exp(2j * np.pi * Q[:, k]).mean()) / (2 * np.pi)
                        for k in range(2)])
        r = Q - off
        e = float(np.mean(np.min([np.abs(r - np.round(r))], axis=0) ** 2))
        if best is None or e < best[0]:
            best = (e, th, off)
    _, th, off = best
    R = np.float64([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    # A: 이 시트 격자 -> L_ref 격자
    A = np.eye(3)
    A[:2, :2] = R.T
    A[:2, 2] = (R.T @ off)
    return L_ref @ A


def track(regions, Hs, verbose=True):
    """전 프레임에 대해 LEFT/TOP 격자 포즈를 구한다."""
    n = len(Hs)
    res = {"LEFT": {}, "TOP": {}}
    prev = {"LEFT": None, "TOP": None}
    for i, f in C.frames():
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        for name in ("LEFT", "TOP"):
            mask = C.polygon_mask(regions[name], Hs[i], g.shape, erode_px=25)
            if mask.mean() / 255 < MIN_MASK_FRAC:
                prev[name] = None
                continue
            corners = detect_corners(g, mask)
            if corners is None:
                prev[name] = None
                continue
            L = None
            if prev[name] is not None:
                M = Hs[i] @ np.linalg.inv(Hs[i - prev[name][1]])
                L, k, rms = refit(M @ prev[name][0], corners)
            if L is None:
                seed = init_lattice_ordered(g, mask)
                if seed is None and name == "TOP" and i in res["LEFT"]:
                    seed = init_lattice_anchored(corners, res["LEFT"][i][0])
                if seed is None:
                    prev[name] = None
                    continue
                L, k, rms = refit(seed, corners)
                if L is None:
                    prev[name] = None
                    continue
            res[name][i] = (L, k, rms)
            prev[name] = (L, 1)
        if verbose and i % 200 == 0:
            s = " ".join(f"{nm}:{'ok' if i in res[nm] else '--'}" for nm in ("LEFT", "TOP"))
            print(f"  f{i:5d}  {s}")
    return res


def cam_lattice(L):
    """영상 중심 아래의 평면점을 격자좌표로. 단위 = 칸 = 10 mm."""
    return C.warp_pts(np.linalg.inv(L), C.IMG_CENTER[None, :])[0]


# --- 드라이버 ---------------------------------------------------------------
def _fixed_transform(pL, pT):
    """두 시트 격자좌표계를 잇는 고정 강체변환(단위 스케일)을 추정한다."""
    A, _ = cv2.estimateAffinePartial2D(pT.astype(np.float32).reshape(-1, 1, 2),
                                       pL.astype(np.float32).reshape(-1, 1, 2),
                                       method=cv2.LMEDS)
    return C.as_h(A)


def main():
    import csv
    Hs = np.load(C.EXP / "config" / "bootstrap_H.npy")
    regions = C.load_regions()
    info = C.video_info()
    print(f"영상 {info['n']} 프레임 @ {info['fps']:.2f} fps — 격자 앵커 REF 추적")
    res = track(regions, Hs)
    nL, nT = len(res["LEFT"]), len(res["TOP"])
    print(f"\nLEFT 성공 {nL}/{info['n']} 프레임, TOP 성공 {nT}/{info['n']} 프레임")
    if nL < info["n"] * 0.98:
        miss = [i for i in range(info["n"]) if i not in res["LEFT"]]
        print(f"  ! LEFT 실패 프레임 {len(miss)}개: {miss[:20]}")

    # 격자좌표(칸) -> mm
    posL = {i: cam_lattice(L) * C.SQUARE_MM for i, (L, _, _) in res["LEFT"].items()}
    posT = {i: cam_lattice(L) * C.SQUARE_MM for i, (L, _, _) in res["TOP"].items()}

    both = sorted(set(posL) & set(posT))
    xcheck = {}
    if len(both) > 50:
        S = _fixed_transform(np.array([posL[i] for i in both]),
                             np.array([posT[i] for i in both]))
        for i in both:
            q = C.warp_pts(S, [posT[i]])[0]
            xcheck[i] = float(np.linalg.norm(q - posL[i]))
        e = np.array([xcheck[i] for i in both])
        print(f"\n교차검증 (LEFT vs TOP, 독립 측정 {len(both)} 프레임):")
        print(f"  |차이| med={np.median(e):.3f} mm  p95={np.percentile(e,95):.3f} mm  max={e.max():.3f} mm")

    # 원점 = 첫 정지구간 평균
    a, b = C.ORIGIN_SEGMENT
    org = np.mean([posL[i] for i in range(a, b + 1) if i in posL], axis=0)
    print(f"\n원점(첫 정지구간 f{a}-{b} 평균) = 격자 mm ({org[0]:.2f}, {org[1]:.2f})")

    C.DATA.mkdir(parents=True, exist_ok=True)
    path = C.DATA / "ref.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "t_s", "ref_X_mm", "ref_Y_mm", "ref_corners", "ref_rms_px",
                    "mm_per_px", "top_corners", "top_rms_px", "xcheck_mm"])
        for i in range(info["n"]):
            if i not in posL:
                w.writerow([i, f"{i/info['fps']:.4f}", "", "", 0, "", "", "", "", ""])
                continue
            L, k, rms = res["LEFT"][i]
            p = posL[i] - org
            o = C.warp_pts(L, [[0, 0]])[0]
            mmpp = C.SQUARE_MM / np.linalg.norm(C.warp_pts(L, [[1, 0]])[0] - o)
            tk, trms = (res["TOP"][i][1], res["TOP"][i][2]) if i in res["TOP"] else ("", "")
            w.writerow([i, f"{i/info['fps']:.4f}", f"{p[0]:.4f}", f"{p[1]:.4f}", k,
                        f"{rms:.4f}", f"{mmpp:.6f}", tk,
                        f"{trms:.4f}" if trms != "" else "",
                        f"{xcheck[i]:.4f}" if i in xcheck else ""])
    np.save(C.DATA / "ref_L.npy",
            np.array([res["LEFT"][i][0] if i in res["LEFT"] else np.full((3, 3), np.nan)
                      for i in range(info["n"])]))
    np.save(C.DATA / "ref_origin.npy", org)
    print(f"저장: {path}")


if __name__ == "__main__":
    main()
