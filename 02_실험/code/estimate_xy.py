"""피검 추정기 — 열화상 인쇄물 영역만으로 카메라 XY 이동을 추정한다.

방법은 00_프로젝트_개요/README.md 가 정의한 것 그대로다:
  KLT 추적 -> RANSAC/중앙값으로 공통 픽셀 이동량 (du,dv) -> mm 변환 -> 누적.

REF 로부터 받는 것은 두 가지뿐이고, 둘 다 '첫 프레임 기준점 검출로 초기화'에 해당한다
(README 필수조건 3). 이후 추정은 열린 루프다.
  - 마스크: PRINT 폴리곤을 격자 포즈로 전파 (체커보드·매트·나무판·자·흰여백 전부 배제)
  - 초기 스케일/방향: t=0 에서의 영상->격자 야코비안 J0

스케일 정책
  fixed  (주 결과) J0 를 끝까지 고정. 고정높이 실물 시스템과 같은 조건.
  oracle (참고)   매 프레임 REF 의 J_i 사용. fixed 와의 차이 = 스케일/방향 오차 기여분.
"""
import argparse
import csv

import cv2
import numpy as np

import common as C

# 정본은 common.py 로 옮겼다(라이브 도구 live_xy.py 와 공유하기 위해).
# 여기서는 이름을 그대로 유지해 이 모듈을 import 하는 degrade.py 등이 바뀌지 않게 한다.
MIN_TRACKED = C.MIN_TRACKED
MIN_ROI_FRAC = 0.05
ERODE_PX = 20
INLIER_PX = C.INLIER_PX
robust_translation = C.robust_translation


def jacobian_img_to_lattice(L, at=None):
    """영상 좌표 -> 격자 좌표 사상의 2x2 야코비안 (영상 중심에서). C.jacobian_img_to_lattice 참조."""
    return C.jacobian_img_to_lattice(L, C.IMG_CENTER if at is None else at)


def run(region="PRINT", scale="fixed", out_name=None):
    Ls = np.load(C.DATA / "ref_L.npy")
    regions = C.load_regions()
    info = C.video_info()
    n = info["n"]

    L0 = Ls[0]
    poly_lat = C.warp_pts(np.linalg.inv(L0), regions[region])      # f0 영상좌표 -> 격자좌표
    J0 = jacobian_img_to_lattice(L0)

    rows = []
    pos = np.zeros(2)
    prev_g = None
    prev_pts = None
    for i, f in C.frames():
        L = Ls[i]
        ok_ref = np.isfinite(L).all()
        g = C.prep_gray(f)
        mask = None
        vis = 0.0
        if ok_ref:
            poly_img = C.warp_pts(L, poly_lat)
            mask = np.zeros(g.shape, np.uint8)
            cv2.fillPoly(mask, [np.round(poly_img).astype(np.int32)], 255)
            k = 2 * ERODE_PX + 1
            mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
            vis = C.visible_fraction(poly_lat, L, g.shape)

        flags = []
        ntr, ratio = 0, 0.0
        if i > 0:
            if not ok_ref:
                flags.append("no_ref")
            elif vis < MIN_ROI_FRAC:
                flags.append("roi_out_of_frame")
            elif prev_pts is None or len(prev_pts) < MIN_TRACKED:
                flags.append("no_features")
            else:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(
                    prev_g, g, prev_pts.reshape(-1, 1, 2), None,
                    winSize=(21, 21), maxLevel=3)
                st = st.ravel().astype(bool)
                if st.sum() < MIN_TRACKED:
                    flags.append("track_lost")
                else:
                    duv, ratio, ntr = robust_translation(prev_pts[st], p1.reshape(-1, 2)[st])
                    J = J0 if scale == "fixed" else jacobian_img_to_lattice(L)
                    pos = pos - (J @ duv) * C.SQUARE_MM        # 정지 세계가 duv 만큼 움직임 = 카메라는 반대
        rows.append((i, i / info["fps"], pos[0], pos[1], ntr, ratio, vis, "|".join(flags)))

        prev_g = g
        prev_pts = None
        if mask is not None and mask.mean() > 1:
            q = cv2.goodFeaturesToTrack(g, 2000, 0.01, 8, mask=mask)
            if q is not None:
                prev_pts = q.reshape(-1, 2)
        if i % 200 == 0:
            print(f"  f{i:5d}  est=({pos[0]:+8.2f},{pos[1]:+8.2f}) mm  pts={ntr:4d} inl={ratio:.2f} roi={vis:.2f}")

    # 원점 = 첫 정지구간 평균 (REF 와 같은 규약)
    a, b = C.ORIGIN_SEGMENT
    org = np.mean([[r[2], r[3]] for r in rows[a:b + 1]], axis=0)
    C.DATA.mkdir(parents=True, exist_ok=True)
    out = C.DATA / (out_name or f"est_{region.lower()}_{scale}.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "t_s", "est_X_mm", "est_Y_mm", "tracked_points",
                    "inlier_ratio", "roi_visible", "flags"])
        for i, t, x, y, ntr, ratio, vis, fl in rows:
            w.writerow([i, f"{t:.4f}", f"{x-org[0]:.4f}", f"{y-org[1]:.4f}",
                        ntr, f"{ratio:.4f}", f"{vis:.4f}", fl])
    print(f"저장: {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="PRINT", choices=["PRINT", "LEFT", "TOP"],
                    help="LEFT 는 새니티 체크용 (REF 와 거의 일치해야 한다)")
    ap.add_argument("--scale", default="fixed", choices=["fixed", "oracle"])
    ap.add_argument("--out")
    a = ap.parse_args()
    run(a.region, a.scale, a.out)
