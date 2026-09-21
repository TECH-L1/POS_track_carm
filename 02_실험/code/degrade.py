"""해상도·노이즈 열화 스윕 — 모듈 선정 트랙(①)과 연결되는 부분.

시뮬레이션 규약
---------------
- 해상도는 **세로 픽셀 수**로 표기하고 FOV 는 고정한다(같은 장면을 더 적은 픽셀로 본다).
  후보 모듈의 세로 해상도에 대응: 640x480 -> 480, 320x240 -> 240, 256x192 -> 192, 160x120 -> 120.
  가로는 원본 16:9 를 유지하므로 실제 모듈의 4:3 과는 화각비가 다르다.
- 노이즈는 8bit 그레이 레벨의 가우시안 σ 로 준다. 이 영상은 복사(radiometric) 데이터가 아니라
  인쇄된 열화상 사진이므로 σ 를 NETD 로 직접 환산할 수 없다. 장면의 온도폭을 ΔT 라 하면
      NETD ≈ σ / 255 × ΔT
  이므로, ΔT 를 아는 실촬영에서 이 표를 다시 읽으면 된다.
- 순서: 다운샘플(area) -> 노이즈 주입 -> CLAHE -> KLT. 실제 센서에서 잡음이 들어가는 순서와 같다.
"""
import csv

import cv2
import numpy as np

import common as C
from common import jacobian_img_to_lattice, robust_translation
from estimate_xy import ERODE_PX, MIN_ROI_FRAC, MIN_TRACKED

HEIGHTS = [1080, 480, 240, 192, 120]
LABELS = {1080: "1920x1080 (원본)", 480: "~854x480", 240: "~427x240 (320x240급)",
          192: "~341x192 (256x192급)", 120: "~213x120 (160x120급)"}
SIGMAS = [0.0, 1.0, 2.0, 4.0, 8.0]


def load_stack(h):
    """세로 h 로 다운샘플한 그레이 프레임과 ROI 마스크를 한 번에 만든다."""
    Ls = np.load(C.DATA / "ref_L.npy")
    poly_lat = C.warp_pts(np.linalg.inv(Ls[0]), C.load_regions()["PRINT"])
    s = h / 1080.0
    w = int(round(1920 * s))
    er = max(1, int(round(ERODE_PX * s)))
    frames, masks, vis, Js = [], [], [], []
    for i, f in C.frames():
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA)
        L = Ls[i]
        if not np.isfinite(L).all():
            frames.append(g); masks.append(None); vis.append(0.0); Js.append(None)
            continue
        p = C.warp_pts(L, poly_lat) * s
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [np.round(p).astype(np.int32)], 255)
        m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * er + 1,) * 2))
        frames.append(g); masks.append(m)
        vis.append(C.visible_fraction(poly_lat, L, (1080, 1920)))
        Js.append(jacobian_img_to_lattice(L, C.IMG_CENTER))
    return frames, masks, vis, Js, s


def one_run(frames, masks, vis, Js, s, sigma, rng):
    """한 (해상도, σ) 조합. -> 프레임별 (x,y,n,inl,flag)"""
    J0 = Js[0]
    pos = np.zeros(2)
    prev_g, prev_pts = None, None
    out = []
    for i in range(len(frames)):
        g = frames[i]
        if sigma > 0:
            g = np.clip(g.astype(np.float32) + rng.normal(0, sigma, g.shape), 0, 255).astype(np.uint8)
        g = C.prep_gray(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
        flag = ""
        n, inl = 0, 0.0
        if i > 0:
            if masks[i] is None:
                flag = "no_ref"
            elif vis[i] < MIN_ROI_FRAC:
                flag = "roi_out_of_frame"
            elif prev_pts is None or len(prev_pts) < MIN_TRACKED:
                flag = "no_features"
            else:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_g, g, prev_pts.reshape(-1, 1, 2),
                                                     None, winSize=(21, 21), maxLevel=3)
                st = st.ravel().astype(bool)
                if st.sum() < MIN_TRACKED:
                    flag = "track_lost"
                else:
                    duv, inl, n = robust_translation(prev_pts[st], p1.reshape(-1, 2)[st])
                    pos = pos - (J0 @ (duv / s)) * C.SQUARE_MM
        out.append((pos[0], pos[1], n, inl, flag))
        prev_g = g
        prev_pts = None
        if masks[i] is not None and masks[i].mean() > 1:
            q = cv2.goodFeaturesToTrack(g, 2000, 0.01, max(3, int(round(8 * s))), mask=masks[i])
            if q is not None:
                prev_pts = q.reshape(-1, 2)
    return out


def main():
    from validate_ref import load_ref
    ref = load_ref()
    idx = sorted(ref)
    R = np.array([[ref[i]["x"], ref[i]["y"]] for i in idx])
    trav = float(np.linalg.norm(np.diff(R, axis=0), axis=1).sum())
    a, b = C.ORIGIN_SEGMENT
    still = [k for k, i in enumerate(idx) if a <= i <= b]

    rows = []
    for h in HEIGHTS:
        print(f"\n[{LABELS[h]}] 프레임 캐시 생성 중...")
        frames, masks, vis, Js, s = load_stack(h)
        for sg in SIGMAS:
            rng = np.random.default_rng(12345)
            res = one_run(frames, masks, vis, Js, s, sg, rng)
            E = np.array([[res[i][0], res[i][1]] for i in idx])
            E = E - E[still].mean(axis=0)
            err = np.linalg.norm(E - R, axis=1)
            drift = float(np.linalg.norm(E[still[-1]] - E[still[0]]))
            # 증분 오차: 누적에 오염되지 않아 해상도/노이즈 의존성을 직접 보여준다
            inc = np.linalg.norm(np.diff(E, axis=0) - np.diff(R, axis=0), axis=1)
            inc_rms_um = 1000.0 * float(np.sqrt((inc ** 2).mean()))
            npts = np.array([res[i][2] for i in idx])
            lost = sum(1 for i in idx if res[i][4])
            rows.append(dict(height=h, label=LABELS[h], sigma=sg,
                             final_mm=float(err[-1]), rms_mm=float(np.sqrt((err ** 2).mean())),
                             max_mm=float(err.max()), pct_travel=100 * float(err[-1]) / trav,
                             still_drift_mm=drift, inc_rms_um=inc_rms_um,
                             med_points=int(np.median(npts)), lost_frames=lost))
            r = rows[-1]
            print(f"   σ={sg:4.1f}  최종 {r['final_mm']:7.2f} mm ({r['pct_travel']:5.2f}%)  "
                  f"RMS {r['rms_mm']:6.2f}  최대 {r['max_mm']:6.2f}  "
                  f"정지드리프트 {r['still_drift_mm']:6.3f}  증분RMS {r['inc_rms_um']:6.1f}um  "
                  f"점수 {r['med_points']:5d}  실패 {r['lost_frames']}")
        del frames, masks

    C.DATA.mkdir(parents=True, exist_ok=True)
    out = C.DATA / "sweep.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n총 이동거리 {trav:.0f} mm 기준. 저장: {out}")


if __name__ == "__main__":
    main()
