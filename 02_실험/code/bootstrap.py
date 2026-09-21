"""부트스트랩: 전체 프레임 강체 누적 변환 H_i (world=f0 -> frame i).

장면 전체가 강체·평면이므로 마스크 없이도 인라이어 900+ 가 나온다.
이 결과는 (1) f0 폴리곤을 전 프레임에 전파하고 (2) REF의 초기값을 주는 데만 쓴다.
REF 자체는 ref_track.py 가 좌·상 보드 코너만으로 다시 계산한다.
"""
import cv2
import numpy as np

import common as C

OUT_NPY = C.EXP / "config" / "bootstrap_H.npy"


def run():
    info = C.video_info()
    n = info["n"]
    Hs = np.zeros((n, 3, 3), np.float64)
    Hs[0] = np.eye(3)
    H = np.eye(3)
    prev = None
    stats = []
    for i, f in C.frames():
        g = C.prep_gray(f)
        if prev is not None:
            pts = cv2.goodFeaturesToTrack(prev, 2500, 0.008, 10)
            M, ratio, ntr = C.track_similarity(prev, g, pts)
            if M is None:
                M = np.eye(3)
                ratio, ntr = 0.0, 0
            H = M @ H
            stats.append((i, ratio, ntr))
        Hs[i] = H
        prev = g
        if i % 200 == 0:
            s, th, tx, ty = C.decompose_sim(H)
            cx, cy = C.cam_world(H)
            print(f"  f{i:5d}  cam_world=({cx:8.1f},{cy:8.1f})px  s={s:.4f} th={th:+.2f}deg")
    np.save(OUT_NPY, Hs)
    st = np.array(stats)
    print(f"\n저장: {OUT_NPY}")
    print(f"인라이어비 min/med = {st[:,1].min():.3f} / {np.median(st[:,1]):.3f}"
          f"   추적점 min/med = {int(st[:,2].min())} / {int(np.median(st[:,2]))}")


if __name__ == "__main__":
    run()
