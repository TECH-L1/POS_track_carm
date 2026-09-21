"""REF 자체 검증 (V1~V6).

REF 는 '진값'이 아니라 또 하나의 추정이다. 추정기와 비교하기 전에 REF 자신의 노이즈 바닥이
비교 대상 오차보다 충분히 작다는 걸 먼저 보여야 한다. V1/V2 가 1차 관문이다.

측정 설계상 주의 두 가지
------------------------
1. 첫 정지구간(f2-456, 카메라를 놓아둔 상태)만이 진짜 무이동 구간이다. 이후의 "정지"는
   손에 든 채의 멈춤이라 1.3~2.3 mm 매끄럽게 미끄러진다. 이것은 LEFT/TOP 두 독립 보드가
   0.12 mm 이내로 일치하게 관측하므로 REF 오차가 아니라 실제 움직임이다.
   따라서 정지구간 평탄도로 REF 잡음을 재면 안 된다.
2. REF 잡음의 올바른 척도는 **두 시트의 독립 측정 차이**(V3)다. 실제 움직임은 두 측정에
   공통이라 상쇄되고 측정 잡음만 남는다.
"""
import csv

import numpy as np

import common as C

PASS = {"V1": 0.5, "V2": 2.0, "V3": 0.3, "V4": 3.0, "V5": 0.5, "V6": 5.0}
FAST_MM = 2.5      # 프레임당 이 이상이면 고속 = 모션블러 구간
GATE = ("V1", "V2", "V3")


def load_ref():
    rows = {}
    with open(C.DATA / "ref.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["ref_X_mm"] == "":
                continue
            rows[int(r["frame"])] = dict(
                t=float(r["t_s"]), x=float(r["ref_X_mm"]), y=float(r["ref_Y_mm"]),
                n=int(r["ref_corners"]), rms=float(r["ref_rms_px"]),
                mmpp=float(r["mm_per_px"]),
                xchk=float(r["xcheck_mm"]) if r["xcheck_mm"] else np.nan)
    return rows


def seg_xy(ref, a, b):
    idx = [i for i in range(a, b + 1) if i in ref]
    return np.array([[ref[i]["x"], ref[i]["y"]] for i in idx]), idx


def reverse_run():
    """시간 역방향으로 REF를 재실행한다 (격자 인덱스 원점은 정방향과 다를 수 있음)."""
    import cv2
    import ref_track as RT
    Hs = np.load(C.EXP / "config" / "bootstrap_H.npy")
    regions = C.load_regions()
    prev, back, reinit = None, {}, []
    for i in range(len(Hs) - 1, -1, -1):
        g = cv2.cvtColor(C.read_frame(i), cv2.COLOR_BGR2GRAY)
        mask = C.polygon_mask(regions["LEFT"], Hs[i], g.shape, erode_px=25)
        corners = RT.detect_corners(g, mask)
        if corners is None:
            prev = None
            continue
        L = None
        if prev is not None:
            L, _, _ = RT.refit(Hs[i] @ np.linalg.inv(Hs[prev[1]]) @ prev[0], corners)
        if L is None:
            seed = RT.init_lattice_ordered(g, mask)
            if seed is None:
                prev = None
                continue
            L, _, _ = RT.refit(seed, corners)
            if L is None:
                prev = None
                continue
            reinit.append(i)
        back[i] = RT.cam_lattice(L) * C.SQUARE_MM
        prev = (L, i)
    return back, reinit


def main():
    ref = load_ref()
    idx = np.array(sorted(ref))
    xy = np.array([[ref[i]["x"], ref[i]["y"]] for i in idx])
    print(f"REF 프레임 {len(ref)}개\n")
    results = []

    a, b = C.ORIGIN_SEGMENT
    p, _ = seg_xy(ref, a, b)
    win = max(1, len(p) // 20)
    drift = float(np.linalg.norm(p[-win:].mean(0) - p[:win].mean(0)))
    sig = float(np.linalg.norm(p - p.mean(0), axis=1).std())
    results.append(("V1", f"정지 f{a}-{b} ({(b-a)/59.4:.1f}s) 누적 드리프트",
                    f"{drift:.4f} mm", f"< {PASS['V1']}", drift < PASS["V1"]))
    print(f"V1  진짜 무이동 구간: 누적 드리프트 {drift:.4f} mm, 프레임별 흔들림 σ {sig:.4f} mm")

    d0 = np.linalg.norm(xy, axis=1)
    late = np.where(idx > 950)[0]
    k = late[np.argmin(d0[late])]
    loop = float(d0[k])
    results.append(("V2", f"이동 후 원점 최근접(f{idx[k]}) 잔차", f"{loop:.4f} mm",
                    f"< {PASS['V2']}", loop < PASS["V2"]))
    print(f"V2  약 200 mm 왕복 후 f{idx[k]} 에서 원점까지 {loop:.4f} mm "
          f"(좌표 {xy[k,0]:+.3f}, {xy[k,1]:+.3f})")

    xc = np.array([ref[i]["xchk"] for i in idx if not np.isnan(ref[i]["xchk"])])
    xcm = float(np.median(xc))
    results.append(("V3", f"LEFT vs TOP 독립 측정 차이 (n={len(xc)})", f"{xcm:.4f} mm",
                    f"< {PASS['V3']}", xcm < PASS["V3"]))
    print(f"V3  두 시트 독립 측정 차이: med={xcm:.4f}  p95={np.percentile(xc,95):.4f}"
          f"  max={xc.max():.4f} mm")

    mm = np.array([ref[i]["mmpp"] for i in idx])
    var = 100 * (mm.max() - mm.min()) / np.median(mm)
    results.append(("V4", "mm/px 변동폭 (카메라 높이 변화 지표)", f"{var:.2f} %",
                    f"< {PASS['V4']} %", var < PASS["V4"]))
    print(f"V4  mm/px med={np.median(mm):.5f} min={mm.min():.5f} max={mm.max():.5f}"
          f" → 높이 변화 {var:.2f}%")

    print("\nV5  역방향 재실행 중...")
    back, reinit = reverse_run()
    common = [i for i in idx if i in back and i - 1 in back and i - 1 in ref]
    dF = np.array([[ref[i]["x"] - ref[i-1]["x"], ref[i]["y"] - ref[i-1]["y"]] for i in common])
    dB = np.array([back[i] - back[i-1] for i in common])
    e = np.linalg.norm(dF - dB, axis=1)
    spd = np.linalg.norm(dF, axis=1)
    slow = spd < FAST_MM
    emax = float(e[slow].max())
    results.append(("V5", f"정/역방향 증분 차이, 저속 프레임만 (n={int(slow.sum())})",
                    f"max {emax:.4f} mm", f"< {PASS['V5']}", emax < PASS["V5"]))
    print(f"V5  저속(<{FAST_MM} mm/f, {100*slow.mean():.0f}%): "
          f"med={np.median(e[slow]):.4f} p99={np.percentile(e[slow],99):.4f} max={emax:.4f} mm")
    print(f"    고속({100*(~slow).mean():.0f}%): med={np.median(e[~slow]):.3f} max={e[~slow].max():.3f} mm"
          f"  ← 모션블러로 REF 정밀도 저하 (보드 선명도 -27%, 격자잔차 0.48->0.84 px, 코너 164->114)")
    print(f"    (증분은 격자 인덱스 원점에 불변이라 재초기화 {len(reinit)}회의 영향을 받지 않는다)")

    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    results.append(("V6", "프레임간 최대 이동 (반 칸 5 mm 미만이어야 배정 무모호)",
                    f"{step.max():.3f} mm", f"< {PASS['V6']}", step.max() < PASS["V6"]))
    print(f"\nV6  프레임간 이동 med={np.median(step):.3f} p99={np.percentile(step,99):.3f} "
          f"max={step.max():.3f} mm  (한 칸 {C.SQUARE_MM} mm)")

    W, DEG = 15, 2
    res = []
    for s0 in range(0, len(idx) - W):
        blk = idx[s0:s0 + W]
        if blk[-1] - blk[0] != W - 1:
            continue
        t = np.arange(W, dtype=float)
        q = np.array([[ref[i]["x"], ref[i]["y"]] for i in blk])
        for ax in (0, 1):
            res.append(q[:, ax] - np.polyval(np.polyfit(t, q[:, ax], DEG), t))
    hf = float(np.sqrt((np.concatenate(res) ** 2).mean()))
    print(f"\n참고  슬라이딩 창 고주파 잔차 = {hf:.4f} mm "
          f"(REF 잡음 + 실제 손떨림. V3 보다 크며 상한으로만 읽을 것)")

    print("\n" + "=" * 82)
    print(f"{'ID':4} {'검증':48} {'값':>16} {'기준':>9}  판정")
    print("-" * 82)
    for i, d, v, c, ok in results:
        print(f"{i:4} {d:48} {v:>16} {c:>9}  {'PASS' if ok else 'FAIL'}")
    print("=" * 82)
    gate = all(ok for i, *_, ok in results if i in GATE)
    print("\n1차 관문(V1·V2·V3) 통과 — 추정기 비교로 진행 가능."
          if gate else "\n1차 관문 실패 — 추정기 비교는 의미가 없다.")
    rms = np.array([ref[i]["rms"] for i in idx])
    n = np.array([ref[i]["n"] for i in idx])
    print(f"부가: 격자적합 잔차 med={np.median(rms):.3f} max={rms.max():.3f} px, "
          f"코너수 min={n.min()} med={int(np.median(n))}")


if __name__ == "__main__":
    main()
