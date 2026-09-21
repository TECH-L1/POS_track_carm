# XY추정 v1 — 체커보드 REF 대비 검증 결과

전체 해석은 **`report.md`** 를 볼 것. 이 파일은 산출물 목록과 재현 방법만 적는다.

## 결과 한 줄

열화상 인쇄물 텍스처만으로 카메라 XY 이동을 추정 →
총 이동 773 mm에 대해 **최종 오차 6.36 mm (0.82%)**, 7.6초 정지구간 **드리프트 0.017 mm**.
오차의 지배 요인은 해상도가 아니라 **손에 든 카메라의 롤(+2.5°)·높이 변화(±3~4%)** 였다.

## 재현

```
cd 02_실험/code
python bootstrap.py      # 프레임간 강체 누적 (마스크 전파·격자 배정 예측용)
python calibrate.py      # 렌즈 왜곡 자가보정 -> config/distortion.json (결론: 무시 가능)
python ref_track.py      # 격자 앵커 REF -> data/ref.csv, data/ref_L.npy
python validate_ref.py   # V1~V6 검증
python estimate_xy.py --region PRINT --scale fixed
python estimate_xy.py --region PRINT --scale oracle
python estimate_xy.py --region LEFT  --scale fixed     # 대조 실험
python degrade.py        # 해상도x노이즈 스윕 -> data/sweep.csv
python report.py         # data/trajectory.csv, visuals/*.png, visuals/*.mp4
python compose_xy_v1_dashboard.py  # visuals/ref_est_trajectory_dashboard.mp4
python export_xy_v1_rerun.py       # rerun/xy_v1_ref_est.rrd
```

의존: `opencv-python`, `numpy`, `scipy`, `matplotlib`.
영역 폴리곤은 `02_실험/config/regions_f0.json` (f0 영상좌표계).

## 산출물

| 파일 | 내용 |
|---|---|
| `report.md` | **해석·결론·한계·다음 촬영 조건** |
| `data/trajectory.csv` | 프레임별 REF·EST·오차·신뢰도 통합 (20열) |
| `visuals/trajectory.png` | X·Y 궤적 + 오차 시계열 (정지구간 음영) |
| `visuals/*.mp4` | REF-only, EST-only, combined, and dashboard videos |
| `data/sweep.csv` / `visuals/sweep_heatmap.png` / `data/sweep_log.txt` | 해상도×노이즈 스윕 |
| `data/ref.csv` / `data/ref_L.npy` / `data/ref_origin.npy` | REF 궤적, 프레임별 격자 포즈, 원점 |
| `data/est_*.csv` | Fixed-scale, oracle, and checkerboard control estimates |
| `rerun/xy_v1_ref_est.rrd` | Rerun recording |

## 읽을 때 주의

- **REF는 진값이 아니다.** 집계 지표는 매우 좋지만(독립 교차검증 중앙값 0.11 mm,
  200 mm 왕복 후 루프 오차 0.23 mm), 프레임 1% 수준에서 최대 5 mm 단발 글리치가 있다(V5 FAIL).
  누적 궤적 비교에는 영향이 작지만 **프레임별 순간 오차를 인용할 땐 함께 밝힐 것**.
- **프레임 단위 증분 비교는 REF 잡음(419 µm)에 막힌다.** 해상도 의존성은 정지구간 드리프트로 볼 것.
- **이 영상은 피사체 움직임 분리를 검증하지 못한다** — 인쇄물은 움직이지 않는다.
  원래 블로커의 절반만 풀린 상태다.
