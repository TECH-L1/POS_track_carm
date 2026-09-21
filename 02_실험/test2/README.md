# test2 — IR-only 카메라 XY 역산

`KakaoTalk_20260901_164138790.mp4`에서 중앙 IR 인쇄물의 텍스처만으로 카메라의
평면 XY 이동을 추정하는 실험이다.

실행:

```powershell
cd ..\code
python run_test2_ir_only.py
```

추정기에 들어가는 특징점은 중앙 IR ROI뿐이다. 좌측·상단 10 mm 체커보드는 다음 용도로만
사용한다.

- 프레임별 체커보드 제외 마스크 생성
- 독립 XY 기준궤적(LEFT 보드)과 TOP 보드 교차검증

휴대폰 IMU는 추정기에 사용하지 않으며, 결과 그래프의 보조 비교 곡선으로만 표시된다.

결과는 `03_결과/test2_IR_only/`에 생성된다.

- `trajectory.csv`: IR-only 추정, 보드 REF, 오차, 마스크 감사값, 보조 IMU 곡선
- `trajectory.png`: XY 및 오차 시계열
- `overlay.mp4`: 노란색 체커보드 제외 마스크, 빨간색 IR ROI, 궤적 오버레이
- `summary.json`: REF 품질과 IR-only 오차 요약

`mask_violation_points`는 체커보드 제외 마스크 내부에 남은 IR KLT 점 수이며, 0이어야 한다.
