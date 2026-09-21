"""라이브 XY 추적 — GUI 캘리브레이션 + 영점 대비 카메라 이동 오버레이.

전제: 피사체는 움직이지 않는다(00_프로젝트_개요 참고). 따라서 화면에서 관측되는 모든
KLT 이동은 카메라 자신의 이동이며, 추적 영역의 기본값은 화면 전체다. 중앙값 + MAD
이상점 제거는 오추적·반사광·모션블러를 걸러내기 위해서만 남긴다.

좌표 규약
---------
world 좌표계 = 체커보드 격자(캘리브레이션 시점 L0, 단위 1 = SQUARE_MM). 카메라 이동은

    pos_mm -= (A @ J0 @ duv) * SQUARE_MM

로 누적한다. J0 는 캘리브레이션 시점에 고정한 영상->격자 야코비안(estimate_xy.py 의
'fixed' 스케일 정책과 동일), duv 는 그 프레임의 공통 픽셀 이동량, A 는 사용자가
축 지정(키 a)으로 정한 +X(테이블 장축) 방향에 맞춘 부호있는 2x2 축 치환행렬이다.
영점(키 0)은 pos_mm 을 0 으로 리셋할 뿐 별도 누적기를 두지 않는다.

Y 방향 규약: 궤적 패널은 +Y 를 위로 그린다(compose_xy_v1_dashboard.py 와 동일).
report.py 는 뒤집지 않아 이 규약과 다르다 — 이 파일 안에서만 일관되게 쓴다.

실행:
    python live_xy.py                                  # OAK 라이브, 모노 글로벌셔터(CAM_B)
    python live_xy.py --oak-cam color                   # IMX378 컬러 1920x1080
    python live_xy.py --source ../KakaoTalk_20260901_144821449.mp4   # 녹화 재생(회귀 검증)
    python live_xy.py --calib ../config/live_calib.json --log --record

키: c 캘리브레이션 | a 축(+X) 지정 | r ROI 그리기 | 0/Space 영점 | f 특징점 재검출 |
    g CSV 로깅 | v 영상 녹화 | s 캘리브레이션 저장 | l 캘리브레이션 불러오기 | h 도움말 | q 종료
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import signal
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import common as C

ROOT = C.ROOT
RESULT = ROOT / "03_결과" / "라이브_XY_v1"
DATA = RESULT / "data"
VISUALS = RESULT / "visuals"
CALIB_JSON = C.EXP / "config" / "live_calib.json"

# --- 추정 파라미터 (estimate_xy.py 와 동일 계열, 라이브용으로 조정) -----------
ERODE_PX = 20
MIN_ROI_FRAC = 0.05
REFRESH_BELOW = 250          # 추적점이 이 아래로 떨어지면 재검출
FORCE_REFRESH_EVERY = 30     # 최소 이 주기로는 무조건 재검출 (fps 확보)
TRAJ_MAXLEN = 3000
FLUSH_EVERY = 30

BOARD_CELL_WARN_PX = 15.0    # 체커보드 한 칸이 이보다 작으면 경고
BOARD_RMS_MAX = 2.0
BOARD_MIN_CORNERS = 12

OAK_CONNECT_TIMEOUT_S = 90.0  # 오프라인 직결 시 장치의 DHCP 폴백 지연을 기다리기 위한 총 재시도 시간
OAK_RETRY_INTERVAL_S = 3.0


# =============================================================================
# 프레임 소스
# =============================================================================
class FrameSource(contextlib.AbstractContextManager):
    """(seq, t_s, bgr) 를 내놓는 반복자. with 블록을 벗어나면 자원을 해제한다."""

    def __iter__(self):
        raise NotImplementedError

    def set_ae_awb_lock(self, locked: bool):
        """가능하면 노출/화이트밸런스를 고정한다. 지원 안 하면 조용히 무시한다."""
        pass


class VideoFileSource(FrameSource):
    def __init__(self, path):
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"영상을 열 수 없다: {self.path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0

    def __iter__(self):
        i = 0
        while True:
            ok, f = self.cap.read()
            if not ok:
                break
            yield i, i / self.fps, f
            i += 1

    def __exit__(self, *exc):
        self.cap.release()
        return False


class OakSource(FrameSource):
    """depthai v3 OAK 카메라. test_oak_stream.py 에서 검증한 패턴을 따른다.

    강제 종료(SIGTERM) 시 장치가 ~30초 잠기는 것을 확인했으므로, SIGINT/SIGTERM은
    stop_event만 세우고 루프가 그것을 보고 정상적으로 with 블록을 빠져나가게 한다.

    오프라인 직결(PoE, DHCP 서버 없음) 환경에서는 재연결 시 장치가 DHCP 타임아웃 후
    폴백 주소로 내려오는 데 시간이 걸려, `dai.Pipeline()`의 1회 탐색 창(내부 기본
    탐색 시간)만으로는 못 찾는 경우가 있다. 그래서 탐색 자체를 OAK_CONNECT_TIMEOUT_S
    동안 바깥에서 반복한다.
    """

    FALLBACK_SIZES = [(1280, 800), (1280, 720), (640, 480)]

    def __init__(self, ip="192.168.0.9", cam="left", size=None,
                 connect_timeout=OAK_CONNECT_TIMEOUT_S, retry_interval=OAK_RETRY_INTERVAL_S):
        import depthai as dai
        self.dai = dai
        self.ip = ip
        self.socket = {
            "left": dai.CameraBoardSocket.CAM_B,
            "right": dai.CameraBoardSocket.CAM_C,
            "color": dai.CameraBoardSocket.CAM_A,
        }[cam]
        self.pixel_type = dai.ImgFrame.Type.NV12 if cam != "color" else dai.ImgFrame.Type.BGR888i
        sizes = [size] + self.FALLBACK_SIZES if size else self.FALLBACK_SIZES
        self.pipeline = None
        self.queue = None
        self.ctrl_queue = None
        self.stop_event = False
        self._prev_handlers = {}

        deadline = time.time() + connect_timeout
        attempt = 0
        last_err = None
        while True:
            attempt += 1
            for sz in sizes:
                try:
                    self.pipeline = dai.Pipeline()
                    self.cam_node = self.pipeline.create(dai.node.Camera, boardSocket=self.socket).build()
                    self.ctrl_queue = self.cam_node.inputControl.createInputQueue()
                    self.queue = self.cam_node.requestOutput(sz, self.pixel_type).createOutputQueue()
                    self.pipeline.start()
                    self.size = sz
                    print(f"OAK 연결됨: {ip} / cam={cam} / {sz[0]}x{sz[1]} (탐색 {attempt}회)")
                    return
                except Exception as e:  # noqa: BLE001 — 폴백 체인이므로 넓게 잡는다
                    last_err = e
                    self.pipeline = None
                    continue
            if time.time() >= deadline:
                raise RuntimeError(
                    f"OAK 스트림을 열 수 없다 ({ip}, cam={cam}), {attempt}회 탐색 후 포기"
                    f" (총 {connect_timeout:.0f}s): {last_err}")
            remaining = deadline - time.time()
            print(f"OAK 장치를 찾지 못함 (탐색 {attempt}회) — 오프라인 직결 시 장치의 DHCP 폴백"
                  f" 지연일 수 있음. {retry_interval:.0f}초 후 재시도 (남은 시간 {remaining:.0f}s): {last_err}")
            time.sleep(retry_interval)

    def _install_signal_handlers(self):
        def handler(signum, frame):
            self.stop_event = True
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._prev_handlers[sig] = signal.signal(sig, handler)

    def _restore_signal_handlers(self):
        for sig, h in self._prev_handlers.items():
            signal.signal(sig, h)

    def set_ae_awb_lock(self, locked: bool):
        if self.ctrl_queue is None:
            return
        ctrl = self.dai.CameraControl()
        ctrl.setAutoExposureLock(bool(locked))
        ctrl.setAutoWhiteBalanceLock(bool(locked))
        try:
            self.ctrl_queue.send(ctrl)
        except Exception as e:  # noqa: BLE001
            print(f"노출 고정 실패(무시하고 계속): {e}")

    def __iter__(self):
        self._install_signal_handlers()
        t0 = time.time()
        i = 0
        try:
            while self.pipeline.isRunning() and not self.stop_event:
                msg = self.queue.get()
                frame = msg.getCvFrame()
                yield i, time.time() - t0, frame
                i += 1
        finally:
            self._restore_signal_handlers()

    def __exit__(self, *exc):
        if self.pipeline is not None:
            self.pipeline.stop()
        return False


def open_source(spec: str, oak_cam: str, oak_size, oak_connect_timeout=OAK_CONNECT_TIMEOUT_S) -> FrameSource:
    if spec == "oak":
        return OakSource(cam=oak_cam, size=oak_size, connect_timeout=oak_connect_timeout)
    return VideoFileSource(spec)


# =============================================================================
# 상태
# =============================================================================
class Mode:
    IDLE = "idle"
    ROI = "roi"
    AXIS = "axis"


class AppState:
    def __init__(self, frame_shape):
        self.img_center = C.img_center(frame_shape)
        self.frame_shape = frame_shape

        # 캘리브레이션
        self.calibrated = False
        self.L0 = None
        self.J0 = None
        self.mm_per_px = None
        self.board_size = None
        self.axis_matrix = np.eye(2)          # A: 격자축 -> +X/+Y
        self.axis_note = "not set (using lattice axes)"
        self.banner = None                     # (text, color, expire_ts)

        # 추적 상태
        self.pos_mm = np.zeros(2)
        self.prev_gray = None
        self.prev_pts = None
        self.frames_since_refresh = 0
        self.roi_polygon = None                # None = 화면 전체
        self.roi_mode_draw = []                 # 그리는 중인 폴리곤 꼭짓점
        self.axis_click_pts = []
        self.mode = Mode.IDLE
        self.last_flags = ""
        self.last_tracked = 0
        self.last_inlier = 0.0
        self.last_roi_visible = 1.0
        self.last_du = np.zeros(2)

        # 궤적 / IO
        self.trajectory = []                    # ring buffer (mm)
        self.csv_writer = None
        self.csv_file = None
        self.csv_rows_since_flush = 0
        self.video_writer = None
        self.session_ts = None

        self.show_help = False

    # --- 캘리브레이션 -------------------------------------------------------
    def set_banner(self, text, color, seconds=2.0):
        self.banner = (text, color, time.time() + seconds)

    def apply_calibration(self, L0, mm_per_px, board_size):
        self.L0 = L0
        self.mm_per_px = mm_per_px
        self.board_size = board_size
        self.J0 = C.jacobian_img_to_lattice(L0, self.img_center)
        self.calibrated = True
        # 재캘리브레이션 시 기존 축 지정은 더 이상 이 L0 기준이 아니므로 초기화한다.
        self.axis_matrix = np.eye(2)
        self.axis_note = "not set (using lattice axes)"

    def set_axis(self, p1_img, p2_img):
        """클릭 두 점으로 +X 방향을 지정 -> 가장 잘 정렬되는 격자축을 +X로 삼는 A."""
        if self.L0 is None:
            return
        v_img = np.asarray(p2_img, float) - np.asarray(p1_img, float)
        if np.linalg.norm(v_img) < 1e-6:
            return
        # 격자좌표계에서 이 방향 벡터: J0 는 영상->격자 야코비안이므로 그대로 곱하면 된다.
        v_lat = self.J0 @ v_img
        # (axis_matrix, 오버레이용 영문 설명) — axis_note 는 draw_overlay 상태줄에 그려진다.
        candidates = [
            (np.array([[1.0, 0.0], [0.0, 1.0]]), "as-is (+X=axis1, +Y=axis2)"),
            (np.array([[-1.0, 0.0], [0.0, -1.0]]), "180deg (+X=-axis1, +Y=-axis2)"),
            (np.array([[0.0, 1.0], [-1.0, 0.0]]), "CW90 (+X=axis2, +Y=-axis1)"),
            (np.array([[0.0, -1.0], [1.0, 0.0]]), "CCW90 (+X=-axis2, +Y=axis1)"),
        ]
        best = max(candidates, key=lambda c: (c[0] @ v_lat)[0] / (np.linalg.norm(v_lat) + 1e-9))
        self.axis_matrix, desc = best
        self.axis_note = f"+X aligned to click [{desc}]"
        print(f"축 지정: {self.axis_note}")

    def zero_here(self):
        self.pos_mm = np.zeros(2)
        self.trajectory.clear()
        print("영점 재설정: 현재 위치 = (0, 0) mm")

    def origin_img_pos(self):
        """영점(세계 원점)의 현재 화면 픽셀 위치. pos_mm 에서 역산한다."""
        if not self.calibrated:
            return None
        try:
            Ainv = np.linalg.inv(self.axis_matrix)
            J0inv = np.linalg.inv(self.J0)
        except np.linalg.LinAlgError:
            return None
        duv_lat = Ainv @ (self.pos_mm / C.SQUARE_MM)
        offset_img = J0inv @ duv_lat
        return self.img_center - offset_img


# =============================================================================
# 캘리브레이션
# =============================================================================
def try_calibrate(gray, mask=None):
    """체커보드 검출 -> (result_dict, None) 또는 (None, reason_en).

    reason_en 은 화면 배너에 그대로 쓰이므로 영어다. 콘솔 로그는 호출부에서 따로 한국어로 남긴다.
    """
    corners, size, L0, rms = C.detect_board(
        gray, mask=mask, max_rms=BOARD_RMS_MAX, min_corners=BOARD_MIN_CORNERS)
    if corners is None:
        return None, "Checkerboard not found - show the full board and retry"
    mmpp = C.mm_per_px(L0, C.SQUARE_MM)
    cell_px = C.SQUARE_MM / mmpp
    result = dict(L0=L0, mm_per_px=mmpp, board_size=size, cell_px=cell_px, rms=rms, n_corners=len(corners))
    return result, None


# =============================================================================
# 추정 루프
# =============================================================================
def build_mask(shape, roi_polygon):
    if roi_polygon is None or len(roi_polygon) < 3:
        m = np.full(shape[:2], 255, np.uint8)
        if ERODE_PX > 0:
            k = 2 * ERODE_PX + 1
            m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        return m
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(roi_polygon)).astype(np.int32)], 255)
    k = 2 * ERODE_PX + 1
    return cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


def roi_visible_fraction(shape, roi_polygon):
    if roi_polygon is None or len(roi_polygon) < 3:
        return 1.0
    h, w = shape[:2]
    pts = np.asarray(roi_polygon, np.float32)
    full = cv2.contourArea(pts)
    if full <= 1:
        return 0.0
    clip = np.clip(pts, [0, 0], [w, h]).astype(np.float32)
    return float(cv2.contourArea(clip) / full)


def step(st: AppState, gray, roi_mode="fixed"):
    """한 프레임 처리. st.pos_mm 등을 갱신하고 flags 문자열을 반환한다."""
    flags = []
    mask = build_mask(gray.shape, st.roi_polygon)
    vis = roi_visible_fraction(gray.shape, st.roi_polygon)
    st.last_roi_visible = vis

    ntr, ratio = st.last_tracked, st.last_inlier
    duv = np.zeros(2)
    if not st.calibrated:
        flags.append("no_calib")
    elif vis < MIN_ROI_FRAC:
        flags.append("roi_out_of_frame")
    elif st.prev_gray is None or st.prev_pts is None or len(st.prev_pts) < C.MIN_TRACKED:
        flags.append("no_features")
    else:
        p1, stt, _ = cv2.calcOpticalFlowPyrLK(
            st.prev_gray, gray, st.prev_pts.reshape(-1, 1, 2), None,
            winSize=(21, 21), maxLevel=3)
        stt = stt.ravel().astype(bool) if p1 is not None else np.zeros(len(st.prev_pts), bool)
        if stt.sum() < C.MIN_TRACKED:
            flags.append("track_lost")
        else:
            duv, ratio, ntr = C.robust_translation(st.prev_pts[stt], p1.reshape(-1, 2)[stt])
            if ratio < 0.5:
                flags.append("low_inlier")
            st.pos_mm = st.pos_mm - (st.axis_matrix @ st.J0 @ duv) * C.SQUARE_MM
            # ROI 전파(병진만) — 사용자가 부분 ROI를 지정했을 때만 의미가 있다.
            # 스케일·회전까지 먹이면 ROI가 서서히 커지거나 돌아가며 드리프트하므로 쓰지 않는다.
            if roi_mode == "follow" and st.roi_polygon is not None and "track_lost" not in flags:
                st.roi_polygon = [(p[0] + duv[0], p[1] + duv[1]) for p in st.roi_polygon]

    st.last_du = duv
    st.last_tracked, st.last_inlier = ntr, ratio
    st.last_flags = "|".join(flags)

    # 특징점 재검출 정책: 부족하거나 주기 도래 시
    st.frames_since_refresh += 1
    need_refresh = (
        st.prev_pts is None or len(st.prev_pts) < REFRESH_BELOW
        or st.frames_since_refresh >= FORCE_REFRESH_EVERY
    )
    if need_refresh and mask.mean() > 1:
        mask = build_mask(gray.shape, st.roi_polygon)  # ROI가 전파됐을 수 있으므로 다시 계산
        q = cv2.goodFeaturesToTrack(gray, 800, 0.01, 8, mask=mask)
        st.prev_pts = None if q is None else q.reshape(-1, 2)
        st.frames_since_refresh = 0

    st.prev_gray = gray
    st.trajectory.append(tuple(st.pos_mm))
    if len(st.trajectory) > TRAJ_MAXLEN:
        st.trajectory.pop(0)
    return st.last_flags


# =============================================================================
# 오버레이
# =============================================================================
def put_text_outlined(img, text, org, scale=0.55, color=(255, 255, 255), thickness=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def clamp_to_edge(pt, shape, margin=24):
    h, w = shape[:2]
    x, y = pt
    x = np.clip(x, margin, w - margin)
    y = np.clip(y, margin, h - margin)
    return np.array([x, y])


def draw_trajectory_panel(canvas, trajectory, size=220, margin=16):
    h, w = canvas.shape[:2]
    x0, y0 = w - size - margin, h - size - margin
    panel = np.full((size, size, 3), 28, np.uint8)
    cv2.rectangle(panel, (0, 0), (size - 1, size - 1), (90, 90, 90), 1)
    if len(trajectory) > 1:
        pts = np.asarray(trajectory, np.float64)
        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        span = max(float((hi - lo).max()), 10.0) * 1.15
        center = (lo + hi) / 2
        lo2, hi2 = center - span / 2, center + span / 2

        def mapper(p):
            q = (p - lo2) / (hi2 - lo2)
            # +Y 를 위로: 화면 y 는 아래가 양수이므로 1-q[1]
            return (int(q[0] * (size - 20) + 10), int((1.0 - q[1]) * (size - 20) + 10))

        for frac in (0.25, 0.5, 0.75):
            v = int(10 + frac * (size - 20))
            cv2.line(panel, (v, 10), (v, size - 10), (55, 55, 55), 1)
            cv2.line(panel, (10, v), (size - 10, v), (55, 55, 55), 1)
        q = np.array([mapper(p) for p in pts], np.int32)
        cv2.polylines(panel, [q], False, (60, 200, 255), 2, cv2.LINE_AA)
        cv2.circle(panel, tuple(q[-1]), 5, (60, 200, 255), -1, cv2.LINE_AA)
        cv2.circle(panel, mapper(np.zeros(2)), 4, (0, 255, 0), 2, cv2.LINE_AA)
        put_text_outlined(panel, f"{span:.0f}mm span", (8, size - 8), 0.35, (200, 200, 200))
    canvas[y0:y0 + size, x0:x0 + size] = panel
    cv2.rectangle(canvas, (x0, y0), (x0 + size, y0 + size), (110, 110, 110), 1)


def draw_overlay(frame, st: AppState, fps):
    vis = frame.copy()
    h, w = vis.shape[:2]

    # ROI
    if st.roi_polygon is not None and len(st.roi_polygon) >= 3:
        pts = np.round(np.asarray(st.roi_polygon)).astype(np.int32)
        overlay = vis.copy()
        cv2.fillPoly(overlay, [pts], (0, 0, 255))
        vis = cv2.addWeighted(vis, 1.0, overlay, 0.22, 0)
        cv2.polylines(vis, [pts], True, (0, 0, 255), 2)
    if st.mode == Mode.ROI and len(st.roi_mode_draw) > 0:
        pts = np.round(np.asarray(st.roi_mode_draw)).astype(np.int32)
        for p in pts:
            cv2.circle(vis, tuple(p), 4, (0, 255, 255), -1)
        if len(pts) > 1:
            cv2.polylines(vis, [pts], False, (0, 255, 255), 1)

    # 추적 벡터 (샘플링)
    if st.prev_pts is not None and len(st.prev_pts) > 0:
        step_n = max(1, len(st.prev_pts) // 150)
        for p in st.prev_pts[::step_n]:
            cv2.circle(vis, (int(p[0]), int(p[1])), 2, (0, 200, 0), -1)

    center_i = tuple(np.round(st.img_center).astype(int))
    cv2.drawMarker(vis, center_i, (200, 200, 200), cv2.MARKER_CROSS, 14, 1)

    # 영점 십자선 + 변위 화살표
    if st.calibrated:
        origin = st.origin_img_pos()
        if origin is not None:
            clamped = clamp_to_edge(origin, vis.shape)
            oi = tuple(np.round(clamped).astype(int))
            off_screen = (origin[0] < 0 or origin[0] >= w or origin[1] < 0 or origin[1] >= h)
            color = (0, 255, 255) if not off_screen else (0, 140, 255)
            cv2.drawMarker(vis, oi, color, cv2.MARKER_TILTED_CROSS, 22, 2)
            cv2.circle(vis, oi, 10, color, 1)
            cv2.arrowedLine(vis, oi, center_i, (0, 200, 255), 2, tipLength=0.15)
            if off_screen:
                d = float(np.linalg.norm(st.pos_mm))
                put_text_outlined(vis, f"origin off-screen ({d:.0f}mm)", (oi[0] - 60, oi[1] - 16), 0.45, (0, 140, 255))

    # 축 지정 클릭
    if st.mode == Mode.AXIS and len(st.axis_click_pts) > 0:
        for p in st.axis_click_pts:
            cv2.circle(vis, tuple(np.round(p).astype(int)), 5, (255, 0, 255), -1)
        if len(st.axis_click_pts) == 2:
            cv2.arrowedLine(vis, tuple(np.round(st.axis_click_pts[0]).astype(int)),
                            tuple(np.round(st.axis_click_pts[1]).astype(int)), (255, 0, 255), 2)

    # HUD
    d = float(np.linalg.norm(st.pos_mm))
    hud = (f"XY=({st.pos_mm[0]:+.2f},{st.pos_mm[1]:+.2f})mm  d={d:.2f}mm  "
           f"pts={st.last_tracked}  inl={st.last_inlier:.2f}  "
           f"mm/px={'%.4f' % st.mm_per_px if st.mm_per_px else '--'}  fps={fps:.1f}")
    hud_color = (255, 255, 255)
    if st.last_flags:
        hud_color = (0, 165, 255)
    put_text_outlined(vis, hud, (10, 26), 0.55, hud_color)

    status = f"{'calibrated' if st.calibrated else 'not calibrated'} | axis: {st.axis_note}"
    put_text_outlined(vis, status, (10, 50), 0.45, (200, 200, 200))
    if st.last_flags:
        put_text_outlined(vis, f"flag: {st.last_flags}", (10, 72), 0.45, (0, 165, 255))

    rec_x = w - 20
    if st.video_writer is not None:
        cv2.circle(vis, (rec_x, 20), 6, (0, 0, 255), -1)
        put_text_outlined(vis, "REC", (rec_x - 42, 26), 0.5, (0, 0, 255))
        rec_x -= 70
    if st.csv_writer is not None:
        put_text_outlined(vis, "LOG", (rec_x - 42, 26), 0.5, (0, 220, 0))

    if st.banner is not None:
        text, color, expire = st.banner
        if time.time() < expire:
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            bx, by = (w - tw) // 2, 40
            cv2.rectangle(vis, (bx - 12, by - th - 10), (bx + tw + 12, by + 10), (0, 0, 0), -1)
            cv2.putText(vis, text, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        else:
            st.banner = None

    if st.show_help:
        draw_help(vis)

    draw_trajectory_panel(vis, st.trajectory)
    return vis


HELP_LINES = [
    "c calibrate board   a set +X axis (click 2 pts)   r draw ROI (LMB add / RMB undo / Enter confirm)",
    "0 / Space reset origin   f force refeature   g toggle CSV log   v toggle overlay recording",
    "s save calibration   l load calibration   h toggle help   q quit   (ESC: cancel mode / quit)",
]


def draw_help(vis):
    h, w = vis.shape[:2]
    y = h - 20 - 22 * len(HELP_LINES)
    for line in HELP_LINES:
        put_text_outlined(vis, line, (10, y), 0.42, (220, 220, 220))
        y += 22


# =============================================================================
# 출력
# =============================================================================
def start_csv(st: AppState, ts):
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / f"live_xy_{ts}.csv"
    fh = open(path, "w", newline="", encoding="utf-8")
    w = csv.writer(fh)
    w.writerow(["frame", "t_s", "X_mm", "Y_mm", "dX_mm", "dY_mm", "tracked_points",
                "inlier_ratio", "roi_visible", "mm_per_px", "du_px", "dv_px", "flags"])
    st.csv_file, st.csv_writer = fh, w
    st.csv_rows_since_flush = 0
    print(f"CSV 로깅 시작: {path}")


def stop_csv(st: AppState):
    if st.csv_file is not None:
        st.csv_file.close()
        print("CSV 로깅 종료")
    st.csv_file = st.csv_writer = None


def log_row(st: AppState, frame_idx, t_s, prev_pos):
    if st.csv_writer is None:
        return
    dpos = st.pos_mm - prev_pos
    st.csv_writer.writerow([
        frame_idx, f"{t_s:.4f}", f"{st.pos_mm[0]:.4f}", f"{st.pos_mm[1]:.4f}",
        f"{dpos[0]:.4f}", f"{dpos[1]:.4f}", st.last_tracked, f"{st.last_inlier:.4f}",
        f"{st.last_roi_visible:.4f}", f"{st.mm_per_px:.6f}" if st.mm_per_px else "",
        f"{st.last_du[0]:.4f}", f"{st.last_du[1]:.4f}", st.last_flags,
    ])
    st.csv_rows_since_flush += 1
    if st.csv_rows_since_flush >= FLUSH_EVERY:
        st.csv_file.flush()
        st.csv_rows_since_flush = 0


def start_video(st: AppState, ts, frame_shape, fps):
    VISUALS.mkdir(parents=True, exist_ok=True)
    path = VISUALS / f"overlay_{ts}.mp4"
    h, w = frame_shape[:2]
    st.video_writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), max(fps, 1.0), (w, h))
    print(f"오버레이 녹화 시작: {path}")


def stop_video(st: AppState):
    if st.video_writer is not None:
        st.video_writer.release()
        print("오버레이 녹화 종료")
    st.video_writer = None


def save_calibration(st: AppState, source_desc, frame_shape):
    if not st.calibrated:
        print("캘리브레이션 없음 — 저장할 것이 없다")
        return
    C.EXP.joinpath("config").mkdir(parents=True, exist_ok=True)
    h, w = frame_shape[:2]
    data = dict(
        version=1,
        saved_at=datetime.now().isoformat(timespec="seconds"),
        source=source_desc,
        frame_size=[w, h],
        square_mm=C.SQUARE_MM,
        board_size=list(st.board_size) if st.board_size else None,
        L0=st.L0.tolist(),
        mm_per_px=st.mm_per_px,
        axis_matrix=st.axis_matrix.tolist(),
        axis_note=st.axis_note,
        roi_polygon=[list(p) for p in st.roi_polygon] if st.roi_polygon is not None else None,
    )
    CALIB_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"캘리브레이션 저장: {CALIB_JSON}")


def load_calibration(st: AppState, path, frame_shape):
    path = Path(path)
    if not path.exists():
        print(f"캘리브레이션 파일 없음: {path}")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    w, h = data["frame_size"]
    fh, fw = frame_shape[:2]
    if (fw, fh) != (w, h):
        print(f"거부: 저장된 해상도({w}x{h})가 현재 프레임({fw}x{fh})과 다르다 — 스케일이 틀어진다")
        return
    st.L0 = np.array(data["L0"], float)
    st.mm_per_px = data["mm_per_px"]
    st.board_size = tuple(data["board_size"]) if data.get("board_size") else None
    st.J0 = C.jacobian_img_to_lattice(st.L0, st.img_center)
    st.calibrated = True
    st.axis_matrix = np.array(data.get("axis_matrix", np.eye(2).tolist()), float)
    st.axis_note = data.get("axis_note", "loaded calibration")
    if data.get("roi_polygon"):
        st.roi_polygon = [tuple(p) for p in data["roi_polygon"]]
    st.set_banner("Calibration loaded", (0, 255, 0))
    print(f"캘리브레이션 불러옴: {path} (mm/px={st.mm_per_px:.4f})")


# =============================================================================
# 마우스 / 키
# =============================================================================
def make_mouse_callback(st: AppState):
    def cb(event, x, y, flags, param):
        if st.mode == Mode.ROI:
            if event == cv2.EVENT_LBUTTONDOWN:
                st.roi_mode_draw.append((x, y))
            elif event == cv2.EVENT_RBUTTONDOWN and st.roi_mode_draw:
                st.roi_mode_draw.pop()
        elif st.mode == Mode.AXIS:
            if event == cv2.EVENT_LBUTTONDOWN:
                st.axis_click_pts.append((x, y))
                if len(st.axis_click_pts) == 2:
                    st.set_axis(st.axis_click_pts[0], st.axis_click_pts[1])
                    st.mode = Mode.IDLE
                    st.axis_click_pts = []
    return cb


def handle_key(key, st: AppState, gray_raw, source_desc, frame_shape, fps, calib_path_arg, source=None):
    """gray_raw: CLAHE 없는 원본 그레이 프레임 (체커보드 검출 전용, §CLAHE 주의 참고)."""
    if key == -1:
        return True  # 계속
    ch = chr(key & 0xFF) if 0 <= (key & 0xFF) < 256 else ""

    if ch == "q":
        return False
    if key == 27:  # ESC
        if st.mode != Mode.IDLE:
            st.mode = Mode.IDLE
            st.roi_mode_draw = []
            st.axis_click_pts = []
        else:
            return False
        return True
    if ch == "c":
        result, err = try_calibrate(gray_raw, build_mask(gray_raw.shape, st.roi_polygon)
                                     if st.roi_polygon else None)
        if result is None:
            st.set_banner(err, (0, 0, 255), 3.0)
            print(f"캘리브레이션 실패: {err}")
        else:
            st.apply_calibration(result["L0"], result["mm_per_px"], result["board_size"])
            print(f"캘리브레이션 성공: {result['board_size']} mm/px={result['mm_per_px']:.4f} "
                  f"한칸={result['cell_px']:.1f}px")
            banner_msg = (f"Calibrated: {result['board_size']} mm/px={result['mm_per_px']:.4f} "
                         f"cell={result['cell_px']:.1f}px")
            st.set_banner(banner_msg, (0, 255, 0), 2.0)
            if source is not None:
                source.set_ae_awb_lock(True)
                print("노출/화이트밸런스 고정 (_STATUS.md 촬영 조건)")
            if result["cell_px"] < BOARD_CELL_WARN_PX:
                print(f"경고: 체커보드 한 칸이 {result['cell_px']:.1f}px 로 작다 — 검출이 불안정할 수 있다")
                warn_en = f"Warning: board cell only {result['cell_px']:.1f}px - detection may be unstable"
                st.set_banner(warn_en, (0, 165, 255), 3.0)
    elif ch == "a":
        st.mode = Mode.AXIS
        st.axis_click_pts = []
        print("축 지정 모드: 테이블 장축(+X) 방향으로 두 점을 클릭한다")
    elif ch == "r":
        st.mode = Mode.ROI
        st.roi_mode_draw = []
        print("ROI 그리기 모드: 좌클릭으로 꼭짓점 추가, 우클릭으로 취소, Enter 로 확정")
    elif key == 13 or key == 10:  # Enter
        if st.mode == Mode.ROI:
            if len(st.roi_mode_draw) >= 3:
                st.roi_polygon = list(st.roi_mode_draw)
                print(f"ROI 확정: 꼭짓점 {len(st.roi_polygon)}개")
            else:
                st.roi_polygon = None
                print("ROI 해제: 화면 전체 사용")
            st.mode = Mode.IDLE
            st.roi_mode_draw = []
    elif ch in ("0", " "):
        st.zero_here()
    elif ch == "f":
        st.prev_pts = None
        st.frames_since_refresh = FORCE_REFRESH_EVERY
        print("특징점 강제 재검출 예약")
    elif ch == "g":
        if st.csv_writer is None:
            start_csv(st, st.session_ts)
        else:
            stop_csv(st)
    elif ch == "v":
        if st.video_writer is None:
            start_video(st, st.session_ts, frame_shape, fps)
        else:
            stop_video(st)
    elif ch == "s":
        save_calibration(st, source_desc, frame_shape)
    elif ch == "l":
        load_calibration(st, calib_path_arg or CALIB_JSON, frame_shape)
    elif ch == "h":
        st.show_help = not st.show_help
    return True


# =============================================================================
# 메인
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="라이브 XY 추적 — GUI 캘리브레이션 + 영점 오버레이")
    ap.add_argument("--source", default="oak", help="'oak' 또는 재생할 영상 경로")
    ap.add_argument("--oak-cam", default="left", choices=["left", "right", "color"],
                    help="left/right=OV9282 글로벌셔터 모노, color=IMX378 롤링셔터 컬러 (기본 left)")
    ap.add_argument("--oak-size", default=None, help="예: 1280x800. 생략 시 자동 폴백")
    ap.add_argument("--oak-connect-timeout", type=float, default=OAK_CONNECT_TIMEOUT_S,
                    help=f"OAK 장치 탐색 재시도 총 시간(초), 기본 {OAK_CONNECT_TIMEOUT_S:.0f}s "
                         "(오프라인 직결 시 DHCP 폴백 대기용)")
    ap.add_argument("--calib", default=None, help="시작 시 불러올 캘리브레이션 JSON 경로")
    ap.add_argument("--roi-mode", default="fixed", choices=["fixed", "follow"],
                    help="follow: 지정한 ROI 를 추적점으로 전파(병진만). 화면전체 기본경로에는 영향 없음")
    ap.add_argument("--no-clahe", action="store_true", help="CLAHE 끄기 (OAK 노출고정 시 비교용)")
    ap.add_argument("--log", action="store_true", help="시작하자마자 CSV 로깅")
    ap.add_argument("--record", action="store_true", help="시작하자마자 오버레이 영상 녹화")
    ap.add_argument("--display-width", type=int, default=None, help="표시 창 폭(픽셀). 원본 비율 유지")
    args = ap.parse_args()

    oak_size = None
    if args.oak_size:
        w, h = args.oak_size.lower().split("x")
        oak_size = (int(w), int(h))

    with open_source(args.source, args.oak_cam, oak_size, args.oak_connect_timeout) as source:
        source_desc = f"oak:{args.oak_cam}" if args.source == "oak" else f"file:{args.source}"
        st = None
        window = "live_xy - live XY tracking"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

        session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        t_fps_window = []
        prev_t = None

        for idx, t_s, frame in source:
            if st is None:
                st = AppState(frame.shape)
                st.session_ts = session_ts
                cv2.setMouseCallback(window, make_mouse_callback(st))
                if args.calib:
                    load_calibration(st, args.calib, frame.shape)
                if args.log:
                    start_csv(st, session_ts)
                if args.record:
                    start_video(st, session_ts, frame.shape, 30.0)

            gray_raw = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 체커보드 검출(캘리브레이션)은 항상 CLAHE 없는 원본 명암을 쓴다 — CLAHE 는 국소
            # 대비를 과증폭해 findChessboardCornersSB 의 서브픽셀 코너를 오히려 흐트러뜨린다
            # (calibrate.py/ref_track.py/run_test2_ir_only.py 전부 보드 검출에 CLAHE를 쓰지 않는다).
            # KLT 추적은 별도로 CLAHE 를 켤 수 있다(폰 AE/AWB 변동 대응 목적. OAK 노출고정 시 불필요할 수 있음).
            gray = C.clahe_gray(gray_raw) if not args.no_clahe else gray_raw

            prev_pos = st.pos_mm.copy()
            step(st, gray, roi_mode=args.roi_mode)

            now = time.time()
            if prev_t is not None:
                t_fps_window.append(now - prev_t)
                if len(t_fps_window) > 30:
                    t_fps_window.pop(0)
            prev_t = now
            fps = 1.0 / (sum(t_fps_window) / len(t_fps_window)) if t_fps_window else 0.0

            vis = draw_overlay(frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR), st, fps)
            if args.display_width:
                sc = args.display_width / vis.shape[1]
                vis_disp = cv2.resize(vis, (args.display_width, int(vis.shape[0] * sc)))
            else:
                vis_disp = vis
            cv2.imshow(window, vis_disp)

            log_row(st, idx, t_s, prev_pos)
            if st.video_writer is not None:
                st.video_writer.write(vis)

            key = cv2.waitKey(1)
            if key != -1:
                keep_going = handle_key(key, st, gray_raw, source_desc, frame.shape, fps, args.calib, source)
                if not keep_going:
                    break

        if st is not None:
            stop_csv(st)
            stop_video(st)
        cv2.destroyAllWindows()
        print("종료")


if __name__ == "__main__":
    main()
