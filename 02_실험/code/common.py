"""공용 유틸 — 영상 IO, 영역 마스크, 변환 누적, 광도 정규화.

좌표 규약
---------
world 좌표계 = f0 영상 좌표계. 정지 세계점 X(world)는 프레임 i에서 x = H_i @ X 로 보인다.
따라서 프레임 i의 카메라 위치(world) = H_i^{-1} @ (영상 중심).
H_0 = I, H_i = M_{i-1→i} @ H_{i-1}.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "02_실험"
VIDEO = EXP / "KakaoTalk_20260901_144821449.mp4"
CONFIG = EXP / "config" / "regions_f0.json"
OUT = ROOT / "03_결과" / "XY추정_v1"
DATA = OUT / "data"
VISUALS = OUT / "visuals"
RERUN = OUT / "rerun"

SQUARE_MM = 10.0
IMG_CENTER = np.float64([960.0, 540.0])

# _STATUS.md 실측 구간 (frame 단위, 끝 포함)
STILL_SEGMENTS = [(2, 456), (866, 902), (1034, 1058), (1154, 1205)]
MOVE_SEGMENTS = [(624, 866), (902, 1038), (1058, 1154)]
ORIGIN_SEGMENT = (2, 456)


def open_video(path=None):
    path = path or VIDEO
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없다: {path}")
    return cap


def video_info(path=None):
    cap = open_video(path)
    info = dict(
        n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        fps=float(cap.get(cv2.CAP_PROP_FPS)),
        w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()
    return info


def frames(start=0, stop=None, step=1, path=None):
    """(index, BGR frame) 순차 생성."""
    cap = open_video(path)
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    i = start
    while True:
        ok, f = cap.read()
        if not ok or (stop is not None and i >= stop):
            break
        if (i - start) % step == 0:
            yield i, f
        i += 1
    cap.release()


def read_frame(idx, path=None):
    cap = open_video(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, f = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"프레임 {idx}를 읽을 수 없다")
    return f


def img_center(shape):
    """(h,w[,c]) 프레임 크기에서 영상 중심을 구한다. 해상도가 다른 라이브 소스용."""
    h, w = shape[:2]
    return np.float64([w / 2.0, h / 2.0])


def load_regions():
    with open(CONFIG, encoding="utf-8") as fh:
        cfg = json.load(fh)
    return {k: np.float32(v) for k, v in cfg["regions"].items()}


# --- 광도 정규화 ------------------------------------------------------------
# 폰의 AE/AWB가 촬영 중 작동한다 (median saturation 49 -> 167). 추적 전 필수.
_CLAHE = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))


def prep_gray(bgr):
    return _CLAHE.apply(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))


def clahe_gray(gray):
    """이미 그레이인 프레임에 같은 CLAHE 를 적용한다 (라이브 소스가 색변환을 이미 한 경우)."""
    return _CLAHE.apply(gray)


# --- 변환 -------------------------------------------------------------------
def as_h(affine):
    return np.vstack([affine, [0.0, 0.0, 1.0]])


def decompose_sim(H):
    """유사변환 H -> (scale, theta_deg, tx, ty)."""
    s = float(np.hypot(H[0, 0], H[1, 0]))
    th = float(np.degrees(np.arctan2(H[1, 0], H[0, 0])))
    return s, th, float(H[0, 2]), float(H[1, 2])


def warp_pts(H, pts):
    pts = np.asarray(pts, np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H.astype(np.float64)).reshape(-1, 2)


def cam_world(H):
    """프레임 i의 카메라 위치를 world(=f0) 좌표로. H는 world->frame_i."""
    return warp_pts(np.linalg.inv(H), IMG_CENTER[None, :])[0]


def rescale_about_center(H, factor):
    """영상 중심을 고정한 채 H의 스케일만 factor배 한다."""
    c = IMG_CENTER
    T = np.float64([[factor, 0, c[0] * (1 - factor)],
                    [0, factor, c[1] * (1 - factor)],
                    [0, 0, 1]])
    return T @ H


# --- 마스크 -----------------------------------------------------------------
def polygon_mask(poly_f0, H, shape, erode_px=0):
    """f0 좌표 폴리곤을 H로 프레임에 전파해 uint8 마스크를 만든다."""
    h, w = shape[:2]
    m = np.zeros((h, w), np.uint8)
    pts = warp_pts(H, poly_f0)
    cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 255)
    if erode_px > 0:
        k = 2 * int(erode_px) + 1
        m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return m


def visible_fraction(poly_f0, H, shape):
    """폴리곤 중 프레임 안에 들어온 면적 비율(0~1)."""
    h, w = shape[:2]
    pts = warp_pts(H, poly_f0)
    full = cv2.contourArea(pts.astype(np.float32))
    if full <= 1:
        return 0.0
    clip = np.clip(pts, [0, 0], [w, h]).astype(np.float32)
    return float(cv2.contourArea(clip) / full)


# --- 격자 피치 --------------------------------------------------------------
def lattice_pitch(corners):
    """체커보드 새들점 집합의 최근접이웃 거리 중앙값 = 사각 피치(px)."""
    p = np.asarray(corners, np.float32).reshape(-1, 2)
    if len(p) < 8:
        return float("nan")
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return float(np.median(d.min(axis=1)))


def jacobian_img_to_lattice(L, at):
    """영상 좌표 -> 격자 좌표 사상의 2x2 야코비안 (at 에서, 수치미분).

    at 은 명시 인자다(기본값 없음) — 해상도가 다른 라이브 소스에서 1080p 전용
    IMG_CENTER 를 암묵적으로 쓰면 조용히 틀린 스케일이 나오기 때문이다.
    estimate_xy.py 등 1080p 오프라인 파이프라인은 C.IMG_CENTER 를 그대로 넘긴다.
    """
    Li = np.linalg.inv(L)
    h = 1.0
    p0 = warp_pts(Li, [at])[0]
    px = warp_pts(Li, [at + [h, 0]])[0]
    py = warp_pts(Li, [at + [0, h]])[0]
    return np.column_stack([(px - p0) / h, (py - p0) / h])


MIN_TRACKED = 30
INLIER_PX = 1.0


def robust_translation(p0, p1, iters=5, min_tracked=MIN_TRACKED, inlier_px=INLIER_PX):
    """공통 픽셀 이동량 (du,dv) — 중앙값 + MAD 이상점 제거.

    -> (duv, inlier_ratio, n_used). inlier_ratio 는 최종 (du,dv) 에서 inlier_px 이내인
    추적점 비율이며, 오추적·반사광·모션블러 같은 이상점이 얼마나 섞였는지의 지표다.
    """
    d = p1 - p0
    keep = np.ones(len(d), bool)
    for _ in range(iters):
        m = np.median(d[keep], axis=0)
        r = np.linalg.norm(d - m, axis=1)
        mad = np.median(np.abs(r[keep] - np.median(r[keep]))) + 1e-6
        new = r < 3.0 * 1.4826 * mad + 0.5
        if new.sum() < min_tracked or (new == keep).all():
            break
        keep = new
    duv = np.median(d[keep], axis=0)
    inl = float((np.linalg.norm(d - duv, axis=1) < inlier_px).mean())
    return duv, inl, int(keep.sum())


# --- 체커보드 검출 -----------------------------------------------------------
BOARD_SIZES = [(9, 6), (8, 6), (9, 5), (8, 5), (7, 5), (6, 5), (7, 4), (6, 4), (5, 4), (5, 3), (4, 3)]


def detect_board(gray, sizes=BOARD_SIZES, mask=None, max_rms=2.0, min_corners=12):
    """체커보드를 검출해 (코너, (cols,rows), L, rms)를 반환한다. 실패 시 (None, None, None, inf).

    calibrate.py 의 findChessboardCornersSB 탐색 + ref_track.py 의 (7,7) 서브픽셀
    보정을 합친 것이다. mask 를 주면 그 밖은 흰색으로 지워 보드 탐색을 좁힌다
    (run_test2_ir_only.py 의 관용구).

    findChessboardCornersSB 의 첫 성공만 받아들이면 격자 주기성 때문에 잘못된 부분격자에
    걸리는 인덱스 점프 오검출(잔차 20px+)이 섞인다 — ref_track.py 가 이미 문서화한 문제다.
    따라서 후보 크기마다 이상 정수격자에 대한 호모그래피를 실제로 적합해 재투영 RMS로
    검증하고, 통과하는 첫 후보만 채택한다.
    """
    masked = gray if mask is None else np.where(mask > 0, gray, 255).astype(np.uint8)
    for sz in sizes:
        ok, corners = cv2.findChessboardCornersSB(masked, sz, flags=cv2.CALIB_CB_ACCURACY)
        if not ok or len(corners) < min_corners:
            continue
        sub = cv2.cornerSubPix(
            gray, corners.astype(np.float32), (7, 7), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01),
        ).reshape(-1, 2)
        L, rms = lattice_homography(sub, sz)
        if L is not None and rms <= max_rms:
            return sub, sz, L, rms
    return None, None, None, float("inf")


def lattice_homography(corners, size):
    """검출 코너(순서: 행우선, size=(cols,rows))와 이상 정수격자 사이 호모그래피와 재투영 RMS."""
    cols, rows = size
    ideal = np.float32([[i, j] for j in range(rows) for i in range(cols)])
    obs = np.asarray(corners, np.float32).reshape(-1, 2)
    L, _ = cv2.findHomography(ideal, obs, 0)
    if L is None:
        return None, float("inf")
    pred = warp_pts(L, ideal)
    rms = float(np.sqrt(((pred - obs) ** 2).sum(axis=1).mean()))
    return L, rms


def mm_per_px(L, square_mm=SQUARE_MM):
    """L(격자->영상)의 한 칸 픽셀크기로부터 mm/px 스칼라를 구한다."""
    o = warp_pts(L, [[0, 0]])[0]
    return square_mm / float(np.linalg.norm(warp_pts(L, [[1, 0]])[0] - o))


def track_similarity(g0, g1, pts0, ransac_px=1.5):
    """KLT + RANSAC 유사변환. -> (H 3x3, inlier_ratio, n_tracked) 또는 (None, 0, 0)."""
    if pts0 is None or len(pts0) < 8:
        return None, 0.0, 0
    p0 = np.asarray(pts0, np.float32).reshape(-1, 1, 2)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, winSize=(21, 21), maxLevel=3)
    if p1 is None:
        return None, 0.0, 0
    st = st.ravel().astype(bool)
    if st.sum() < 8:
        return None, 0.0, int(st.sum())
    a, inl = cv2.estimateAffinePartial2D(
        p0[st], p1[st], method=cv2.RANSAC, ransacReprojThreshold=ransac_px,
        maxIters=4000, confidence=0.999)
    if a is None:
        return None, 0.0, int(st.sum())
    ratio = float(inl.sum()) / float(st.sum())
    return as_h(a), ratio, int(st.sum())


# --- 렌즈 왜곡 --------------------------------------------------------------
DIST_JSON = EXP / "config" / "distortion.json"
# 초점거리는 자유 캘리브레이션 불가(전체 보드가 안 보임)라 고정값을 쓴다.
# k1·k2 는 이 f 아래에서 정의된 계수이며, 목적은 절대 초점거리 복원이 아니라
# 반경방향 왜곡의 형상 보정이다.
FOCAL_PX = 1450.0


def camera_matrix():
    return np.float64([[FOCAL_PX, 0, IMG_CENTER[0]],
                       [0, FOCAL_PX, IMG_CENTER[1]],
                       [0, 0, 1]])


def load_distortion():
    """(K, dist 4벡터) — distortion.json 이 없으면 왜곡 0."""
    K = camera_matrix()
    if DIST_JSON.exists():
        with open(DIST_JSON, encoding="utf-8") as fh:
            d = json.load(fh)
        return K, np.float64([d["k1"], d["k2"], 0.0, 0.0])
    return K, np.zeros(4)


class Rectifier:
    """왜곡 보정. 보정 후 좌표계를 rectified 라 부르고 이후 모든 기하는 여기서 이뤄진다."""

    def __init__(self, size=(1920, 1080)):
        self.K, self.D = load_distortion()
        self.size = size
        self.enabled = bool(np.any(self.D))
        if self.enabled:
            self.map1, self.map2 = cv2.initUndistortRectifyMap(
                self.K, self.D, None, self.K, size, cv2.CV_16SC2)

    def image(self, img):
        if not self.enabled:
            return img
        return cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR)

    def points(self, pts):
        p = np.asarray(pts, np.float32).reshape(-1, 1, 2)
        if not self.enabled:
            return p.reshape(-1, 2)
        return cv2.undistortPoints(p, self.K, self.D, P=self.K).reshape(-1, 2)
