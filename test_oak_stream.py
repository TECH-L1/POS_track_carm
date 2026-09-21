import sys
import time

import cv2
import depthai as dai

DEVICE_IP = "192.168.0.9"
MAX_SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 0  # 0 = 무제한 (q/ESC로 종료)
CONNECT_TIMEOUT_S = 90.0  # 오프라인 직결 시 장치의 DHCP 폴백 지연을 기다리기 위한 총 재시도 시간
RETRY_INTERVAL_S = 3.0


def connect_pipeline(timeout=CONNECT_TIMEOUT_S, interval=RETRY_INTERVAL_S):
    """dai.Pipeline()은 생성 시점에 장치를 탐색/연결한다. 오프라인 직결 환경에서는 장치가
    DHCP 타임아웃 후 폴백 주소로 내려오는 데 시간이 걸려, 1회 탐색만으로는 못 찾을 수 있어
    바깥에서 재시도한다."""
    deadline = time.time() + timeout
    attempt = 0
    last_err = None
    while True:
        attempt += 1
        try:
            return dai.Pipeline()
        except Exception as e:  # noqa: BLE001 — 탐색 재시도이므로 넓게 잡는다
            last_err = e
        if time.time() >= deadline:
            raise RuntimeError(f"OAK 장치를 찾을 수 없다 ({attempt}회 탐색, 총 {timeout:.0f}s): {last_err}") from last_err
        remaining = deadline - time.time()
        print(f"OAK 장치를 찾지 못함 (탐색 {attempt}회) — {interval:.0f}초 후 재시도 (남은 시간 {remaining:.0f}s): {last_err}", flush=True)
        time.sleep(interval)


with connect_pipeline() as pipeline:
    cam = pipeline.create(dai.node.Camera).build()
    video_queue = cam.requestOutput((640, 480), dai.ImgFrame.Type.BGR888i).createOutputQueue()

    pipeline.start()
    print(f"연결됨: {DEVICE_IP} / 파이프라인 시작, ESC 또는 q 로 종료", flush=True)

    start = time.time()
    frame_count = 0
    while pipeline.isRunning():
        video_in = video_queue.get()
        frame = video_in.getCvFrame()
        frame_count += 1

        cv2.imshow("OAK RGB Stream - " + DEVICE_IP, frame)
        key = cv2.waitKey(1)
        if key in (ord("q"), 27):
            break
        if MAX_SECONDS and (time.time() - start) > MAX_SECONDS:
            break

    elapsed = time.time() - start
    print(f"수신 프레임: {frame_count}, 경과: {elapsed:.1f}s, FPS: {frame_count/elapsed:.1f}", flush=True)

cv2.destroyAllWindows()
print("장치 정상 종료됨", flush=True)
