"""regions_f0.json 폴리곤을 여러 프레임에 전파해 육안 검증용 시트를 만든다."""
import sys

import cv2
import numpy as np

import common as C

COLORS = {"LEFT": (0, 255, 255), "TOP": (255, 128, 0), "PRINT": (0, 0, 255)}


def main(out_path, frame_ids=(0, 400, 700, 900, 1100, 1200)):
    Hs = np.load(C.EXP / "config" / "bootstrap_H.npy")
    regions = C.load_regions()
    tiles = []
    for fi in frame_ids:
        f = C.read_frame(fi)
        ov = f.copy()
        for name, poly in regions.items():
            pts = C.warp_pts(Hs[fi], poly)
            cv2.polylines(ov, [np.round(pts).astype(np.int32)], True, COLORS[name], 6)
            fill = np.zeros_like(f)
            cv2.fillPoly(fill, [np.round(pts).astype(np.int32)], COLORS[name])
            ov = cv2.addWeighted(ov, 1.0, fill, 0.22, 0)
        vis = {n: 100 * C.visible_fraction(p, Hs[fi], f.shape) for n, p in regions.items()}
        cv2.putText(ov, f"#{fi}  L{vis['LEFT']:.0f}% T{vis['TOP']:.0f}% P{vis['PRINT']:.0f}%",
                    (16, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 8)
        cv2.putText(ov, f"#{fi}  L{vis['LEFT']:.0f}% T{vis['TOP']:.0f}% P{vis['PRINT']:.0f}%",
                    (16, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3)
        tiles.append(cv2.resize(ov, (640, 360)))
    sheet = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])
    cv2.imwrite(out_path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("saved", out_path)


if __name__ == "__main__":
    main(sys.argv[1])
