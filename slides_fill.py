# -*- coding: utf-8 -*-
"""Thermal Camera Translation Estimation 슬라이드 3~8 본문 채우기.

서식은 기존 슬라이드 2에서 추출한 규칙을 그대로 따른다.
  L0 = 20pt bold, L1 = 16pt, "라벨:" 부분만 bold + 나머지 regular,
  본문 첫 문단만 buNone(불릿 없음).

문체는 academic-tone 스킬을 따른다. 덱의 종결 정책은 슬라이드 2·10과 동일한
명사형 개조식(~함/~됨/체언 종결)으로 고정하며, 서술형과 혼용하지 않는다.
"""
import copy
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.oxml.ns import qn

BASE = Path(r"D:\02_research\BRAIN\80_기타\의료 분야 응용 가능 열화상 모듈(교수님 요청")
SRC = BASE / "Thermal Camera Translation Estimation_미완.pptx"
DST = BASE / "Thermal Camera Translation Estimation.pptx"
HEATMAP = BASE / "03_결과" / "XY추정_v1" / "visuals" / "sweep_heatmap.png"

LIGHT_STYLE_1 = "{9D7B26C5-4107-4FEC-AEDC-1716B250A1EF}"


# ---------------------------------------------------------------- helpers
def add_runs(para, spec, size, lang="ko-KR"):
    """spec: "굵은부분|나머지" 또는 "전체". '|' 앞부분만 bold."""
    if "|" in spec:
        head, tail = spec.split("|", 1)
        parts = [(head, True), (tail, False)]
    else:
        parts = [(spec, False)]
    for text, bold in parts:
        if text == "":
            continue
        r = para.add_run()
        r.text = text
        r.font.size = Pt(size)
        r.font.bold = bold
        rPr = r._r.get_or_add_rPr()
        rPr.set("lang", lang)
        rPr.set("altLang", lang)


def fill_body(tf, lines, sizes=(20, 16), no_bullet="first", space_before=None):
    """lines: [(level, spec), ...] / no_bullet: "first" | "all_l0" | "none" """
    tf.clear()
    tf.word_wrap = True
    for i, (lvl, spec) in enumerate(lines):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.level = lvl
        drop = (no_bullet == "first" and i == 0) or (no_bullet == "all_l0" and lvl == 0)
        if drop:
            pPr = para._p.get_or_add_pPr()
            pPr.set("marL", "0")
            pPr.set("indent", "0")
            pPr.append(pPr.makeelement(qn("a:buNone"), {}))
        if space_before is not None:
            para.space_before = Pt(space_before)
        add_runs(para, spec, sizes[lvl])


def add_textbox(slide, left, top, width, height, lines, size=14):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    for i, spec in enumerate(lines):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.space_before = Pt(4)
        add_runs(para, spec, size)
    return box


def style_table(tbl_shape, font_size=13, col_widths=None, first_col_bold=False):
    tbl = tbl_shape.table
    # 기본 파랑 밴딩 대신 Light Style 1 (헤더 아래 얇은 선)
    tblPr = tbl._tbl.find(qn("a:tblPr"))
    for child in list(tblPr):
        if child.tag == qn("a:tableStyleId"):
            tblPr.remove(child)
    sid = tblPr.makeelement(qn("a:tableStyleId"), {})
    sid.text = LIGHT_STYLE_1
    tblPr.append(sid)
    tbl.first_row = True
    tbl.horz_banding = False

    if col_widths:
        for c, w in zip(tbl.columns, col_widths):
            c.width = Inches(w)

    for ri, row in enumerate(tbl.rows):
        row.height = Inches(0.30)
        for ci, cell in enumerate(row.cells):
            cell.margin_left = Inches(0.06)
            cell.margin_right = Inches(0.06)
            cell.margin_top = Inches(0.02)
            cell.margin_bottom = Inches(0.02)
            for para in cell.text_frame.paragraphs:
                for r in para.runs:
                    r.font.size = Pt(font_size)
                    r.font.bold = (ri == 0) or (first_col_bold and ci == 0)
                    rPr = r._r.get_or_add_rPr()
                    rPr.set("lang", "ko-KR")
                    rPr.set("altLang", "ko-KR")


def add_table(slide, data, left, top, width, height, **kw):
    rows, cols = len(data), len(data[0])
    shp = slide.shapes.add_table(rows, cols, Inches(left), Inches(top),
                                 Inches(width), Inches(height))
    for ri, row in enumerate(data):
        for ci, val in enumerate(row):
            shp.table.cell(ri, ci).text = val
    style_table(shp, **kw)
    return shp


def place(sh, left, top, width, height):
    """placeholder는 left/width도 함께 써야 상속값이 0으로 떨어지지 않는다."""
    sh.left, sh.top = Inches(left), Inches(top)
    sh.width, sh.height = Inches(width), Inches(height)


def body_ph(slide):
    for sh in slide.shapes:
        if sh.is_placeholder and sh.placeholder_format.idx == 13:
            return sh
    raise KeyError("body placeholder not found")


# ---------------------------------------------------------------- content
S3 = [
    (0, "파이프라인 이원화:| REF(기준)와 EST(추정)의 입력 정보를 분리 — 6개 표본 프레임에서 EST ROI와 체커보드의 겹침 0 px를 확인함"),
    (1, "입력:| 1920×1080, 59.398 fps, 1,206프레임(20.30 s), 총 이동거리 773 mm"),
    (0, "REF 경로 — 비교용 기준 궤적 산출"),
    (1, "체커보드 코너 검출 → 격자좌표 (i, j)에서 영상좌표로의 호모그래피를 매 프레임 독립 적합"),
    (1, "프레임 간 변환을 누적하지 않으므로 랜덤워크 성분이 발생하지 않음. world 좌표계는 좌 보드 격자, 단위 1 = 10 mm"),
    (0, "EST 경로 — 성능 평가 대상 추정기"),
    (1, "열화상 인쇄물 ROI만 잔존, 경계 20 px 침식 → 피라미드 KLT가 특징점 대응을 산출"),
    (1, "중앙값·MAD 이상점 제거 → 공통 픽셀 이동 Δu → Jacobian J_0 이 mm 단위로 변환 → 누적"),
    (1, "EST의 REF 수용 정보는 ① ROI 마스크, ② t=0 초기 축척·방위 2종으로 한정. 이후 구간은 열린 루프"),
    (0, "성립 전제"),
    (1, "영상 조건 — 평면 · 강체 배경 · 밝기 보존 · 소변위 (KLT 동작 조건)"),
    (1, "시스템 조건 — 카메라 높이·방위 불변 (픽셀-mm 고정 변환 조건)"),
    (0, "비교 지표:| 프레임별 REF−EST 오차, 정지구간 드리프트, 해상도×잡음 스윕"),
]

S4 = [
    (0, "① 특징점 선택 — 모서리 선택의 근거"),
    (1, "평탄 영역과 단일 에지는 이동 방향을 결정하지 못함. 가로·세로 명암 변화가 모두 존재하는 점만이 2차원 이동의 유일 해를 제공함"),
    (1, "goodFeaturesToTrack:| ROI 내 최대 2,000점 선택, 품질 임계값 0.01, 점 간 최소 간격 8 px"),
    (0, "② 밝기 보존 가정과 선형화"),
    (1, "I(x, y, t) ≈ I(x+u, y+v, t+Δt) 를 1차 Taylor 전개 → I_x·u + I_y·v + I_t = 0"),
    (1, "미지수 2개에 식 1개 → 창 W 내 전 픽셀에 동일 (u, v)를 부과하고 오차 제곱합을 최소화"),
    (1, "G·[u v]ᵀ = −Σ[I_x·I_t, I_y·I_t]ᵀ,   G = Σ[[I_x², I_x·I_y], [I_x·I_y, I_y²]]  (구조 텐서)"),
    (1, "G의 두 고유값이 모두 충분히 클 때 추적이 성립함 — Tomasi 기준과 동일한 조건"),
    (0, "③ 대변위 대응 — 피라미드"),
    (1, "저해상도에서 개략 해 산출 → 상위 단계로 확대하여 초기값 부여 → 원 해상도에서 최종 보정"),
    (1, "calcOpticalFlowPyrLK:| 창 21×21 px, 최대 피라미드 레벨 3"),
    (0, "④ 이상점 제거 — 중앙값·MAD"),
    (1, "m = median(d_j), r_j = ‖d_j − m‖ 에 대해 r_j < 3 × 1.4826 × MAD + 0.5 px 를 만족하는 점만 유지 (최대 5회 반복)"),
    (1, "KLT 성공점 또는 MAD 통과점이 30개 미만인 프레임은 추적 실패로 처리. 실측 결과 실패 0프레임, inlier 비율 중앙값 0.99"),
]

S5 = [
    (0, "픽셀-mm 변환 — 초기 프레임 Jacobian"),
    (1, "시작 프레임 체커보드 호모그래피의 역변환 L_0^(−1) 을 영상 중심 c에서 미분하여 2×2 행렬 J_0 [격자칸/px] 를 구성"),
    (1, "해석적 전개 대신 c, c+(1,0), c+(0,1) 의 역호모그래피 상을 이용한 수치 미분 — 원근 효과와 축 간 결합을 동시에 반영"),
    (0, "누적 갱신식"),
    (1, "p_i = p_(i−1) − 10 · J_0 · Δu_i     (10 mm = 체커보드 한 칸)"),
    (1, "음부호의 근거:| 정지 장면의 영상 내 변위는 카메라 변위의 역상이므로 부호 반전을 요함"),
    (0, "축척 정책 2종"),
    (1, "fixed (주 결과):| J_0 를 t=0에 1회 결정 후 고정 — 실장비의 고정 마운트 전제를 직접 반영"),
    (1, "oracle (진단용):| 매 프레임 J_i 를 갱신 — 핸드헬드 축척·방위 변동이 유발한 오차 성분의 분리용 참고값"),
    (0, "프레임 단위 처리 순서"),
    (1, "ROI 가시면적 ≥ 5 % 확인 → 이전 특징점 ≥ 30개 확인 → 피라미드 KLT로 현 프레임에 대응"),
    (1, "MAD 이상점 제거 → 공통 이동 Δu_i 산출 → J_0 로 mm 변환 → 부호 반전 후 직전 위치에 누적"),
    (1, "현 ROI에서 특징점 최대 2,000개 재검출 — 점 소실 및 공간적 편중 방지"),
]

S6_TOP = [
    (0, "결론:| 집계 지표는 기준을 충족(V3 중앙값 0.1120 mm, V2 루프 오차 0.2277 mm)하나, 프레임 1 % 수준에서 최대 5.51 mm 단발 편차가 잔존함"),
    (1, "설계:| 프레임 간 체인 누적에서 매 프레임 격자 호모그래피 독립 적합으로 전환 — 체인 방식은 왕복 후 14~20 px(3~4 mm) 드리프트를 잔류시킴"),
    (1, "검출:| 좌 보드 1,206/1,206 프레임(코너 중앙값 163개, 격자적합 잔차 중앙값 0.496 px), 상 보드 846/1,206 프레임을 독립 적합한 2차 REF"),
]

S6_TABLE = [
    ["ID", "검증 항목", "값", "기준", "판정"],
    ["V1", "무이동 7.6 s 누적 드리프트", "0.0377 mm (σ 0.062 mm)", "< 0.5 mm", "PASS"],
    ["V2", "약 200 mm 왕복 후 원점 재통과(f1082) 잔차", "0.2277 mm", "< 2.0 mm", "PASS"],
    ["V3", "좌·상 보드 독립 측정 차이 (n = 846)", "med 0.1120 / p95 0.516 / max 4.27 mm", "med < 0.3 mm", "PASS"],
    ["V4", "mm/px 변동폭 (= 카메라 높이 변화)", "2.57 %", "< 3 %", "PASS"],
    ["V5", "정/역방향 증분 차이, 저속 구간 (n = 1117)", "med 0.185 / p99 3.83 / max 5.51 mm", "max < 0.5 mm", "FAIL"],
    ["V6", "프레임 간 최대 이동 (반 칸 미만이어야 배정 무모호)", "4.865 mm", "< 5 mm", "PASS"],
]

S6_BOTTOM = [
    "▪ V5 FAIL 기전:|  격자의 주기성으로 인해 코너-정수노드 배정이 추적 방향에 따라 한 칸 이동함. V3 최댓값 4.27 mm도 동일 기전임. 배정·적합의 반복 수렴을 구현하였으나 V2 0.23 → 1.06 mm, V5 5.51 → 12.74 mm로 전 궤적 지표가 악화하여 단일 패스로 복귀함",
    "▪ 해석 지침:|  REF 잡음 바닥은 중앙값 기준 ≈ 0.1 mm이므로 mm 단위 누적 궤적 비교는 성립함. 프레임 단위 증분 비교는 이 바닥에 가려 불가하며, 프레임별 순간 오차 인용 시 상기 단발 편차를 병기할 것",
]

S7_TOP = [
    (0, "주 결과 (fixed, 인쇄물 영역):| 최종 오차 6.36 mm (이동거리 773 mm 대비 0.82 %), RMS 2.23 mm, p95 6.09 mm, 최댓값 6.78 mm, 무이동 7.6 s 드리프트 0.017 mm"),
    (1, "추적 실패 0프레임, ROI 최소 가시면적 42 %, inlier 비율 중앙값 0.99"),
    (0, "오차의 축별 분포 — 총 이동의 92 %가 Y축 성분이며, Y축의 REF·EST 궤적 차이는 도시 축척에서 식별 한계 미만임. 잔여 8 %인 X축이 최종 오차의 전량을 흡수함"),
]

S7_TABLE = [
    ["성분", "최종 차이", "이동거리 대비"],
    ["fixed − REF  (주 결과)", "(−5.77, −2.68) → 6.36 mm", "0.82 %"],
    ["oracle − REF  (순수 추적 오차)", "(+3.34, +0.80) → 3.43 mm", "0.44 %"],
    ["fixed − oracle  (축척·방위 기여)", "(−9.11, −3.49) → 9.75 mm", "1.26 %"],
]

S7_BOTTOM = [
    "▪ 기전 — REF가 원인 변수를 직접 측정함:|  촬영 구간에서 카메라 롤이 +2.546° 변화(범위 −0.01°~+2.65°), 높이(축척)가 −2.81 %~+4.22 % 변동함. fixed 정책은 J_0 를 고정하므로 두 변동을 보상하지 못하며, 그 결과 세로 773 mm 변위가 가로축으로 사영됨",
    "▪ 대조 실험:|  동일 추정기를 체커보드 영역에 적용 시 최종 오차 8.13 mm(1.05 %), 정지 드리프트 0.103 mm로 인쇄물 영역(6.36 mm, 0.017 mm) 대비 오차가 증가함. 성능 제한 요인은 열화상 텍스처의 품질이 아니라 프레임 간 누적 자체임",
    "▪ 실장비 함의:|  고정 마운트 조건에서는 롤·높이 변동이 발생하지 않으므로 축척·방위 기여 성분(9.75 mm, 1.26 %)이 소거됨. 기대 성능은 순수 추적 오차 3.43 mm(0.44 %) 수준임",
]

S8 = [
    (0, "스윕 설계:| FOV 고정, 세로 해상도만 하향. 처리 순서는 다운샘플 → 가산 잡음 → CLAHE → KLT이며, σ의 단위는 8 bit 그레이 레벨임"),
    (0, "25개 조합 전부에서 추적 실패 0회. 누적 오차는 0.80~1.05 % 범위에 머물러 해상도 의존성을 나타내지 않음 — 축척·방위 항이 오차를 지배하며 이 항은 해상도와 무관하기 때문임 (해상도 순 비단조성도 동일 기전)"),
    (0, "해상도 의존성의 유일한 분해 지표는 정지 드리프트임:| 1080p 0.017 → 480p 0.047 → 240p 0.070 → 192p 0.083 → 120p 0.104 mm 로 단조 증가하며, σ 증가 방향으로도 일관됨"),
    (0, "증분 오차는 전 조합에서 419 µm로 고정| — 추정기의 특성이 아니라 REF 자체의 프레임 간 잡음 바닥임. 본 영상은 증분 정밀도의 해상도 의존성을 분해하지 못함(측정 한계)"),
    (0, "모듈 선정 함의:| XY 이동 추정 단독 요구 시 256×192급으로 충분하며 160×120급에서도 성립 — 해상도는 본 과제의 병목이 아님. 단, 본 실험은 인쇄된 열화상 사진을 RGB 카메라로 촬영한 것이므로 마이크로볼로미터의 NETD 거동·FPN·셔터 보정 아티팩트를 재현하지 못함 (NETD ≈ σ/255 × ΔT)"),
]


# ---------------------------------------------------------------- build
prs = Presentation(str(SRC))
sl = prs.slides

# --- 3, 4, 5: 본문만
fill_body(body_ph(sl[2]).text_frame, S3)
fill_body(body_ph(sl[3]).text_frame, S4, no_bullet="all_l0")
fill_body(body_ph(sl[4]).text_frame, S5)

# --- 6: 리드 + 검증표 + 마무리
ph6 = body_ph(sl[5])
place(ph6, 0.24, 1.29, 12.85, 1.55)
fill_body(ph6.text_frame, S6_TOP, sizes=(17, 14))
add_table(sl[5], S6_TABLE, 0.24, 2.90, 12.85, 2.10,
          font_size=13, col_widths=[0.60, 4.35, 4.30, 1.85, 1.75])
add_textbox(sl[5], 0.24, 5.35, 12.85, 1.50, S6_BOTTOM, size=14)

# --- 7: 리드 + 성분 분해표 + 해석
ph7 = body_ph(sl[6])
place(ph7, 0.24, 1.29, 12.85, 1.75)
fill_body(ph7.text_frame, S7_TOP, sizes=(18, 15))
add_table(sl[6], S7_TABLE, 0.24, 3.10, 9.40, 1.30,
          font_size=13, col_widths=[3.90, 3.40, 2.10])
add_textbox(sl[6], 0.24, 4.65, 12.85, 2.30, S7_BOTTOM, size=14)

# --- 8: 히트맵 + 해석 (슬라이드 2와 같은 '그림 + 텍스트' 구성)
fig_w = 9.10
fig_h = fig_w * 644 / 1820
sl[7].shapes.add_picture(str(HEATMAP), Inches((13.3333 - fig_w) / 2), Inches(1.22),
                         Inches(fig_w), Inches(fig_h))
ph8 = body_ph(sl[7])
place(ph8, 0.24, 1.22 + fig_h + 0.12, 12.85, 2.60)
fill_body(ph8.text_frame, S8, sizes=(12, 11), space_before=3)

prs.save(str(DST))
print("saved:", DST)
print("fig height (in):", round(fig_h, 2), "| text top:", round(1.35 + fig_h + 0.15, 2))
