"""Xuat baocao/Bao_cao_Cai_tien_pipeline.docx tu cac muc 3-7 cua
baocao/Bao_cao_Phan_5_Cai_tien_pipeline.md (danh so lai thanh 1-5), dung chung kieu
trinh bay voi cac bao cao truoc (scripts/build_preprocessing_docx.py).

Chay tu thu muc goc: python scripts/build_cai_tien_docx.py
"""
import re
import sys
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Pt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_preprocessing_docx as base  # noqa: E402

ROOT = base.ROOT
SOURCE = ROOT / "baocao" / "Bao_cao_Phan_5_Cai_tien_pipeline.md"
OUTPUT = ROOT / "baocao" / "Bao_cao_Cai_tien_pipeline.docx"

SECTION_MAP = {3: 1, 4: 2, 5: 3, 6: 4, 7: 5}
TABLE_MAP = {2: 1, 3: 2, 4: 3, 5: 4, 7: 5}

TITLE = "# BÁO CÁO CẢI TIẾN PIPELINE PHÁT HIỆN LÒNG ĐƯỜNG"
SUBTITLE = ("*Phân tích nguyên nhân, chuẩn hóa tương phản và ngưỡng Canny thích nghi, "
            "phân vùng bằng GrabCut, Canny + Hough trên viền vùng đường, kết quả*")

# Hai muc dau (tom tat, hien trang) bi luoc bo nen giai thich ngan cac ten goi ngay cho dung dau tien.
CLARIFY = {
    "Các con số trong mục này đo trên đầu ra của notebook.":
        "Các con số trong mục này đo trên đầu ra của notebook `Pipeline_HSV_Canny_Hough` "
        "(mặt nạ màu S-V → Canny → Hough, IoU 0,559).",
    "tức là trùng với baseline tầm thường.":
        "tức là trùng với baseline tầm thường (tô kín cả dải từ đường chân trời tới mép capo, IoU 0,558).",
}


def extract_sections(markdown: str) -> str:
    start = markdown.index("## 3.")
    end = markdown.index("## 8.")
    body = markdown[start:end].rstrip() + "\n"
    body = re.sub(r"^(#{2,3}) (\d)(\.\d)?",
                  lambda m: f"{m.group(1)} {SECTION_MAP[int(m.group(2))]}{m.group(3) or ''}", body, flags=re.M)
    body = re.sub(r"\bmục (\d)\b", lambda m: f"mục {SECTION_MAP.get(int(m.group(1)), int(m.group(1)))}", body)
    body = re.sub(r"\bBảng (\d)\b", lambda m: f"Bảng {TABLE_MAP[int(m.group(1))]}", body)
    for old, new in CLARIFY.items():
        if old not in body:
            raise ValueError(f"Khong tim thay doan can lam ro: {old}")
        body = body.replace(old, new, 1)
    # Bo tach bang cua build_preprocessing_docx khong hieu "\\|" (tri tuyet doi) trong o bang.
    body = body.replace("\\|", "\u2223")
    return f"{TITLE}\n\n{SUBTITLE}\n\n{body}"


# Thu tu phan tu con cua w:tcPr theo schema OOXML; set_cell_margins cua script cu chen tcMar sau vAlign.
TC_PR_ORDER = ["cnfStyle", "tcW", "gridSpan", "hMerge", "vMerge", "tcBorders", "shd", "noWrap", "tcMar",
               "textDirection", "tcFitText", "vAlign", "hideMark"]


def fix_schema(document) -> None:
    for tc_pr in document.element.body.iter(qn("w:tcPr")):
        children = list(tc_pr)
        rank = {qn(f"w:{name}"): i for i, name in enumerate(TC_PR_ORDER)}
        children.sort(key=lambda el: rank.get(el.tag, len(rank)))
        for child in children:
            tc_pr.remove(child)
            tc_pr.append(child)
    for shd in document.element.body.iter(qn("w:shd")):
        if shd.get(qn("w:val")) is None:
            shd.set(qn("w:val"), "clear")
    zoom = document.settings.element.find(qn("w:zoom"))
    if zoom is not None and zoom.get(qn("w:percent")) is None:
        zoom.set(qn("w:percent"), "100")


def build() -> Path:
    markdown = extract_sections(SOURCE.read_text(encoding="utf-8"))
    document = Document()
    base.configure_document(document)
    for run in document.sections[0].header.paragraphs[0].runs:
        run.text = "CẢI TIẾN PIPELINE · HSV → CANNY → HOUGH"
    base.parse_markdown(document, markdown)
    # Bang nhieu cot (ket qua, ablation): chu nho hon va le o hep de vua kho A4 doc
    for table in document.tables:
        columns = len(table.columns)
        if columns < 7:
            continue
        size = Pt(8.5) if columns >= 10 else Pt(9.5)
        for row in table.rows:
            for cell in row.cells:
                base.set_cell_margins(cell, top=50, start=50, bottom=50, end=50)
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.font.size = size
    fix_schema(document)
    props = document.core_properties
    props.title = "Báo cáo cải tiến pipeline phát hiện lòng đường"
    props.subject = "Chuẩn hóa tương phản, Canny thích nghi, GrabCut, Canny + Hough"
    props.author = ""
    props.keywords = "road detection, Canny, Hough, GrabCut, CLAHE, low contrast"
    document.save(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print(build())
