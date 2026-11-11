"""Xuat baocao/Bao_cao_Tom_tat_Tien_xu_ly_Canny_Hough.docx tu file .md cung ten,
dung chung kieu trinh bay voi cac bao cao truoc (scripts/build_preprocessing_docx.py).

Chay tu thu muc goc: python scripts/build_tom_tat_3_buoc_docx.py
(can chay scripts/figures_tom_tat_3_buoc.py truoc de co Hinh 1, 2)
"""
import sys
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_cai_tien_docx as cai_tien  # noqa: E402
import build_preprocessing_docx as base  # noqa: E402

ROOT = base.ROOT
SOURCE = ROOT / "baocao" / "Bao_cao_Tom_tat_Tien_xu_ly_Canny_Hough.md"
OUTPUT = ROOT / "baocao" / "Bao_cao_Tom_tat_Tien_xu_ly_Canny_Hough.docx"


def build() -> Path:
    document = Document()
    base.configure_document(document)
    for run in document.sections[0].header.paragraphs[0].runs:
        run.text = "TÓM TẮT · TIỀN XỬ LÝ → CANNY → HOUGH · TRƯỚC VÀ SAU"
    base.parse_markdown(document, SOURCE.read_text(encoding="utf-8"))
    for table in document.tables:
        if len(table.columns) < 5:
            continue
        for row in table.rows:
            for cell in row.cells:
                base.set_cell_margins(cell, top=60, start=70, bottom=60, end=70)
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.font.size = Pt(9.5)
    # Giu hinh cung trang voi chu thich; Hinh 1 (4 hang) thu nho de vua mot trang cung chu thich
    for paragraph in document.paragraphs:
        shapes = paragraph._p.findall(".//" + qn("w:drawing"))
        if not shapes:
            continue
        paragraph.paragraph_format.keep_with_next = True
    for shape in document.inline_shapes:
        if shape.width > Cm(14) and shape.height > Cm(14):
            ratio = Cm(12) / shape.width
            shape.width, shape.height = Cm(12), int(shape.height * ratio)
    cai_tien.fix_schema(document)
    props = document.core_properties
    props.title = "Tóm tắt tiền xử lý, Canny, Hough: trước và sau cải tiến"
    props.subject = "So sánh pipeline gốc (Phần 2-4) và pipeline v3"
    props.author = ""
    props.keywords = "road detection, preprocessing, Canny, Hough, GrabCut"
    document.save(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print(build())
