"""Xuat bao cao .docx tu file .md, dung chung kieu trinh bay voi cac bao cao truoc
(scripts/build_preprocessing_docx.py).

Chay tu thu muc goc:
  python scripts/build_tom_tat_3_buoc_docx.py           # baocao/Bao_cao_Tom_tat_Tien_xu_ly_Canny_Hough.docx
  python scripts/build_tom_tat_3_buoc_docx.py khung     # baocao/Bao_cao_Khung_Abstract_den_3.1.docx
(chay truoc scripts/figures_tom_tat_3_buoc.py hoac scripts/figures_khung_bao_cao.py de co hinh)
"""
import sys
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from docx.text.paragraph import Paragraph

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_cai_tien_docx as cai_tien  # noqa: E402
import build_preprocessing_docx as base  # noqa: E402

ROOT = base.ROOT
REPORTS = {
    "tom_tat": dict(
        name="Bao_cao_Tom_tat_Tien_xu_ly_Canny_Hough",
        header="TÓM TẮT · TIỀN XỬ LÝ → CANNY → HOUGH · TRƯỚC VÀ SAU",
        title="Tóm tắt tiền xử lý, Canny, Hough: trước và sau cải tiến",
        subject="So sánh pipeline gốc (Phần 2-4) và pipeline v3",
    ),
    "khung": dict(
        name="Bao_cao_Khung_Abstract_den_3.1",
        header="ROAD BOUNDARY DETECTION · KHUNG BÁO CÁO",
        title="Phát hiện lòng đường và biên đường bằng xử lý ảnh cổ điển",
        subject="Abstract, Introduction, Fundamentals, 3.1 Tiền xử lý, Conclusion, References",
    ),
}


# Thu tu phan tu con cua w:pPr theo schema; set_cell_like_shading cua script goc chen w:shd vao cuoi
P_PR_ORDER = ["pStyle", "keepNext", "keepLines", "pageBreakBefore", "framePr", "widowControl", "numPr",
              "suppressLineNumbers", "pBdr", "shd", "tabs", "suppressAutoHyphens", "kinsoku", "wordWrap",
              "overflowPunct", "topLinePunct", "autoSpaceDE", "autoSpaceDN", "bidi", "adjustRightInd",
              "snapToGrid", "spacing", "ind", "contextualSpacing", "mirrorIndents", "suppressOverlap", "jc",
              "textDirection", "textAlignment", "textboxTightWrap", "outlineLvl", "divId", "cnfStyle", "rPr",
              "sectPr", "pPrChange"]


def _sort_p_pr(document) -> None:
    rank = {qn(f"w:{name}"): i for i, name in enumerate(P_PR_ORDER)}
    for p_pr in document.element.body.iter(qn("w:pPr")):
        children = sorted(p_pr, key=lambda el: rank.get(el.tag, len(rank)))
        for child in children:
            p_pr.remove(child)
            p_pr.append(child)


def _keep_next(p) -> None:
    # Qua API cua python-docx de w:keepNext nam dung thu tu schema trong w:pPr
    Paragraph(p, None).paragraph_format.keep_with_next = True


def build(key: str = "tom_tat") -> Path:
    spec = REPORTS[key]
    source = ROOT / "baocao" / f"{spec['name']}.md"
    output = ROOT / "baocao" / f"{spec['name']}.docx"
    document = Document()
    base.configure_document(document)
    for run in document.sections[0].header.paragraphs[0].runs:
        run.text = spec["header"]
    base.parse_markdown(document, source.read_text(encoding="utf-8"))
    for table in document.tables:
        if len(table.columns) < 5:
            continue
        for row in table.rows:
            for cell in row.cells:
                base.set_cell_margins(cell, top=60, start=70, bottom=60, end=70)
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.font.size = Pt(9.5)
    # Giu hinh cung trang voi chu thich; hinh cao (vd. luoi 4 hang) thu nho de vua mot trang cung chu thich
    for paragraph in document.paragraphs:
        if paragraph._p.findall(".//" + qn("w:drawing")):
            paragraph.paragraph_format.keep_with_next = True
    # Bang di cung chu thich: keep_with_next cho moi dong cua bang va doan trong ngay sau bang
    body = list(document.element.body)
    for i, element in enumerate(body):
        if element.tag != qn("w:tbl"):
            continue
        for p in element.iter(qn("w:p")):
            _keep_next(p)
        if i + 1 < len(body) and body[i + 1].tag == qn("w:p") and not "".join(body[i + 1].itertext()).strip():
            _keep_next(body[i + 1])
    for shape in document.inline_shapes:
        if shape.width > Cm(14) and shape.height > Cm(14):
            ratio = Cm(12) / shape.width
            shape.width, shape.height = Cm(12), int(shape.height * ratio)
    cai_tien.fix_schema(document)
    _sort_p_pr(document)
    props = document.core_properties
    props.title = spec["title"]
    props.subject = spec["subject"]
    props.author = ""
    props.keywords = "road detection, preprocessing, CLAHE, Canny, Hough, GrabCut"
    document.save(output)
    return output


if __name__ == "__main__":
    print(build(sys.argv[1] if len(sys.argv) > 1 else "tom_tat"))
