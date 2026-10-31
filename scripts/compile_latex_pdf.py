"""Render Hough report sources to PDF using installed Microsoft Word.

Only Module 4 reports are rebuilt. Previous modules and their outputs are untouched.
DOCX/PDF use the same Markdown sources; no independent hard-coded HTML report.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'


def main():
    for script in ['build_hough_report.py', 'build_hough_docx.py', 'build_hough_tom_tat.py']:
        subprocess.run([sys.executable, str(SCRIPTS / script)], cwd=ROOT, check=True)
    targets = [
        ('baocao/Bao_cao_Phan_4_Hough.docx', 'baocao/Bao_cao_Phan_4_Hough.pdf'),
        ('baocao/hough_road_boundary_tom_tat.docx', 'hough_road_boundary_tom_tat.pdf'),
    ]
    for source, destination in targets:
        subprocess.run([
            'powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass',
            '-File', str(SCRIPTS / 'render_docx_with_word.ps1'),
            '-InputDocx', str(ROOT / source), '-OutputPdf', str(ROOT / destination),
        ], cwd=ROOT, check=True)
    # Update both report PDFs that already existed in the project.
    shutil.copyfile(ROOT / 'baocao/Bao_cao_Phan_4_Hough.pdf', ROOT / 'baocao/Bao_cao_Phan_4_Hough_Hau_xu_ly.pdf')
    print('Updated Hough reports only; previous module reports and inputs are unchanged.')


if __name__ == '__main__':
    main()
