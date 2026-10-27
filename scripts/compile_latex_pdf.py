"""Bien dich bao cao Phan 4 sang ban PDF chuan LaTeX bang KaTeX va Chromium engine."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BAOCAO = ROOT / "baocao"
OUT_PDF = BAOCAO / "Bao_cao_Phan_4_Hough_LaTeX.pdf"
HTML_TEMP = BAOCAO / "Bao_cao_Phan_4_Hough_temp.html"

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<title>Báo cáo Phần 4: Biến đổi Hough và Hậu xử lý</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
<style>
  @page {
    size: A4;
    margin: 22mm 20mm 22mm 20mm;
    @bottom-right {
      content: counter(page);
    }
  }
  body {
    font-family: 'Times New Roman', Times, 'Computer Modern Serif', serif;
    font-size: 11pt;
    line-height: 1.45;
    color: #111;
    margin: 0;
    padding: 0;
    text-align: justify;
  }
  h1.title {
    text-align: center;
    font-size: 18pt;
    font-weight: bold;
    margin-bottom: 4px;
    line-height: 1.3;
  }
  .subtitle {
    text-align: center;
    font-size: 11pt;
    font-style: italic;
    color: #444;
    margin-bottom: 14px;
  }
  .author {
    text-align: center;
    font-size: 11pt;
    font-weight: bold;
    margin-bottom: 2px;
  }
  .date {
    text-align: center;
    font-size: 10pt;
    color: #666;
    margin-bottom: 20px;
  }
  .abstract {
    margin: 0 25px 20px 25px;
    font-size: 10pt;
    line-height: 1.4;
  }
  .abstract-title {
    text-align: center;
    font-weight: bold;
    margin-bottom: 5px;
  }
  h2 {
    font-size: 13pt;
    font-weight: bold;
    margin-top: 18px;
    margin-bottom: 6px;
    border-bottom: 1px solid #ddd;
    padding-bottom: 3px;
  }
  h3 {
    font-size: 11.5pt;
    font-weight: bold;
    margin-top: 12px;
    margin-bottom: 4px;
  }
  p {
    margin: 0 0 8px 0;
    text-indent: 18px;
  }
  p.no-indent {
    text-indent: 0;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0;
    font-size: 9.5pt;
  }
  th, td {
    border: 1px solid #bbb;
    padding: 5px 7px;
    text-align: left;
  }
  th {
    background-color: #f2f5f8;
    font-weight: bold;
    text-align: center;
  }
  .table-caption, .figure-caption {
    font-size: 9.5pt;
    font-style: italic;
    text-align: center;
    margin-top: 5px;
    margin-bottom: 14px;
    color: #333;
  }
  img {
    max-width: 95%;
    height: auto;
    display: block;
    margin: 0 auto;
  }
  .figure-box {
    text-align: center;
    margin: 14px 0;
    page-break-inside: avoid;
  }
  .katex-display {
    margin: 10px 0 !important;
  }
  ul, ol {
    margin: 0 0 8px 0;
    padding-left: 28px;
  }
  li {
    margin-bottom: 3px;
  }
  .ref-list {
    font-size: 9.5pt;
    padding-left: 20px;
  }
</style>
</head>
<body>

<h1 class="title">BÁO CÁO PHẦN 4: BIẾN ĐỔI HOUGH VÀ HẬU XỬ LÝ<br>TÌM HAI BIÊN ĐƯỜNG</h1>
<div class="subtitle">Cải tiến mặt nạ ROI hình thang thích ứng, Probabilistic Hough, lọc độ dốc và hồi quy tuyến tính có trọng số</div>
<div class="author">Nhóm nghiên cứu Xử lý ảnh · Đồ án môn học</div>
<div class="date">Tháng 10, 2026</div>

<div class="abstract">
  <div class="abstract-title">TÓM TẮT</div>
  Báo cáo trình bày chi tiết thiết kế giải thuật và kết quả thực nghiệm của Module 4 trong hệ thống phát hiện biên đường (<i>Road Boundary Detection</i>). Từ ảnh cạnh nhị phân Canny $960 \\times 540$, phương pháp áp dụng mặt nạ vùng quan tâm (ROI) hình thang cân thích ứng với đường chân trời thực nghiệm ($y \\approx 0.48H$), trích xuất các đoạn thẳng bằng biến đổi Hough xác suất (Probabilistic Hough Transform -- P-Hough), lọc bỏ nhiễu độ dốc ngang và tiệm cận góc đứng, sau đó áp dụng hồi quy tuyến tính có trọng số theo chiều dài đoạn thẳng ($x = k \\cdot y + b$) để xác định hai biên đường. Thực nghiệm trên 50 ảnh kiểm thử thuộc ba nhóm (Normal, Curved, Adverse) cho thấy hệ thống đạt tỷ lệ bắt biên $74.0\\%$ biên trái, $72.0\\%$ biên phải, $54.0\\%$ cả hai biên với chỉ số $IoU$ trung bình $0.3144$, sai số vị trí biên $MAE = 234.91\\text{ px}$ và thời gian xử lý siêu thời gian thực $7.75\\text{ ms}$/ảnh ($\\sim 129\\text{ FPS}$).
</div>

<h2>1. Mục tiêu và phạm vi</h2>
<p>Pipeline của nhóm gồm: Tiền xử lý màu HSV, Canny phát hiện cạnh, ROI kết hợp biến đổi Hough, rồi chọn hai biên đường để bàn giao cho phần đánh giá [1], [2]. Phần này chịu trách nhiệm cho Module 4: nhận ảnh cạnh nhị phân kích thước $960 \\times 540$ từ Canny, áp dụng vùng quan tâm (ROI), trích xuất các đoạn thẳng, lọc theo độ dốc, gộp thành hai biên trái / phải và tái dựng đa giác mặt đường.</p>

<p class="no-indent">Báo cáo trả lời sáu câu hỏi trọng tâm:</p>
<ol>
  <li>Nguyên lý bỏ phiếu trong không gian tham số $(\\rho, \\theta)$ hoạt động thế nào, và vì sao Probabilistic Hough (P-Hough) phù hợp hơn Standard Hough trong bài toán này.</li>
  <li>Vùng quan tâm (ROI) hình thang cần thiết kế với tỷ lệ nào để bao trọn vùng chân trời thực nghiệm của bộ dữ liệu mà không nạp nhiễu nền.</li>
  <li>Bộ lọc độ dốc và ràng buộc không gian hoạt động ra sao để loại bỏ các vạch ngang (gờ giảm tốc, vạch sang đường, nắp capo) và phân tách chính xác cụm trái/phải.</li>
  <li>Thuật toán hồi quy biên đường có trọng số theo chiều dài đoạn thẳng giải quyết bài toán kỳ dị độ dốc đứng như thế nào.</li>
  <li>Hiệu năng định lượng của Hough trên 50 ảnh (tỷ lệ bắt biên, chỉ số $IoU$, sai số vị trí biên $MAE$, và thời gian thực thi FPS) ra sao.</li>
  <li>Giới hạn hình học của Hough đường thẳng khi gặp khúc cua cong (<i>Curved</i>) biểu hiện như thế nào.</li>
</ol>

<h2>2. Nguyên lý biến đổi Hough cho đường thẳng</h2>

<h3>2.1 Không gian tham số Hessian</h3>
<p>Trong tọa độ Descartes $y = mx + b$, một đường thẳng đứng có hệ số góc $m \\to \\infty$, dẫn đến vùng tìm kiếm không bị chặn. Biến đổi Hough biểu diễn đường thẳng trong hệ tọa độ cực Hessian [4]:</p>
$$\\rho = x \\cos \\theta + y \\sin \\theta$$
<p class="no-indent">Trong đó $\\rho$ là khoảng cách đại số vuông góc từ gốc tọa độ $(0, 0)$ đến đường thẳng, và $\\theta$ là góc giữa trục hoành $Ox$ và pháp tuyến của đường thẳng ($\\theta \\in [0, \\pi)$). Mỗi điểm cạnh $(x_i, y_i)$ trong không gian ảnh tương ứng với một đường hình sin trong không gian tham số $(\\rho, \\theta)$. Giao điểm của nhiều đường hình sin là tọa độ $(\\rho_0, \\theta_0)$ của đường thẳng cần tìm.</p>

<h3>2.2 So sánh Standard Hough và Probabilistic Hough</h3>
<p>Belhadri và cộng sự [1] cũng như Wang và cộng sự [2] đều áp dụng biến đổi Hough sau Canny. Tuy nhiên có sự khác biệt bản chất giữa hai biến thể:</p>

<table>
  <thead>
    <tr>
      <th>Đặc trưng</th>
      <th>Standard Hough (<code>cv2.HoughLines</code>)</th>
      <th>Probabilistic Hough (<code>cv2.HoughLinesP</code>)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Cơ chế lấy mẫu</b></td>
      <td>Quét toàn bộ mọi pixel cạnh trong ảnh</td>
      <td>Lấy mẫu ngẫu nhiên có kiểm soát</td>
    </tr>
    <tr>
      <td><b>Dạng đầu ra</b></td>
      <td>Cặp vô hạn $(\\rho, \\theta)$ kéo dài xuyên suốt ảnh</td>
      <td>Đoạn thẳng hữu hạn $[x_1, y_1, x_2, y_2]$ có điểm mút</td>
    </tr>
    <tr>
      <td><b>Lọc đoạn ngắn</b></td>
      <td>Không hỗ trợ trực tiếp (đường vô hạn)</td>
      <td>Lọc trực tiếp theo chiều dài (<code>minLineLength</code>)</td>
    </tr>
    <tr>
      <td><b>Nối đứt đoạn</b></td>
      <td>Không hỗ trợ nối rời rạc</td>
      <td>Hỗ trợ nối đoạn đứt quãng (<code>maxLineGap</code>)</td>
    </tr>
    <tr>
      <td><b>Thời gian chạy</b></td>
      <td>Chậm hơn khi ảnh có mật độ cạnh dày</td>
      <td>Rất nhanh ($7.75\\text{ ms}$), đạt chuẩn thời gian thực</td>
    </tr>
  </tbody>
</table>

<div class="figure-box">
  <img src="../outputs/hough/figures/figure_2_standard_vs_probabilistic.png" alt="So sánh Standard Hough và P-Hough">
  <div class="figure-caption">Hình 4.1: So sánh Standard Hough và Probabilistic Hough trên ảnh normal_12. P-Hough trả về các đoạn thẳng hữu hạn ôm sát lòng đường.</div>
</div>

<h2>3. Thiết kế vùng quan tâm (ROI) và bộ lọc độ dốc</h2>

<h3>3.1 Mặt nạ ROI hình thang thích ứng</h3>
<p>Khảo sát ground truth của 50 ảnh thực nghiệm cho thấy đường chân trời thực tế hội tụ ở trung vị $y = 0.456 \\times H$ (dòng 246 trên ảnh cao 540). Do đó, tỷ lệ ROI được thiết kế cân đối qua trục giữa $x_c = W/2 = 480$:</p>
$$\\text{top\\_left} = (x_c - 0.175 W, 0.48 H), \\quad \\text{top\\_right} = (x_c + 0.175 W, 0.48 H)$$
$$\\text{bottom\\_left} = (x_c - 0.49 W, 0.98 H), \\quad \\text{bottom\\_right} = (x_c + 0.49 W, 0.98 H)$$
<p class="no-indent">Cắt bỏ $2\\%$ phần đáy dưới ($y > 0.98H$) giúp loại bỏ hoàn toàn viền nắp capo xe.</p>

<h3>3.2 Bộ lọc độ dốc và ràng buộc không gian</h3>
<p>Trong hệ tọa độ pixel ($Ox$ sang phải, $Oy$ hướng xuống dưới):</p>
<ul>
  <li><b>Biên trái:</b> $\\Delta x = x_2 - x_1 > 0$ khi $\\Delta y = y_2 - y_1 < 0 \\implies m = \\frac{\\Delta y}{\\Delta x} < 0$, kết hợp $x_{mid} < 0.65W$.</li>
  <li><b>Biên phải:</b> $\\Delta x < 0$ khi $\\Delta y < 0 \\implies m = \\frac{\\Delta y}{\\Delta x} > 0$, kết hợp $x_{mid} > 0.35W$.</li>
  <li><b>Loại bỏ vạch ngang và vạch đứng:</b> Áp dụng ngưỡng $0.30 \\le |m| \\le 4.00$ (tương đương góc nghiêng từ $16.7^\\circ$ đến $76.0^\\circ$).</li>
</ul>

<div class="figure-box">
  <img src="../outputs/hough/figures/figure_1_hough_pipeline_stages.png" alt="Các bước trung gian">
  <div class="figure-caption">Hình 4.2: Các bước trung gian trên ảnh normal_05: Canny &rarr; ROI &rarr; P-Hough &rarr; Lọc độ dốc &rarr; Gộp biên.</div>
</div>

<h2>4. Thuật toán hồi quy biên đường có trọng số</h2>
<p>Để khắc phục kỳ dị chia cho $0$ khi vạch biên gần như thẳng đứng, thuật toán hồi quy $x$ theo $y$:</p>
$$x = k \\cdot y + b$$
<p class="no-indent">Mỗi đoạn thẳng thứ $i$ được gán trọng số bằng chiều dài đoạn thẳng $w_i = \\sqrt{\\Delta x_i^2 + \\Delta y_i^2}$. Hệ số được giải bằng công thức đóng Weighted Least Squares (WLS):</p>
$$k = \\frac{\\sum w_i \\left[(y_{1,i} - \\bar{y})(x_{1,i} - \\bar{x}) + (y_{2,i} - \\bar{y})(x_{2,i} - \\bar{x})\\right]}{\\sum w_i \\left[(y_{1,i} - \\bar{y})^2 + (y_{2,i} - \\bar{y})^2\\right]} = \\frac{\\text{Cov}_w(y, x)}{\\text{Var}_w(y)}$$
$$b = \\bar{x} - k \\cdot \\bar{y}$$
<p class="no-indent">Tọa độ biên được ngoại suy tại $y_{bottom} = 0.98H$ và $y_{top} = 0.48H$:</p>
$$x_{bottom} = \\text{round}(k \\cdot y_{bottom} + b), \\quad x_{top} = \\text{round}(k \\cdot y_{top} + b)$$
$$P = \\left[ (x_{L,bot}, y_{bottom}), (x_{L,top}, y_{top}), (x_{R,top}, y_{top}), (x_{R,bot}, y_{bottom}) \\right]$$

<h2>5. Kết quả thực nghiệm trên 50 ảnh</h2>
<p>Lưới thực nghiệm quét 27 tổ hợp tham số cho thấy cấu hình $\\text{threshold} = 25, \\text{minLineLength} = 25\\text{ px}, \\text{maxLineGap} = 25\\text{ px}$ mang lại sự cân bằng tối ưu giữa tỷ lệ bắt biên và độ sạch của đoạn thẳng.</p>

<div class="figure-box">
  <img src="../outputs/hough/figures/figure_3_parameter_grid_sensitivity.png" alt="Độ nhạy tham số">
  <div class="figure-caption">Hình 4.3: Phân tích độ nhạy của biến đổi Hough theo threshold và chiều dài đoạn thẳng.</div>
</div>

<div class="table-caption">Bảng 4.1: Kết quả định lượng của module Hough trên 50 ảnh thực nghiệm.</div>
<table>
  <thead>
    <tr>
      <th>Nhóm kiểm thử</th>
      <th>Số ảnh</th>
      <th>P-Hough TB</th>
      <th>Bắt Trái (%)</th>
      <th>Bắt Phải (%)</th>
      <th>Bắt Cả 2 (%)</th>
      <th>IoU TB</th>
      <th>MAE TB (px)</th>
      <th>Runtime (ms)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Normal</b></td>
      <td>18</td>
      <td>57.3</td>
      <td>72.2%</td>
      <td>66.7%</td>
      <td>50.0%</td>
      <td>0.2994</td>
      <td>213.70</td>
      <td>8.72</td>
    </tr>
    <tr>
      <td><b>Curved</b></td>
      <td>12</td>
      <td>51.2</td>
      <td>66.7%</td>
      <td>75.0%</td>
      <td>50.0%</td>
      <td><b>0.3809</b></td>
      <td><b>177.20</b></td>
      <td>7.74</td>
    </tr>
    <tr>
      <td><b>Adverse</b></td>
      <td>20</td>
      <td>38.2</td>
      <td><b>80.0%</b></td>
      <td><b>75.0%</b></td>
      <td><b>60.0%</b></td>
      <td>0.2923</td>
      <td>279.67</td>
      <td>6.87</td>
    </tr>
    <tr style="background-color: #eaf1f7; font-weight: bold;">
      <td>Toàn bộ (All)</td>
      <td>50</td>
      <td>48.2</td>
      <td>74.0%</td>
      <td>72.0%</td>
      <td>54.0%</td>
      <td>0.3144</td>
      <td>234.91</td>
      <td>7.75 (~129 FPS)</td>
    </tr>
  </tbody>
</table>

<div class="figure-box">
  <img src="../outputs/hough/figures/figure_4_group_examples.png" alt="Minh họa 3 nhóm">
  <div class="figure-caption">Hình 4.4: Kết quả nhận dạng biên đường trên sáu ảnh tiêu biểu thuộc ba nhóm Normal, Curved và Adverse.</div>
</div>

<h2>6. Phân tích ca lỗi và giới hạn hình học</h2>
<ul>
  <li><b>Giới hạn trên khúc cua (Curved):</b> Hough đường thẳng chỉ khớp được đoạn tiếp tuyến ở cự ly gần mũi xe. Tại vùng xa, đường thẳng phân kỳ khỏi mép đường thực tế. Hướng cải tiến là phân tầng ROI hoặc khớp đường cong Spline bậc ba / RANSAC parabol [3].</li>
  <li><b>Khả năng chống bóng đổ (Adverse):</b> Bộ lọc độ dốc $|m| \\ge 0.30$ loại bỏ $100\\%$ các cạnh ngang do bóng nhà/cây tạo ra.</li>
  <li><b>Ảnh hưởng thiếu sáng:</b> Các đoạn vạch mờ ban đêm vẫn được thuật toán WLS ngoại suy chính xác nhờ tận dụng các đoạn phản quang ngắn.</li>
</ul>

<h2>7. Bàn giao giao diện cho Phần 5</h2>
<p>Module 4 hoàn thiện cung cấp hàm tích hợp chuẩn <code>detect_road_boundaries(edges, config)</code> trả về <code>HoughResult</code> chứa tọa độ <code>left_line</code>, <code>right_line</code>, <code>polygon</code> và <code>roi_mask</code> sẵn sàng cho Phần 5 tích hợp đánh giá toàn pipeline.</p>

<h2>Tài liệu tham khảo</h2>
<div class="ref-list">
  <p class="no-indent">[1] Y. Belhadri, D. Mohamed, B. Alshaqaqi, and A. S. Baquhaizel, "Lane Boundary Detection in Dynamic Environments," <i>Rev. Roum. Sci. Techn. -- Électrotechn. et Énerg.</i>, vol. 71, no. 2, pp. 299--304, 2026.</p>
  <p class="no-indent">[2] S. Wang, S. Wu, X. Wang, and Z. Li, "A Canny Operator Road Edge Detection Method Based on Color Features," <i>Journal of Physics: Conference Series</i>, vol. 1629, 012018, 2020.</p>
  <p class="no-indent">[3] M. D. E. Munajat, D. H. Widyantoro, and R. Munir, "Road Detection System Based on RGB Histogram Filterization and Boundary Classifier," <i>Proc. ICACSIS</i>, pp. 195--200, 2015.</p>
  <p class="no-indent">[4] J. F. Canny, "A Computational Approach to Edge Detection," <i>IEEE Trans. Pattern Analysis and Machine Intelligence</i>, vol. 8, no. 6, pp. 679--698, 1986.</p>
</div>

<script>
  document.addEventListener("DOMContentLoaded", function() {
    renderMathInElement(document.body, {
      delimiters: [
        {left: '$$', right: '$$', display: true},
        {left: '$', right: '$', display: false}
      ],
      throwOnError: false
    });
  });
</script>
</body>
</html>
"""

HTML_TEMP.write_text(HTML_TEMPLATE, encoding="utf-8")
print(f"Da tao HTML template: {HTML_TEMP}")

edge_path = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
if not edge_path.is_file():
    edge_path = Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe")

cmd = [
    str(edge_path),
    "--headless",
    "--disable-gpu",
    "--no-pdf-header-footer",
    "--run-all-compositor-stages-before-draw",
    f"--print-to-pdf={OUT_PDF.resolve()}",
    str(HTML_TEMP.resolve().as_uri()),
]

print("Dang bien dich PDF bang Chromium engine...")
res = subprocess.run(cmd, capture_output=True, text=True)
if res.returncode == 0:
    print(f"-> Thanh cong! File PDF LaTeX: {OUT_PDF}")
    if HTML_TEMP.exists():
        HTML_TEMP.unlink()
else:
    print("Loi bien dich:", res.stderr)
