# Phần Hough trong phát hiện biên đường

*Tài liệu đọc cho người phụ trách Hough: cần học gì, cần làm gì, dùng được gì từ ba bài báo*
*Nguồn: Belhadri et al. 2026, Wang et al. 2020, Munajat et al. 2015.*

## 1. Bức tranh tổng quát

Bài toán là tìm hai biên của mặt đường (trái và phải) từ ảnh cạnh nhị phân Canny bằng xử lý ảnh cổ điển. Pipeline tổng thể:

```
Ảnh gốc -> Tiền xử lý HSV -> Canny (0/255) -> ROI + HOUGH (Phần của bạn) -> Hai biên đường trái / phải -> Đo IoU & MAE
```

**Vai trò của bạn:** Nhận ảnh cạnh 960 × 540 từ người số 3 (Canny), áp dụng mặt nạ ROI hình thang, dùng biến đổi Hough trích xuất các đoạn thẳng, lọc độ dốc, gộp thành hai biên trái/phải đại diện và xuất đa giác mặt đường để người số 5 tính metric. Bạn cũng là người trả lời câu hỏi: *"Vì sao dùng P-Hough thay vì Standard Hough, ROI đặt thế nào, và xử lý khúc cua cong ra sao?"*

## 2. Bốn bước Hough cần nắm

| Bước | Tên bước | Nội dung, công thức và tham số cốt lõi |
|---|---|---|
| **1** | **Cắt ROI hình thang** | Áp dụng mặt nạ đa giác cân đối: đỉnh trên tại $y = 0.48H$ (đường chân trời thực nghiệm), đáy dưới tại $y = 0.98H$ (loại bỏ nắp capo). Bề rộng trên $0.35W$, bề rộng dưới $0.98W$. Cắt bằng `cv2.fillPoly` và `cv2.bitwise_and`. |
| **2** | **P-Hough trích xuất đoạn thẳng** | Dùng `cv2.HoughLinesP`. Lấy mẫu ngẫu nhiên có kiểm soát trong không gian $(\rho, \theta)$. Trả về các đoạn thẳng hữu hạn $[x_1, y_1, x_2, y_2]$. Khuyến nghị: $\text{threshold} = 25$, $\text{minLineLength} = 25\text{ px}$, $\text{maxLineGap} = 25\text{ px}$. |
| **3** | **Lọc độ dốc & phân cụm** | Hệ tọa độ ảnh ($y$ hướng xuống):<br>• Biên trái: $m = \Delta y / \Delta x < 0$ và $x_{mid} < 0.65W$<br>• Biên phải: $m > 0$ và $x_{mid} > 0.35W$<br>• Loại bỏ vạch ngang: độ dốc tuyệt đối < 0.30 ($< 16.7^\circ$) và vạch tiệm cận đứng: độ dốc tuyệt đối > 4.00 ($> 76.0^\circ$). |
| **4** | **Hồi quy biên có trọng số** | Khắc phục kỳ dị độ dốc đứng bằng mô hình $x = k \cdot y + b$. Dùng Weighted Least Squares với trọng số là độ dài đoạn thẳng $w_i = \sqrt{\Delta x^2 + \Delta y^2}$. Ngoại suy $x$ tại $y_{bottom}$ và $y_{top}$, nối thành đa giác 4 đỉnh. |

## 3. Điểm dùng được từ ba bài báo cho phần Hough

| Nội dung | Chi tiết kỹ thuật | Dùng cho |
|---|---|---|
| **Mặt nạ ROI sau Canny** | Belhadri (mục 3.3): Đặt mặt nạ ROI ngay sau Canny: $I_{ROI} = I_e \cdot R$. Triệt tiêu cạnh ngoài lòng đường, giảm tải tính toán cho Hough. | Thiết kế hình thang ROI trong `hough.py`. |
| **Hạn chế của Hough đường thẳng** | Munajat (mục I & IV): Hough chỉ phát hiện được đường thẳng, gặp khúc cua thì phiếu bầu accumulator bị phân tán đỉnh. Bài báo gợi ý chia phân đoạn hoặc nội suy Spline. | Phân tích ca lỗi nhóm Curved và hướng cải tiến. |
| **P-Hough cho thời gian thực** | Belhadri (mục 4.1): Toàn bộ pipeline cần đạt dưới 50 ms/khung. P-Hough chạy trong 7.75 ms (~129 FPS), nhanh gấp nhiều lần Standard Hough. | Luận điểm chọn P-Hough và chứng minh tốc độ. |

## 4. Việc cần học (theo thứ tự)

1. **Phương trình Hessian $\rho = x \cos \theta + y \sin \theta$:** Hiểu cách mỗi điểm ảnh chuyển thành một đường sin trong không gian tham số, và giao điểm là đường thẳng cần tìm.
2. **Khác biệt cốt lõi giữa `HoughLines` và `HoughLinesP`:** Standard Hough trả về đường thẳng vô hạn không có điểm mút; P-Hough trả về đoạn thẳng có tọa độ $[x_1, y_1, x_2, y_2]$, cho phép lọc theo chiều dài (`minLineLength`) và nối đoạn đứt (`maxLineGap`).
3. **Hình học camera hành trình:** Hiểu vì sao vạch trái có độ dốc âm và vạch phải có độ dốc dương trong hệ trục tọa độ ảnh máy tính ($y$ hướng xuống).
4. **Hồi quy WLS (Weighted Least Squares):** Hiểu vì sao phải hồi quy $x$ theo $y$ thay vì $y$ theo $x$ khi vạch đường có xu hướng dốc đứng.

## 5. Việc cần làm (theo thứ tự)

1. **Cài đặt module `hough.py`:** Triển khai các hàm `apply_roi`, `detect_hough_lines_p`, `split_left_right_segments`, `fit_boundary_line`, `detect_road_boundaries`.
2. **Viết unit test `tests/test_hough.py`:** Kiểm tra đối xứng ROI, loại bỏ đoạn ngang, khớp chính xác đường collinear mẫu.
3. **Chạy thực nghiệm quét tham số:** Quét lưới 27 tổ hợp tham số trên toàn bộ 50 ảnh (`scripts/generate_hough_assets.py`), xuất dữ liệu vào `outputs/hough/`.
4. **Đo đạc chỉ số định lượng:** Tính tỷ lệ bắt biên, $IoU$ diện tích vùng đường và sai số $MAE$ tại 3 cao độ ($60\%, 75\%, 90\% \times H$).
5. **Soạn thảo báo cáo & Slide:** Hoàn thành báo cáo học thuật và chuẩn bị 6 slide then chốt.

## 6. Điểm cần cảnh giác khi dùng ba bài báo

1. **Không bài nào công bố tham số P-Hough cụ thể:** Các bài chỉ nói dùng HoughLines/HoughLinesP mà không đưa bộ tham số chi tiết. Bộ tham số $(25, 25, 25)$ trong đồ án là đóng góp thực nghiệm trực tiếp của nhóm.
2. **Độ cao đường chân trời:** Belhadri lấy ROI cố định từ nửa dưới ảnh ($0.50H$). Nhưng trên tập BDD100K/IDD, chân trời thực tế dao động từ $0.40H$ đến $0.52H$. Thiết kế ROI quá thấp ($0.60H$) sẽ làm mất 50% thông tin biên xa.
3. **Cạnh của chính mặt nạ ROI:** Khi nhân bitwise AND giữa Canny và mask ROI, mép viền của ROI có thể tạo ra các đoạn thẳng nhân tạo nếu Canny có pixel dính vào mép. Cần kiểm tra kỹ để bộ lọc độ dốc hoặc co mask loại trừ hiện tượng này.

## 7. Sản phẩm trình bày và dàn 6 Slide

| Slide | Mạch trình bày | Nội dung chính | Minh họa |
|---|---|---|---|
| **1** | Bối cảnh | Vai trò Module 4: Cắt ROI, Hough tìm đoạn thẳng, lọc độ dốc và hồi quy 2 biên đường. | Sơ đồ pipeline tổng thể |
| **2** | Nguyên lý | So sánh Standard Hough vs Probabilistic Hough; ưu thế của P-Hough. | Hình 4.1 |
| **3** | Thuật toán | ROI hình thang theo chân trời $0.48H$; Bộ lọc độ dốc loại bỏ vạch ngang; Ràng buộc không gian. | Hình 4.2 |
| **4** | Toán học | Hồi quy WLS $x = k \cdot y + b$ có trọng số $w_i = \sqrt{\Delta x^2 + \Delta y^2}$ triệt tiêu kỳ dị độ dốc đứng. | Công thức WLS & Đa giác |
| **5** | Thực nghiệm | Tỷ lệ bắt biên 74% trái, 72% phải; Tốc độ 7.75 ms (~129 FPS); Khảo sát độ nhạy tham số. | Bảng 4.2 & Đồ thị 4.3 |
| **6** | Giới hạn & Hướng đi | Hiện tượng phân kỳ hình học trên đường cong (`Curved`); Đề xuất cải tiến Spline / Parabol. | Hình 4.4 |

## Phụ lục A. Bảng tham số khuyến nghị

| Tham số | Giá trị khuyến nghị | Ý nghĩa & Tác dụng |
|---|---|---|
| $\text{ROI}_{top\_y}$ | **$0.48 \times H$** (259 px) | Bắt trọn điểm hội tụ chân trời thực nghiệm của bộ dữ liệu |
| $\text{ROI}_{bottom\_y}$ | **$0.98 \times H$** (529 px) | Triệt tiêu hoàn toàn viền nắp capo xe |
| $\text{ROI}_{top\_width}$ | **$0.35 \times W$** (336 px) | Đảm bảo không xén mất biên đường ở cự ly xa |
| $\text{ROI}_{bottom\_width}$ | **$0.98 \times W$** (940 px) | Ôm trọn lòng đường ở cự ly gần mũi xe |
| `threshold` | **25** | Ngưỡng tích lũy phiếu tối thiểu (cân bằng giữa nhạy và lọc nhiễu) |
| `minLineLength` | **25.0 px** | Loại bỏ các đoạn đứt vụn nhỏ hơn 25 pixel |
| `maxLineGap` | **25.0 px** | Cho phép nối các vạch kẻ đường đứt quãng |
| `min_abs_slope` | **0.30** ($\approx 16.7^\circ$) | Loại bỏ triệt để các vạch ngang, gờ giảm tốc, vết nối nhựa đường |
| `max_abs_slope` | **4.00** ($\approx 76.0^\circ$) | Loại bỏ các vạch gần như thẳng đứng sát tâm xe |
