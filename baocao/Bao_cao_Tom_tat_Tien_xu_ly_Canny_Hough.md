# BÁO CÁO TÓM TẮT: TIỀN XỬ LÝ, CANNY VÀ HOUGH TRƯỚC VÀ SAU CẢI TIẾN

*So sánh pipeline gốc (Phần 2–4) với pipeline hiện tại (v3, `road_pipeline.py`) trên 50 ảnh của bộ dữ liệu*

## 1. Tóm tắt

Báo cáo này tóm tắt cách pipeline hiện tại xử lý ba bước tiền xử lý, Canny và Hough, rồi so sánh với cách làm ở các Phần 2–4. Câu trả lời ngắn: **có khác biệt rõ ở cả ba bước**. Khác biệt lớn nhất nằm ở cách tạo vùng đường. Bản gốc lấy vùng đường là đa giác nằm giữa hai đường Hough. Bản hiện tại phân vùng lòng đường bằng GrabCut. Canny và Hough được dùng để ước lượng điểm tụ, cắt bớt phần mặt nạ tràn ra ngoài và xác định hai biên trái/phải.

| Chỉ số (50 ảnh) | Trước (Phần 2–4) | Sau (pipeline v3) |
|---|---:|---:|
| IoU trung bình toàn tập | 0,174 | **0,689** |
| F1 | 0,256 | **0,807** |
| Số ảnh có vùng đường dự đoán | 27/50 | 50/50 |
| Số ảnh IoU ≥ 0,7 | 0/50 | 25/50 |
| Số ảnh có đủ hai biên trái/phải | 27/50 | 36/50 |
| MAE vị trí biên (số ảnh đo được) | 272,4 px (10 ảnh) | 97,3 px (42 ảnh) |
| SR15 (MAE < 15 px) | 0% | 8% |
| IoU nhóm ảnh tối (13 ảnh) | 0,094 | 0,622 |

*Bảng 1. So sánh tổng thể. IoU tính trên toàn ảnh, ảnh không có dự đoán tính IoU = 0. Nguồn: `outputs/hough/hough_50_images_results.csv`, `outputs/pipeline_v3/results.csv`.*

Bản v3 có IoU cao hơn bản gốc ở cả 50/50 ảnh. Kết quả này đạt được qua ba phiên bản trung gian:

| Phiên bản | Cách làm chính | IoU | F1 |
|---|---|---:|---:|
| Bản gốc (Phần 2–4) | Lọc vạch trắng/vàng HSV → Canny (50, 150) → Hough → đa giác | 0,174 | 0,256 |
| Notebook `Pipeline_HSV_Canny_Hough` | Mặt nạ màu mặt đường theo S, V → Canny → Hough | 0,559 | 0,707 |
| Pipeline v2 | Chuẩn hóa tương phản → GrabCut → Canny + Hough trên viền mặt nạ | 0,671 | 0,795 |
| **Pipeline v3 (hiện tại)** | v2 + hậu xử lý mặt nạ (mép capo cong, đóng hình thái, cắt theo tia Hough) | **0,689** | **0,807** |

*Bảng 2. Các mốc của pipeline.*

## 2. Sơ đồ hai pipeline

```
TRƯỚC (Phần 2–4)
Ảnh → resize 960×540
  → HSV, chỉ giữ pixel trắng/vàng → Gauss σ = 1,2 → ảnh xám
  → Canny NumPy (50, 150)
  → ROI hình thang cố định → P-Hough → chọn cụm + WLS
  → hai đường → đa giác giữa hai đường = vùng đường

SAU (pipeline v3)
Ảnh → resize 960×540
  → [nếu ảnh tối] gamma + CLAHE trên kênh L (Lab)
  → điểm tụ (Canny + Hough trên kênh V)
    và mép capo (quét hàng theo màu S, V)
  → GrabCut khởi tạo bằng hình học → mặt nạ lòng đường
  → hậu xử lý: cắt theo mép capo cong, đóng hình thái,
    cắt theo tia biên (Canny ngưỡng phân vị → P-Hough)
  → vùng đường cuối cùng
  → Canny trên viền mặt nạ → P-Hough trái/phải + WLS
  → hai biên để vẽ và đo MAE
```

## 3. Tiền xử lý

### 3.1 Trước

Phần 2 dùng chuỗi: resize 960 × 540 (Lanczos) → chuyển RGB sang HSV → giữ pixel trắng (S ≤ 80, V ≥ 170) hoặc vàng (15 ≤ H ≤ 40, S ≥ 70, V ≥ 90) → Gauss σ ≈ 1,2 → ảnh xám. Mục tiêu là giữ vạch sơn, xóa phần còn lại trước khi tính gradient.

Hai hệ quả của cách làm này:

- Mép vật lý của lòng đường (nhựa đường giáp vỉa hè, cỏ, lề đất) không có màu trắng/vàng nên bị xóa ngay từ bước đầu. Bước sau chỉ còn cạnh của vạch sơn, mà ground truth của đề tài là toàn bộ lòng đường.
- Ngưỡng V cố định loại gần hết pixel ở ảnh tối. Hình 1 (cột trái) cho thấy ảnh curved_06 chỉ còn vài vạch vàng và đèn đường.

### 3.2 Sau

- **Resize** giữ nguyên 960 × 540.
- **Chuẩn hóa tương phản có điều kiện** (`normalize_contrast`). Ảnh được chuyển sang Lab, chỉ xử lý kênh độ sáng L và giữ nguyên sắc độ a, b. Nếu trung vị L thấp hơn 0,45 thì áp gamma γ = ln 0,45 / ln(trung vị L), kẹp trong [1/2,5; 1]. Sau đó áp CLAHE (clip 2,0, lưới 8 × 8). Bước này **chỉ chạy khi trung vị L của nửa dưới ảnh nhỏ hơn 50**, áp cho 13/50 ảnh, trong đó 10 ảnh thuộc nhãn "tối" của Phần 3. Ảnh đủ sáng giữ nguyên, vì CLAHE làm đổi màu cục bộ và làm GrabCut kém đi.
- **HSV không còn dùng để lọc bỏ pixel.** Kênh S, V chỉ dùng để ước lượng thông tin hình học (`road_prior`): mặt nạ vạch sơn bằng top-hat, mô hình màu mặt đường (trung vị và MAD của S, V trong cửa sổ ngay trước xe) và quét từng hàng để tìm mép capo. Kênh V dùng cho bước ước lượng điểm tụ (mục 5.2).
- **Bước mới: phân vùng lòng đường bằng GrabCut** (`segment_road`). Đây là phần thay thế cho đa giác Hough. GrabCut được khởi tạo tự động: phía trên đường chân trời và phía dưới mép capo là nền chắc chắn; cửa sổ cột 42–58% W ngay trước xe (đi theo điểm tụ) là đường chắc chắn; nêm từ điểm tụ là "có thể là đường". Chạy 3 lần với seed 0, 1, 2 rồi lấy đa số, chỉ giữ thành phần liên thông chứa cửa sổ mẫu.
- **Hậu xử lý mặt nạ** (`refine_mask`, mới ở v3): cắt theo mép capo dạng parabol khi dò được (12/50 ảnh), đóng hình thái bằng ellipse 141 px để lấp chỗ xe che, cắt theo tia biên do Canny + Hough tìm được (mục 5.2).

### 3.3 Khác biệt

| Khía cạnh | Trước | Sau |
|---|---|---|
| Vai trò của HSV | Lọc giữ vạch trắng/vàng, xóa phần còn lại | Ước lượng màu mặt đường, mép capo, vạch sơn; không xóa pixel |
| Xử lý ảnh tối | Không có; ngưỡng V cố định | Gamma + CLAHE trên kênh L, chỉ cho ảnh tối |
| Đầu ra bàn giao | Ảnh xám đã lọc màu | Ảnh màu đã chuẩn hóa + điểm tụ + mép capo |
| Tạo vùng đường | Không thuộc bước này | GrabCut + hậu xử lý tạo luôn mặt nạ lòng đường |

*Bảng 3. Tiền xử lý trước và sau.*

Đóng góp đo được (ablation trên v2): tắt chuẩn hóa tương phản làm IoU toàn tập giảm từ 0,671 xuống 0,658. Toàn bộ mức giảm nằm ở nhóm tối (0,612 xuống 0,562); ba nhóm còn lại không đổi.

## 4. Canny

### 4.1 Trước

Phần 3 tự cài Canny bằng NumPy (Gauss → Sobel → loại điểm không cực đại → ngưỡng kép + hysteresis), khớp `cv2.Canny` với IoU ảnh cạnh 0,952. Cấu hình bàn giao: nhân Gauss 5 × 5, σ = 1,4, ngưỡng cố định (50, 150), L2gradient. Đầu vào là ảnh xám đã lọc HSV, nên Canny chỉ thấy cạnh của vạch sơn. Ở ảnh tối, chỉ 8,8% điểm trên biên lòng đường có cạnh đi qua.

### 4.2 Sau

Hàm Canny NumPy giữ nguyên. Thay đổi nằm ở đầu vào, ngưỡng và chỗ dùng. Pipeline hiện tại dùng Canny ở ba chỗ:

- **Canny ngưỡng theo phân vị gradient** (`canny_edges_auto`, `edge_map`). Đầu vào là ảnh xám của ảnh **đã chuẩn hóa tương phản, không lọc màu**. Ngưỡng cao bằng phân vị 90% của độ lớn gradient Sobel trong vùng dưới đường chân trời; ngưỡng thấp bằng 0,4 lần ngưỡng cao (sàn 8). Ngưỡng tự co giãn theo độ tương phản của từng ảnh. Ở v3, ảnh cạnh này là đầu vào cho bước cắt theo tia biên.
- **Canny trên viền mặt nạ** (`boundary_lines`). Mặt nạ lòng đường được lấp lỗ bằng contour ngoài, làm mờ Gauss, rồi chạy Canny NumPy (50, 150, σ = 1,4). Kết quả là viền của vùng đường, dùng cho Hough tìm hai biên. Vì đầu vào là ảnh nhị phân, gradient ở viền rất lớn, nên ngưỡng cố định ở đây không gây mất cạnh.
- **Canny cho điểm tụ.** `cv2.Canny(50, 150)` trên kênh V đã làm mờ, kế thừa từ notebook.

### 4.3 So sánh chất lượng ảnh cạnh

| Điều kiện | Gốc: lọc HSV + (50, 150) | Ảnh xám + (50, 150) | Chuẩn hóa + (50, 150) | Chuẩn hóa + ngưỡng phân vị |
|---|---:|---:|---:|---:|
| sáng | 0,632 / 0,031 | 0,679 / 0,035 | 0,835 / 0,052 | 0,861 / 0,062 |
| tương phản cao | 0,368 / 0,026 | 0,592 / 0,041 | 0,778 / 0,066 | 0,836 / 0,074 |
| vừa | 0,182 / 0,019 | 0,379 / 0,024 | 0,703 / 0,046 | 0,824 / 0,063 |
| **tối** | **0,088 / 0,010** | 0,214 / 0,018 | 0,457 / 0,035 | **0,736 / 0,058** |
| tất cả | 0,276 / 0,021 | 0,442 / 0,029 | 0,680 / 0,050 | 0,810 / 0,065 |

*Bảng 4. Recall / density của ảnh cạnh. Recall là tỉ lệ điểm trên biên lòng đường có cạnh trong bán kính 5 px, density là tỉ lệ pixel cạnh (cùng định nghĩa với Phần 3). Nguồn: `outputs/pipeline_v2/canny_contrast_by_condition.csv`.*

Bảng 4 tách đóng góp của từng thay đổi. Bỏ lọc HSV nâng recall từ 0,276 lên 0,442. Chuẩn hóa tương phản nâng tiếp lên 0,680. Ngưỡng phân vị nâng lên 0,810. Ở ảnh tối, recall tăng từ 0,088 lên 0,736. Tỉ lệ mật độ cạnh giữa ảnh sáng và ảnh tối giảm từ 3,1 lần xuống 1,07 lần, nghĩa là lượng cạnh gần như không còn phụ thuộc vào ánh sáng. Đổi lại, density tăng từ 0,021 lên 0,065, tức là có thêm cạnh nhiễu (cây, nhà, xe), như thấy ở Hình 1, cột thứ hai.

### 4.4 Khác biệt

| Khía cạnh | Trước | Sau |
|---|---|---|
| Cài đặt | Canny NumPy tự cài | Giữ nguyên hàm NumPy, thêm hàm ngưỡng tự động |
| Đầu vào | Ảnh xám đã lọc trắng/vàng | Ảnh xám đã chuẩn hóa (không lọc màu); hoặc mặt nạ lòng đường |
| Ngưỡng | Cố định (50, 150) | Phân vị 90% gradient, thấp = 0,4 × cao; (50, 150) chỉ dùng trên mặt nạ nhị phân |
| Vai trò | Nguồn cạnh duy nhất cho Hough | Ba vai trò: điểm tụ, cắt mặt nạ theo tia, viền mặt nạ cho hai biên |
| Recall biên (tối / tất cả) | 0,088 / 0,276 | 0,736 / 0,810 |

*Bảng 5. Canny trước và sau.*

## 5. Hough

### 5.1 Trước

Phần 4 áp ROI hình thang cố định (đáy trên ở 0,48 H, rộng 35% W; đáy dưới ở 0,98 H, rộng 98% W), chạy P-Hough (ngưỡng 25, minLineLength 25, maxLineGap 25), lọc đoạn có |độ dốc| trong [0,3; 4]. Đoạn trái có độ dốc âm và x giữa < 0,65 W, đoạn phải ngược lại. Mỗi bên chọn cụm đồng thuận (đoạn có hai đầu mút cách mô hình ≤ 0,02 W) rồi khớp WLS x = k·y + b. Vùng đường là đa giác giữa hai đường.

Vì đầu vào chỉ có cạnh vạch sơn, Hough thường chọn vạch kẻ làn bên trong lòng đường, nên đa giác hẹp hơn mặt đường thật (Hình 2, cột trái). Chỉ 27/50 ảnh có đa giác hợp lệ; ở ảnh tối là 4/13.

### 5.2 Sau

Hough được dùng ở ba chỗ, và không còn trực tiếp tạo ra vùng đường:

- **Ước lượng điểm tụ** (`estimate_vanishing_point`). HoughLinesP (ngưỡng 40, minLineLength 40, maxLineGap 10) trên ảnh cạnh của kênh V. Mỗi cặp đoạn nghiêng trái và nghiêng phải bỏ một phiếu tại giao điểm, trọng số bằng tích chiều dài. Điểm tụ là ô 20 × 20 px có nhiều phiếu nhất. Điểm tụ dùng để khởi tạo GrabCut và làm gốc cho bước cắt theo tia.
- **Cắt mặt nạ theo tia biên** (mới ở v3, `_vp_rays`). HoughLinesP (ngưỡng 25, minLineLength 35, maxLineGap 8) trên ảnh Canny ngưỡng phân vị. Chỉ giữ đoạn cách điểm tụ dưới 22 px; các đoạn này bỏ phiếu theo góc (ô 1,5°, trọng số là chiều dài). Mỗi bên chỉ cắt khi có tia nghiêng ≥ 62° so với phương thẳng đứng, được ủng hộ ≥ 60 px và màu Lab hai bên tia lệch ΔE ≥ 30. Đây là chỗ Canny + Hough tham gia trực tiếp vào việc tạo vùng đường.
- **Tìm hai biên trái/phải** (`boundary_lines`). Bỏ ROI hình thang. Thay vào đó, ở mỗi hàng từ y_vp + 5% H tới mép capo − 10 px, chỉ giữ cạnh nằm trong ±4 px quanh điểm ngoài cùng bên trái và bên phải của mặt nạ; bỏ hàng mà mặt nạ chạm mép khung ảnh. Chạy P-Hough riêng cho từng bên, ngưỡng độ dốc nới từ 0,30 xuống 0,10 để nhận được biên thoải ở khúc cua. Chọn cụm đồng thuận và khớp WLS vẫn dùng nguyên hàm của `hough.py` (Phần 4).

### 5.3 Khác biệt

| Khía cạnh | Trước | Sau |
|---|---|---|
| Đầu vào | Cạnh vạch sơn trong ROI hình thang | Viền mặt nạ lòng đường; ảnh cạnh ngưỡng phân vị (cắt theo tia); cạnh kênh V (điểm tụ) |
| Vùng quan tâm | Hình thang cố định theo tỉ lệ ảnh | Dải ±4 px quanh mép trái/phải của mặt nạ, giới hạn bởi điểm tụ và mép capo |
| Độ dốc tối thiểu | 0,30 | 0,10 |
| Chọn cụm + WLS | Có | Giữ nguyên (cùng hàm `hough.py`) |
| Vai trò | Hai đường → đa giác = vùng đường | Điểm tụ, cắt vùng tràn, hai biên để vẽ và đo MAE |
| Có đủ hai biên | 27/50 | 36/50 (trái 43, phải 42) |

*Bảng 6. Hough trước và sau.*

## 6. Kết quả so sánh

### 6.1 Theo nhóm ảnh và điều kiện ánh sáng

| Nhóm | N | IoU trước | IoU sau (v3) | F1 sau |
|---|---:|---:|---:|---:|
| Normal | 18 | 0,172 | 0,772 | 0,867 |
| Curved | 12 | 0,192 | 0,668 | 0,791 |
| Adverse | 20 | 0,166 | 0,627 | 0,762 |
| **Toàn tập** | 50 | **0,174** | **0,689** | **0,807** |

*Bảng 7. IoU theo nhóm ảnh.*

| Điều kiện | N | Được chuẩn hóa tương phản | IoU trước | IoU sau (v3) |
|---|---:|---:|---:|---:|
| sáng | 7 | 0 | 0,230 | 0,772 |
| tương phản cao | 15 | 1 | 0,214 | 0,636 |
| vừa | 15 | 2 | 0,178 | 0,761 |
| tối | 13 | 10 | 0,094 | 0,622 |

*Bảng 8. IoU theo điều kiện ánh sáng (nhãn của Phần 3, chỉ dùng để nhóm kết quả).*

Ở bản gốc, IoU giảm mạnh từ ảnh sáng sang ảnh tối (0,230 xuống 0,094). Ở bản v3, khoảng cách này thu hẹp (0,772 xuống 0,622), nhưng ảnh tối và ảnh tương phản cao vẫn là hai nhóm yếu nhất.

### 6.2 Hình minh họa

![Các bước trên ảnh tối curved_06](outputs/pipeline_v3/figures/stages_curved_06.jpg)

*Hình 1. Ảnh tối curved_06. Cột trái: ảnh gốc và pipeline gốc (lọc HSV, Canny (50, 150), Hough và đa giác, IoU 0,00). Cột phải: pipeline v3 (chuẩn hóa tương phản, mặt nạ GrabCut với phần bị hậu xử lý cắt đi màu đỏ và phần được lấp màu lam, Canny ngưỡng phân vị, kết quả cuối IoU 0,77). Viền tím là ground truth.*

![Trước và sau trên ba ảnh](outputs/pipeline_v3/figures/truoc_sau_3_anh.jpg)

*Hình 2. Kết quả của pipeline gốc (trái) và v3 (phải) trên ba ảnh thuộc ba nhóm. Bên trái: đa giác Hough (xanh lá) nằm giữa hai vạch kẻ làn, viền vàng mảnh là ROI. Bên phải: mặt nạ lòng đường (xanh lá), biên trái (đỏ), biên phải (xanh dương). Viền tím là ground truth.*

### 6.3 Đóng góp của từng thay đổi

| Thay đổi | IoU khi có | IoU khi bỏ | Chênh lệch |
|---|---:|---:|---:|
| Chuẩn hóa tương phản (trên v2) | 0,671 | 0,658 | +0,013 |
| Bỏ phiếu 3 lần GrabCut (trên v2) | 0,671 | 0,658 | +0,013 |
| Cửa sổ mẫu đi theo điểm tụ (trên v2) | 0,671 | 0,645 | +0,026 |
| Nền chắc chắn ở 3% hàng cuối (trên v2) | 0,671 | 0,645 | +0,026 |
| Cắt theo mép capo cong (trên v3) | 0,689 | 0,681 | +0,008 |
| Đóng hình thái (trên v3) | 0,689 | 0,683 | +0,006 |
| Cắt theo tia Canny + Hough (trên v3, so với capo + đóng) | 0,689 | 0,684 | +0,005 |

*Bảng 9. Ablation: tắt từng thay đổi, giữ nguyên phần còn lại. Nguồn: `outputs/pipeline_v2/ablation_summary.csv`, `outputs/pipeline_v3/ablation_summary.csv`.*

Phần lớn mức tăng đến từ việc đổi cách tạo vùng đường (GrabCut thay đa giác Hough). Riêng GrabCut, không kèm cải tiến nào, đã đạt 0,603. Các cải tiến ở tiền xử lý và hậu xử lý cộng thêm khoảng 0,09.

## 7. Kết luận

**Có khác biệt, và khác biệt ở cả ba bước:**

- **Tiền xử lý** đổi mục tiêu từ "giữ vạch sơn" sang "giữ toàn bộ thông tin ảnh và làm rõ ảnh tối". HSV không còn xóa pixel; ảnh tối được chuẩn hóa độ sáng trên kênh L. Bước phân vùng GrabCut được thêm vào và trở thành bước quyết định vùng đường.
- **Canny** giữ nguyên thuật toán tự cài nhưng đổi đầu vào (ảnh không lọc màu) và ngưỡng (theo phân vị gradient thay vì cố định). Recall biên lòng đường tăng từ 0,276 lên 0,810; ở ảnh tối từ 0,088 lên 0,736.
- **Hough** không còn dùng ROI hình thang và không còn tạo đa giác vùng đường. Nó được dùng để tìm điểm tụ, cắt phần mặt nạ tràn ra ngoài và tìm hai biên từ viền mặt nạ. Hàm chọn cụm và WLS của Phần 4 được giữ nguyên.

Kết quả: IoU từ 0,174 lên 0,689, F1 từ 0,256 lên 0,807, tốt hơn bản gốc ở cả 50 ảnh.

**Các điểm cần lưu ý khi trình bày:**

- Canny + Hough không còn là thành phần chính tạo ra vùng đường. Trong ablation, bước cắt theo tia Canny + Hough chỉ thêm khoảng 0,005 IoU. Nếu đề tài yêu cầu Canny + Hough là trung tâm thì cần nói rõ điều này.
- Mục tiêu IoU 0,7 chưa đạt trên toàn tập (0,689; 25/50 ảnh đạt). MAE còn lớn (97,3 px), SR15 chỉ 8%.
- Tham số được chọn trên chính 50 ảnh. Ngưỡng hậu xử lý v3 được chọn trên nửa ảnh chỉ số chẵn và kiểm tra trên nửa chỉ số lẻ (IoU 0,690 và 0,688), nhưng đây chưa phải một tập kiểm tra độc lập.
- Chậm hơn nhiều: khoảng 6,2 s/ảnh trên CPU (3 lần GrabCut, Canny NumPy, dò mép capo). Bản gốc mất khoảng 0,2–0,3 s cho Canny và 3,4 ms cho Hough.

## Cách tái tạo

```
python road_pipeline.py --edge-maps              # pipeline v3, ghi outputs/pipeline_v3/
python road_pipeline.py --no-refine --edge-maps  # pipeline v2, ghi outputs/pipeline_v2/
python scripts/compare_canny_contrast.py         # Bảng 4
python scripts/ablation_pipeline_v2.py           # Bảng 9 (v2)
python scripts/ablation_pipeline_v3.py           # Bảng 9 (v3)
python scripts/figures_tom_tat_3_buoc.py         # Hình 1, 2
python scripts/build_tom_tat_3_buoc_docx.py      # xuất file .docx này
```
