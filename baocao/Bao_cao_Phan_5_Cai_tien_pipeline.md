# BÁO CÁO CẢI TIẾN PIPELINE PHÁT HIỆN LÒNG ĐƯỜNG

*Chuẩn hóa tương phản cho ảnh tối, ngưỡng Canny thích nghi, phân vùng lòng đường bằng GrabCut, Canny + Hough trên viền vùng đường*

## 1. Tóm tắt

Báo cáo này trình bày các thay đổi so với pipeline hiện có của nhóm và đo kết quả trên toàn bộ 50 ảnh của bộ dữ liệu. Mã nguồn nằm ở `road_pipeline.py`, kết quả ở `outputs/pipeline_v2/`.

| Cấu hình | IoU | F1 | Ghi chú |
|---|---:|---:|---|
| Bản gốc: lọc vạch HSV → Canny (50, 150) → Hough | 0,174 | 0,256 | Báo cáo tổng hợp |
| Baseline tầm thường: tô kín dải từ chân trời tới capo | 0,558 | – | Không dùng biên trái/phải |
| Notebook `Pipeline_HSV_Canny_Hough`: mặt nạ màu S-V → Canny → Hough | 0,559 | 0,707 | |
| **Pipeline v2 (báo cáo này)** | **0,671** | **0,795** | Precision 0,807, recall 0,825 |

*Bảng 1. Kết quả trên 50 ảnh, đánh giá trên mặt nạ lòng đường, mẫu số là toàn bộ ảnh.*

Các điểm chính:

- IoU tăng từ 0,559 lên 0,671 (+0,112) và F1 từ 0,707 lên 0,795. Bản v2 tốt hơn notebook ở 38/50 ảnh và tốt hơn baseline tầm thường ở 42/50 ảnh.
- Ở bước Canny, tỉ lệ điểm trên biên lòng đường có cạnh đi qua tăng từ 0,276 lên 0,810. Riêng ảnh tối tăng từ 0,088 lên 0,736.
- Nhóm normal đạt IoU 0,744 (F1 0,850). Nhóm adverse vẫn thấp nhất với 0,613.
- Mục tiêu IoU 0,7 chưa đạt trên toàn tập: 23/50 ảnh đạt IoU ≥ 0,7.

## 2. Hiện trạng trước khi cải tiến

Ground truth của đề tài là mặt nạ **toàn bộ lòng đường**, kể cả vạch sơn và phần mặt đường bị xe che.

- **Bản gốc** (Phần 2–4) lọc vạch trắng/vàng rồi mới chạy Canny, nên Hough bám vào vạch kẻ làn bên trong lòng đường. Kết quả: IoU 0,174, chỉ 54% ảnh có đa giác.
- **Notebook pipeline** đổi vai trò của HSV sang tìm mặt đường và đạt IoU 0,559. Tuy nhiên, tô kín cả dải ngang từ đường chân trời tới mép capo, không cần biên trái/phải, đã được 0,558. Nghĩa là gần như toàn bộ IoU của notebook đến từ việc đặt đúng giới hạn trên và dưới, còn hai biên trái/phải hầu như không đóng góp.

## 3. Phân tích nguyên nhân

Trước khi sửa, nhóm đo xem sai số đến từ đâu. Các con số trong mục này đo trên đầu ra của notebook.

**Trần của cách biểu diễn bằng hai đường thẳng.** Lấy hai đường thẳng khớp với mép trái/phải của chính ground truth, IoU chỉ đạt 0,681. Nếu dùng giới hạn trên/dưới mà pipeline ước lượng được thì còn 0,618. Nếu chọn tối ưu cặp tia đi qua điểm tụ để IoU với ground truth lớn nhất thì được 0,686. Ground truth có khúc cua, có xe và có vùng đường rộng hết khung ảnh, nên **mọi phương án chỉ xuất ra hai đường thẳng đều không thể vượt khoảng 0,68–0,69**.

**Giới hạn trên/dưới không phải nguyên nhân chính.** Thay chân trời và mép capo ước lượng bằng giá trị lấy từ ground truth, IoU của notebook chỉ tăng từ 0,559 lên 0,594.

**Bằng chứng màu S-V không phân biệt được mặt đường.** Ứng viên biên của Hough là đủ tốt: nếu biết trước nên chọn cặp nào thì đạt 0,65. Nhưng chọn bằng mặt nạ màu S-V thì chỉ được khoảng 0,556. Nguyên nhân là bản đồ "giống mặt đường" theo S và V phủ cả cây, nhà và trời, nên phép chọn luôn rơi về mép ảnh, tức là trùng với baseline tầm thường. Chuyển sang khoảng cách màu trong Lab (thêm sắc độ a, b) thì precision trong dải đường tăng từ khoảng 0,57 lên khoảng 0,70.

**Ảnh tương phản thấp làm mất cạnh ngay từ đầu.** Ngưỡng V cố định của HSV và ngưỡng Canny cố định (50, 150) khiến ảnh tối gần như không còn cạnh. Theo Phần 3, recall cạnh ở ảnh tối chỉ 0,09, so với 0,63 ở ảnh sáng.

Từ đó rút ra hai hướng sửa: (1) xử lý ảnh tương phản thấp ở bước tiền xử lý và Canny; (2) cho đầu ra cuối cùng là một **vùng** bám theo mép thật, còn Canny + Hough dùng để xác định hai biên.

## 4. Cải tiến 1: tiền xử lý cho ảnh tương phản thấp và ngưỡng Canny thích nghi

**Chuẩn hóa tương phản** (`normalize_contrast` trong `road_pipeline.py`):

1. Chuyển RGB sang Lab và chỉ xử lý kênh độ sáng L, giữ nguyên sắc độ a, b. Như vậy màu của mặt đường không bị lệch.
2. Nếu trung vị L thấp hơn 0,45, áp gamma γ = ln 0,45 / ln(trung vị L), kẹp trong [1/2,5; 1], để kéo độ sáng trung vị về 0,45.
3. Áp CLAHE trên L (Zuiderveld, 1994), clip 2,0, lưới 8 × 8.
4. Chỉ áp dụng khi trung vị L của nửa dưới ảnh nhỏ hơn 50. Thực nghiệm cho thấy áp CLAHE cho mọi ảnh làm IoU phân vùng giảm, vì CLAHE thay đổi màu cục bộ. Với ngưỡng 50, 13/50 ảnh được chuẩn hóa, trong đó có 10 ảnh thuộc nhãn "tối" của Phần 3.

**Ngưỡng Canny theo phân vị gradient** (`canny.gradient_percentile_thresholds` và `canny.canny_edges_auto`, vẫn là bản Canny NumPy tự cài): ngưỡng cao bằng phân vị 90% của độ lớn gradient Sobel trong vùng dưới đường chân trời, ngưỡng thấp bằng 0,4 lần ngưỡng cao (sàn 8). Ngưỡng tự co giãn theo độ tương phản của từng ảnh, nên ảnh tối không còn bị mất cạnh.

| Điều kiện | Gốc: lọc HSV + (50, 150) | Ảnh xám + (50, 150) | Chuẩn hóa + (50, 150) | Chuẩn hóa + ngưỡng phân vị |
|---|---:|---:|---:|---:|
| sáng | 0,632 / 0,031 | 0,679 / 0,035 | 0,835 / 0,052 | 0,861 / 0,062 |
| tương phản cao | 0,368 / 0,026 | 0,592 / 0,041 | 0,778 / 0,066 | 0,836 / 0,074 |
| vừa | 0,182 / 0,019 | 0,379 / 0,024 | 0,703 / 0,046 | 0,824 / 0,063 |
| **tối** | **0,088 / 0,010** | 0,214 / 0,018 | 0,457 / 0,035 | **0,736 / 0,058** |
| tất cả | 0,276 / 0,021 | 0,442 / 0,029 | 0,680 / 0,050 | 0,810 / 0,065 |

*Bảng 2. Recall / density của ảnh cạnh theo điều kiện ánh sáng (cùng chỉ số với Phần 3: recall là tỉ lệ điểm trên biên lòng đường có cạnh trong bán kính 5 px). Nguồn: `outputs/pipeline_v2/canny_contrast_by_condition.csv`, sinh bằng `scripts/compare_canny_contrast.py`.*

Riêng ở ảnh tối, recall tăng từ 0,088 lên 0,736. Mật độ cạnh của ảnh sáng so với ảnh tối giảm từ 3,1 lần xuống 1,07 lần, tức là lượng cạnh gần như không còn phụ thuộc vào ánh sáng. Đây là vấn đề mà Munajat và cộng sự nêu ra và Phần 3 mới khắc phục được một phần (từ 3,1 lần xuống 2,4–2,8 lần). Đổi lại, density tăng từ 0,021 lên 0,065 nên có thêm cạnh nhiễu.

![Canny trên ảnh tối, trước và sau](outputs/pipeline_v2/figures/canny_low_contrast.jpg)

*Hình 1. Ảnh tối hoặc tương phản thấp: ảnh gốc, Canny của bản gốc (lọc HSV + ngưỡng (50, 150)) và Canny sau khi chuẩn hóa tương phản với ngưỡng theo phân vị.*

**Cần nói rõ:** ảnh cạnh của Bảng 2 hiện chỉ được lưu ra `outputs/pipeline_v2/edge_map/` để minh họa. Nó **chưa** được dùng để tạo vùng đường. Mức tăng IoU ở mục 7 chủ yếu đến từ cải tiến 2. Phần chuẩn hóa tương phản thì có đóng góp, thông qua ảnh đầu vào của GrabCut.

## 5. Cải tiến 2: phân vùng lòng đường bằng GrabCut có ràng buộc hình học

GrabCut (Rother và cộng sự, 2004) mô hình hóa màu của "đường" và "nền" bằng hai GMM 5 thành phần, rồi tách hai vùng bằng cắt đồ thị (graph cut), có tính cả độ tương phản giữa các pixel kề nhau. Đây là phương pháp xử lý ảnh cổ điển, không dùng học sâu. Thay vì để người dùng khoanh vùng, nhóm khởi tạo GrabCut tự động bằng hình học của cảnh (`segment_road`):

| Vùng | Nhãn khởi tạo | Lý do |
|---|---|---|
| Phía trên y_vp − 15 px | Nền chắc chắn | Trên đường chân trời không có mặt đường |
| Phía dưới mép capo + 4 px, và 3% hàng cuối ảnh | Nền chắc chắn | Nắp capo cùng tông tối với mặt đường. Đánh dấu vài hàng cuối giúp GMM nền học được màu capo |
| Cửa sổ ngay trước xe, cột 42–58% W | Đường chắc chắn | Ngay trước xe gần như luôn là mặt đường |
| Nêm từ điểm tụ: \|x − x_vp\| ≤ 3 (y − y_vp) | Có thể là đường | Ràng buộc phối cảnh |
| Phần còn lại | Có thể là nền | |

*Bảng 3. Khởi tạo GrabCut. Điểm tụ và mép capo được ước lượng bằng các hàm của notebook (bỏ phiếu giao điểm đoạn Hough; quét hàng theo màu mặt đường), nay chuyển sang `road_pipeline.py`.*

Ba chi tiết quyết định độ ổn định:

1. **Cửa sổ mẫu đi theo điểm tụ.** Ban đầu cửa sổ được đặt cố định ở 60–68% H. Ở curved_08 và adverse_17, điểm tụ nằm ở khoảng 67% H nên cửa sổ bị loại hết, GrabCut không có pixel "đường chắc chắn" và hai ảnh này có IoU = 0. Bản cuối đặt cửa sổ từ max(60% H, y_vp + 4% H), cao ít nhất 6% H và luôn nằm trên mép capo.
2. **Bỏ phiếu giữa 3 lần chạy.** GrabCut khởi tạo GMM bằng k-means có yếu tố ngẫu nhiên. Ở adverse_14, một lần chạy cho vùng đường co lại còn khoảng 11 nghìn pixel, trong khi các lần khác cho khoảng 60 nghìn. Bản cuối chạy với 3 seed cố định (0, 1, 2) rồi lấy đa số, nên kết quả vừa tái lập được vừa tránh được các lần suy biến.
3. **Chỉ giữ thành phần liên thông chứa cửa sổ mẫu**, để loại các mảng cùng màu tách rời như mái nhà hay tường.

GrabCut chạy ở nửa độ phân giải (480 × 270). Chạy ở độ phân giải đầy đủ không tốt hơn (0,632 so với 0,645 ở cấu hình trung gian).

## 6. Cải tiến 3: Canny + Hough trên viền vùng đường

Hai biên trái/phải được tìm bằng các hàm của Phần 3 và Phần 4 (`boundary_lines`):

1. Lấp lỗ của mặt nạ (vùng xe che, vạch sơn) bằng contour ngoài, rồi chạy Canny NumPy (50, 150, σ = 1,4) trên mặt nạ đã làm mờ.
2. Chỉ giữ cạnh nằm trong ±4 px quanh điểm ngoài cùng bên trái / bên phải của mặt nạ ở mỗi hàng, trong dải từ y_vp + 5% H tới mép capo − 10 px. Bỏ các hàng mà mặt nạ chạm mép khung ảnh. Bước này loại viền trên (xe, chân trời) và viền dưới (capo). Các viền đó cũng có độ dốc và từng làm Hough chọn nhầm biên chạy dọc nắp capo.
3. Chạy HoughLinesP và hồi quy WLS của `hough.py` riêng cho từng bên (|m| ≥ 0,10).

Hai biên này dùng để vẽ và để báo cáo vị trí biên. Vùng đường dùng để chấm IoU là mặt nạ ở mục 5. Khi mặt nạ sai thì biên cũng sai theo (Hình 4).

Kết quả: tìm được biên trái ở 32/50 ảnh và biên phải ở 40/50 ảnh. Trong 18 ảnh thiếu biên trái, 15 ảnh có mặt nạ chạm mép trái khung ảnh ở hơn nửa số hàng trong khoảng y = 300–500. Với biên phải, con số này là 7/10. Tức là phần lớn trường hợp thiếu biên là do vùng đường dự đoán kéo ra ngoài khung hình ở bên đó. Điều này có thể đúng (đường rộng hơn khung hình), nhưng cũng có thể do mặt nạ tràn ra.

## 7. Kết quả

### 7.1 Theo nhóm ảnh

| Nhóm | N | IoU | F1 | Precision | Recall | Notebook (IoU / F1) | Baseline tầm thường |
|---|---:|---:|---:|---:|---:|---:|---:|
| Normal | 18 | 0,744 | 0,850 | 0,818 | 0,908 | 0,565 / 0,713 | 0,568 |
| Curved | 12 | 0,659 | 0,785 | 0,758 | 0,867 | 0,561 / 0,707 | 0,531 |
| Adverse | 20 | 0,613 | 0,751 | 0,826 | 0,726 | 0,553 / 0,701 | 0,564 |
| **Toàn tập** | 50 | **0,671** | **0,795** | 0,807 | 0,825 | 0,559 / 0,707 | 0,558 |

*Bảng 4. Nguồn: `outputs/pipeline_v2/summary.csv`, `outputs/pipeline_notebook/results.csv`.*

Phân bố IoU: không ảnh nào dưới 0,3; 6 ảnh trong [0,3; 0,5); 21 ảnh trong [0,5; 0,7); 23 ảnh từ 0,7 trở lên. Ảnh kém nhất đạt 0,366.

MAE (theo mép trái/phải của mặt nạ tại 3 hàng 60/75/90% H) đo được trên 42/50 ảnh, trung bình 105,2 px, so với 145,6 px trên 43 ảnh của notebook. SR15 là 4/50 ảnh (8%).

### 7.2 Theo điều kiện ánh sáng

| Điều kiện | N | Được chuẩn hóa tương phản | IoU v2 | F1 v2 | IoU notebook | Baseline tầm thường |
|---|---:|---:|---:|---:|---:|---:|
| sáng | 7 | 0 | 0,719 | 0,835 | 0,516 | 0,520 |
| tương phản cao | 15 | 1 | 0,629 | 0,762 | 0,508 | 0,531 |
| vừa | 15 | 2 | 0,742 | 0,846 | 0,618 | 0,598 |
| tối | 13 | 10 | 0,612 | 0,753 | 0,574 | 0,562 |

*Bảng 5. Nhãn điều kiện lấy từ `outputs/canny/conditions.csv` (Phần 3). Nhãn này chỉ dùng để nhóm kết quả, pipeline không dùng ground truth.*

Ảnh tối vẫn là nhóm yếu nhất. Mức tăng ở nhóm này (+0,038 so với notebook) nhỏ hơn hẳn so với các nhóm còn lại (+0,12 đến +0,20).

### 7.3 Đóng góp của từng cải tiến (ablation)

| Cấu hình | IoU | Δ IoU | F1 | Precision | Recall | IoU sáng | IoU TP cao | IoU vừa | IoU tối | Ảnh IoU < 0,1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **v2 đầy đủ** | 0,671 | – | 0,795 | 0,807 | 0,825 | 0,719 | 0,629 | 0,742 | 0,612 | 0 |
| Bỏ chuẩn hóa tương phản | 0,658 | -0,013 | 0,783 | 0,813 | 0,808 | 0,719 | 0,629 | 0,742 | 0,562 | 0 |
| Bỏ bỏ phiếu (1 lần GrabCut, seed 0) | 0,658 | -0,013 | 0,780 | 0,792 | 0,813 | 0,722 | 0,587 | 0,742 | 0,608 | 1 |
| Cửa sổ mẫu cố định 60–68% H | 0,645 | -0,026 | 0,763 | 0,769 | 0,799 | 0,721 | 0,628 | 0,705 | 0,553 | 2 |
| Bỏ nền chắc chắn ở 3% hàng cuối | 0,645 | -0,026 | 0,773 | 0,735 | 0,877 | 0,669 | 0,580 | 0,730 | 0,609 | 0 |
| Chỉ GrabCut (bỏ cả 4 cải tiến trên) | 0,603 | -0,068 | 0,728 | 0,709 | 0,820 | 0,674 | 0,575 | 0,683 | 0,504 | 2 |

*Bảng 7. Tắt từng cải tiến, giữ nguyên phần còn lại. Nguồn: `outputs/pipeline_v2/ablation_summary.csv`, sinh bằng `scripts/ablation_pipeline_v2.py`.*

Nhận xét:

- Mỗi cải tiến đóng góp khoảng 0,013–0,026 IoU. Tắt cả bốn thì còn 0,603, vẫn hơn notebook (0,559). Như vậy riêng việc chuyển sang phân vùng bằng GrabCut đã đóng góp khoảng +0,04; bốn cải tiến cộng thêm khoảng +0,07.
- **Chuẩn hóa tương phản** chỉ tác động lên ảnh tối, đúng như thiết kế: IoU nhóm tối tăng từ 0,562 lên 0,612; các nhóm khác giữ nguyên.
- **Cửa sổ mẫu đi theo điểm tụ** loại được 2 ảnh IoU gần 0 (curved_08, adverse_17).
- **Bỏ phiếu 3 lần** chủ yếu cứu ảnh suy biến (adverse_14) và nhóm tương phản cao (0,587 lên 0,629).
- **Nền chắc chắn ở đáy ảnh** giúp tránh lấn vào nắp capo: precision tăng từ 0,735 lên 0,807, đổi lại recall giảm từ 0,877 xuống 0,825.

### 7.4 Hình minh họa

![So sánh notebook và v2](outputs/pipeline_v2/figures/notebook_vs_v2.jpg)

*Hình 2. Notebook (trái) và v2 (phải). Viền vàng: ground truth; vùng xanh lá: dự đoán; đỏ / xanh dương: biên trái / phải. curved_08 là trường hợp v2 kém hơn notebook (0,55 so với 0,77): vùng dự đoán tràn lên tường bê tông cùng màu bên trái và thiếu dải sát đáy ảnh (do 3% hàng cuối bị đánh dấu là nền). Ở normal_06, mặt nạ đúng (IoU 0,88) nhưng hai biên Hough lại bám vào viền trên của mặt nạ, nơi vùng đường trải rộng gần hết khung ảnh, nên cắt chéo nhau.*

![Trường hợp tốt nhất](outputs/pipeline_v2/figures/best_cases.jpg)

*Hình 3. Sáu ảnh có IoU cao nhất.*

![Trường hợp kém nhất](outputs/pipeline_v2/figures/failure_cases.jpg)

*Hình 4. Sáu ảnh có IoU thấp nhất.*

Các kiểu lỗi chính nhìn từ Hình 4:

- **Tràn sang vùng cùng màu nhựa đường** như bãi đỗ, vỉa hè, làn đường ngược chiều (adverse_03, curved_10). Tại thời điểm thiết kế, phần dự đoán tràn sang hai bên chiếm khoảng 19% diện tích hợp, là nguồn sai lớn nhất.
- **Bỏ sót phần đường khác màu cửa sổ mẫu**: vùng bóng đổ, hoặc làn đường có màu nhựa khác (adverse_05, curved_08).
- **Lấn vào nắp capo** khi bộ dò mép capo (kế thừa từ notebook) sai, ví dụ curved_05.
- **Lấn lên cây và trời** ở ảnh ngược sáng (adverse_15).

## 8. Các hướng đã thử nhưng không giữ lại

Các con số dưới đây đo ở cấu hình trung gian, chỉ dùng để so sánh tương đối.

| Hướng thử | Kết quả | Lý do không giữ |
|---|---|---|
| Ứng viên tia qua điểm tụ (Hough 1 chiều theo góc), chọn cặp theo mặt nạ S-V | IoU 0,556 | Bằng chứng màu S-V không phân biệt được (mục 3) |
| Như trên nhưng chọn theo mặt nạ màu Lab | IoU 0,57–0,58 | Tăng ít; vẫn bị trần của hai đường thẳng |
| Mặt nạ màu Lab đưa vào khung quét hàng của notebook | IoU 0,52–0,56 | Không hơn notebook |
| Dò mép capo bằng quy hoạch động (đường cắt ngang có gradient dọc lớn nhất) | Sai số trung vị khoảng 40 px | Hay bám nhầm vào bóng xe, mép đường ở xa |
| Khởi tạo GrabCut chỉ với các pixel giống màu mẫu (Lab) | IoU 0,55–0,62 | Tốt cho ảnh sáng nhưng hỏng ảnh có bóng đổ và ảnh tối |
| Dùng đoạn Hough dài không phải vạch sơn làm "tường chắn" cho GrabCut | Không thay đổi | Mép bó vỉa hiếm khi liền mạch nên không tách được vùng tràn |
| Cắt vùng tràn theo tia Hough, chọn tia mà phần bị cắt khác màu mẫu | IoU giảm | Cắt nhầm vào vùng bóng đổ |
| CLAHE cho mọi ảnh trước GrabCut | IoU giảm ở ảnh sáng | CLAHE làm thay đổi màu cục bộ, nên chỉ áp cho ảnh tối |

*Bảng 6. Các hướng đã thử.*

## 9. Hạn chế

- **Chưa đạt IoU 0,7 trên toàn tập.** Normal đạt 0,744; curved và adverse còn 0,659 và 0,613.
- **Vai trò của Canny + Hough đã thay đổi.** Vùng đường giờ do GrabCut quyết định; Canny + Hough chỉ xác định hai biên từ viền mặt nạ. Ảnh cạnh cải thiện ở mục 4 chưa được đưa vào việc tạo vùng đường. Nếu đề tài yêu cầu Canny + Hough là trung tâm thì cần cân nhắc lại hướng này.
- **Không có tập kiểm tra riêng.** Các lựa chọn thiết kế được thử trên chính 50 ảnh. Để kiểm tra khớp quá mức, nhóm tách tập thành nửa chẵn và nửa lẻ theo thứ tự ảnh: IoU lần lượt 0,670 và 0,673 (notebook: 0,565 và 0,553). Kết quả này chưa thay thế được một tập kiểm tra thật.
- **Chậm.** Khoảng 4–5 s/ảnh trên CPU, chủ yếu do 3 lần GrabCut. Chưa đạt mục tiêu dưới 50 ms/khung của bài báo Belhadri.
- **MAE còn lớn** (105 px) và SR15 chỉ 8%.

## 10. Hướng tiếp theo

1. Đưa ảnh cạnh đã cải thiện (mục 4) vào GrabCut, ví dụ làm tăng chi phí cắt dọc theo cạnh Canny mạnh, để chặn vùng tràn sang vỉa hè.
2. Dò mép capo tốt hơn. Nếu biết đúng mép capo, IoU của GrabCut tăng khoảng +0,05 (đo bằng ground truth).
3. Thử mô hình màu bất biến bóng đổ (illuminant-invariant, Alvarez và López, 2011) cho các ảnh có tương phản cao.
4. Chia tập validation và test trước khi chỉnh thêm tham số.
5. Cập nhật DAG Airflow và notebook để dùng `road_pipeline.py`. Hiện cả hai vẫn chạy pipeline cũ.

## 11. Cách chạy và các file

```bash
pip install -r requirements.txt
python road_pipeline.py --edge-maps            # chạy 50 ảnh, đánh giá, ghi outputs/pipeline_v2/
python scripts/compare_canny_contrast.py       # Bảng 2
python scripts/ablation_pipeline_v2.py         # Bảng 7
python scripts/figures_pipeline_v2.py          # Hình 1–4
python -m pytest tests                         # 38 test
```

| File | Nội dung |
|---|---|
| `road_pipeline.py` | Pipeline v2: `normalize_contrast`, `road_prior`, `segment_road`, `boundary_lines`, `run_pipeline`, CLI chạy hàng loạt + đánh giá |
| `canny.py` | Thêm `gradient_percentile_thresholds`, `canny_edges_auto` |
| `evaluation.py` | Thêm `evaluate_road_mask` (IoU, precision, recall, F1, MAE, SR15 trên mặt nạ) |
| `tests/test_road_pipeline.py` | Test ngưỡng thích nghi, chuẩn hóa tương phản, GrabCut trên ảnh tổng hợp, đánh giá mặt nạ |
| `outputs/pipeline_v2/` | `road_mask/`, `overlay/`, `mask_edges/`, `edge_map/`, `results.csv`, `summary.csv`, `ablation*.csv`, `canny_contrast_by_condition.csv`, `figures/` |

## Tài liệu tham khảo

[1] Y. Belhadri, D. Mohamed, B. Alshaqaqi, and A. S. Baquhaizel, "Lane Boundary Detection in Dynamic Environments," *Rev. Roum. Sci. Techn. - Électrotechn. et Énerg.*, vol. 71, no. 2, pp. 299-304, 2026.

[2] S. Wang, S. Wu, X. Wang, and Z. Li, "A Canny Operator Road Edge Detection Method Based on Color Features," *Journal of Physics: Conference Series*, vol. 1629, 012018, 2020.

[3] M. D. E. Munajat, D. H. Widyantoro, and R. Munir, "Road Detection System Based on RGB Histogram Filterization and Boundary Classifier," *ICACSIS*, pp. 195-200, 2015.

[4] J. F. Canny, "A Computational Approach to Edge Detection," *IEEE Trans. PAMI*, vol. 8, no. 6, pp. 679-698, 1986.

[5] C. Rother, V. Kolmogorov, and A. Blake, "GrabCut: Interactive Foreground Extraction Using Iterated Graph Cuts," *ACM Trans. Graphics (SIGGRAPH)*, vol. 23, no. 3, pp. 309-314, 2004.

[6] K. Zuiderveld, "Contrast Limited Adaptive Histogram Equalization," in *Graphics Gems IV*, Academic Press, pp. 474-485, 1994.

[7] H. Kong, J.-Y. Audibert, and J. Ponce, "Vanishing Point Detection for Road Detection," *CVPR*, pp. 96-103, 2009.

[8] J. Fritsch, T. Kühnl, and A. Geiger, "A New Performance Measure and Evaluation Benchmark for Road Detection Algorithms," *ITSC*, pp. 1693-1700, 2013.

[9] J. M. Álvarez and A. M. López, "Road Detection Based on Illuminant Invariance," *IEEE Trans. ITS*, vol. 12, no. 1, pp. 184-193, 2011.
