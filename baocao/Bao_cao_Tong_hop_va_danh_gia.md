# BÁO CÁO TỔNG HỢP VÀ ĐÁNH GIÁ · PHÁT HIỆN BIÊN ĐƯỜNG BẰNG HSV, CANNY VÀ HOUGH

*Tổng hợp bốn phần của nhóm, chạy toàn bộ 50 ảnh và đối chiếu với ground truth*

## 1. Tóm tắt kết quả

Pipeline HSV → Canny → Hough được chạy trên toàn bộ 50 ảnh của bộ dữ liệu và so với ground truth vùng mặt đường. Kết quả chính:

- **Có vùng đường dự đoán (đa giác hợp lệ):** 27/50 ảnh (54,0%).
- **IoU trung bình trên toàn tập:** 0,1743 ± 0,1855. Nếu chỉ tính 27 ảnh có đa giác, IoU trung bình là 0,3227.
- **Không ảnh nào đạt IoU ≥ 0,7**; chỉ 2 ảnh đạt IoU từ 0,5 đến dưới 0,7.
- **Sai số vị trí biên (MAE):** chỉ đo được trên 10/50 ảnh, trung bình 272,40 ± 86,54 px.
- **Tỉ lệ thành công SR15 (MAE < 15 px):** 0,0%.
- **Thời gian Hough:** 3,374 ms/ảnh (chỉ riêng bước Hough và hậu xử lý).

Kết luận ngắn: chuỗi xử lý chạy trọn vẹn và tái lập được, nhưng độ chính xác so với ground truth còn thấp. Vùng đường dự đoán thường hẹp hơn mặt đường thật vì Hough bám vào vạch sơn bên trong lòng đường thay vì mép vật lý của đường (mục 6). Các con số trên là kết quả của cấu hình cố định, không phải kết quả đã tối ưu.

## 2. Bài toán và bộ dữ liệu

Theo Phần 1, đối tượng của đề tài là **road boundary**: ranh giới vật lý giữa lòng đường và vỉa hè, cỏ, lan can hoặc dải phân cách. Vạch sơn chia làn (lane boundary) nằm bên trong lòng đường và không phải đối tượng cần tìm. Vì vậy ground truth là mặt nạ nhị phân bao trọn toàn bộ lòng đường, kể cả các vạch kẻ.

Bộ dữ liệu gồm 50 ảnh chọn có chủ đích từ BDD100K và IDD. Kiểm tra lại trực tiếp trên thư mục `dataset/`: cả 50 ảnh đều có kích thước 1280 × 720, và mọi mặt nạ ground truth chỉ chứa hai giá trị 0 và 255.

| Nhóm | Số ảnh | Đặc trưng (theo Phần 1) |
|---|---:|---|
| Normal | 18 | Đường thẳng, đủ sáng, mép đường rõ; làm mốc đối sánh. |
| Curved | 12 | Khúc cua, biên phi tuyến; đo giới hạn của Hough đường thẳng. |
| Adverse | 20 | Ban đêm, chói đèn, bóng đổ, đường ướt, tuyết, xe che biên, giao lộ. |
| Tổng | 50 | |

*Bảng 1. Ba nhóm kiểm thử; số ảnh đếm lại từ dataset/images.*

![Ground truth nhóm Normal](dataset/visualizations/normal_tong_hop.jpg)
*Hình 1. Ground truth của 18 ảnh nhóm Normal: vùng xanh là mặt đường được gán nhãn, bao trọn cả vạch kẻ (dataset/visualizations).*

## 3. Pipeline tổng hợp

Bốn module do các thành viên xây dựng được nối thành một chuỗi. Mỗi module chỉ nhận đầu ra của module trước, ground truth chỉ được dùng ở bước đánh giá.

| Bước | Nội dung và tham số | Đầu ra | Nguồn |
|---|---|---|---|
| Tiền xử lý HSV | Resize 960 × 540 (Lanczos); giữ vạch trắng `S ≤ 80, V ≥ 170` và vạch vàng `15 ≤ H ≤ 40, S ≥ 70, V ≥ 90`; Gaussian bán kính 1,2; chuyển xám | Ảnh xám `uint8` | Phần 2 |
| Canny tự cài (NumPy) | Gauss 5 × 5, σ = 1,4; Sobel 3 × 3; NMS; ngưỡng kép (50, 150) + hysteresis | Ảnh cạnh 0/255 | Phần 3 |
| ROI + P-Hough | ROI hình thang 0,48H–0,98H, đáy trên 35% W, đáy dưới 98% W; ρ = 1 px, θ = 1°, threshold 25, minLineLength 25, maxLineGap 25 | Đoạn thẳng | Phần 4 |
| Hậu xử lý | Lọc độ dốc có giá trị tuyệt đối từ 0,3 đến 4; chọn cụm đồng thuận (dung sai 0,02W); WLS; đa giác dưới giao điểm hai biên | Hai biên + đa giác | Phần 4 |
| Đánh giá | IoU, MAE tại 3 hàng, SR15 (`evaluation.py`) | Bảng số liệu | Phần 1, 4 |

*Bảng 2. Các bước của pipeline và tham số đang dùng.*

Phần 3 đã kiểm tra bản Canny tự cài so với `cv2.Canny` cùng cấu hình: IoU giữa hai ảnh cạnh là 0,952 trung bình và 0,919 thấp nhất trên 50 ảnh.

Notebook `Pipeline_HSV_Canny_Hough.ipynb` chạy trọn chuỗi trên trong một file và cho phép chọn một hoặc nhiều ảnh đầu vào bằng hộp thoại. Khi chạy trên cả 50 ảnh, notebook tạo ra ảnh cạnh trùng khớp từng pixel với `outputs/canny/edges` (50/50 ảnh) và cho kết quả đánh giá trùng khớp hoàn toàn với `outputs/hough/hough_50_images_results.csv`.

## 4. Phương pháp đánh giá

Gọi P là tập pixel đường dự đoán (đa giác giữa hai biên) và G là tập pixel đường trong ground truth (Phần 1, mục 5):

$$IoU = \frac{|P \cap G|}{|P \cup G|}$$

Tại ba hàng ngang y = 60%, 75% và 90% chiều cao ảnh, sai số vị trí biên của một ảnh là:

$$MAE = \frac{1}{2K}\sum_{k=1}^{K}\left(|x^L_{pred}-x^L_{gt}| + |x^R_{pred}-x^R_{gt}|\right), \quad K = 3$$

Quy ước tính (Phần 4, mục 5; `evaluation.py`):

- IoU tính trên toàn ảnh và lấy trung bình trên **tất cả** ảnh. Ảnh không có đa giác hợp lệ được tính như dự đoán rỗng (IoU = 0), không bị loại khỏi mẫu số.
- MAE chỉ tính khi có đa giác hợp lệ, ground truth có pixel đường và hai biên phủ đủ cả ba hàng. Không ngoại suy và không thay giá trị thiếu bằng 0; luôn ghi kèm số ảnh đo được.
- SR15 là số ảnh có MAE < 15 px chia cho **toàn bộ** số ảnh. Ảnh không đo được MAE tính là không thành công.
- Tỉ lệ "bắt được biên" hoặc "có đa giác" chỉ cho biết pipeline có đưa ra kết quả, không phải độ chính xác.

## 5. Kết quả độ chính xác

### 5.1 Theo nhóm ảnh

| Nhóm | N | Bắt được biên trái | Bắt được biên phải | Có đa giác |
|---|---:|---:|---:|---:|
| Normal | 18 | 72,2% | 66,7% | 50,0% |
| Curved | 12 | 66,7% | 75,0% | 50,0% |
| Adverse | 20 | 80,0% | 75,0% | 60,0% |
| Toàn tập | 50 | 74,0% | 72,0% | 54,0% |

*Bảng 3. Tỉ lệ ảnh có đầu ra theo nhóm (chưa phải độ chính xác).*

| Nhóm | IoU toàn tập ± σ | IoU trên ảnh có đa giác | N đo MAE | MAE ± σ (px) | SR15 |
|---|---:|---:|---:|---:|---:|
| Normal | 0,1721 ± 0,1931 | 0,3443 | 1 | 219,13 ± 0,00 | 0,0% |
| Curved | 0,1919 ± 0,1998 | 0,3837 | 3 | 189,61 ± 23,89 | 0,0% |
| Adverse | 0,1656 ± 0,1680 | 0,2760 | 6 | 322,67 ± 75,95 | 0,0% |
| Toàn tập | 0,1743 ± 0,1855 | 0,3227 | 10 | 272,40 ± 86,54 | 0,0% |

*Bảng 4. Độ chính xác theo nhóm. Nguồn: outputs/hough/hough_group_statistics.csv, tính lại và khớp từ ground truth.*

### 5.2 Phân bố IoU

| Nhóm | Không có đa giác (IoU = 0) | 0 < IoU < 0,3 | 0,3 ≤ IoU < 0,5 | 0,5 ≤ IoU < 0,7 | IoU ≥ 0,7 |
|---|---:|---:|---:|---:|---:|
| Normal | 9 | 3 | 5 | 1 | 0 |
| Curved | 6 | 1 | 4 | 1 | 0 |
| Adverse | 8 | 5 | 7 | 0 | 0 |
| Toàn tập | 23 | 9 | 16 | 2 | 0 |

*Bảng 5. Số ảnh theo khoảng IoU.*

Năm ảnh có IoU cao nhất: normal_08 (0,608); curved_11 (0,542); adverse_07 (0,464); normal_03 (0,446); adverse_19 (0,423). Không ảnh nào trong số này đo được MAE (mục 6.4).

### 5.3 Theo điều kiện ánh sáng

Phần 3 đã gán mỗi ảnh một nhãn điều kiện theo độ sáng trung bình và độ lệch chuẩn của vùng lòng đường (`outputs/canny/conditions.csv`). Ghép nhãn này với kết quả Hough:

| Điều kiện | N | Số đoạn P-Hough TB | Có đa giác | IoU toàn tập ± σ | N đo MAE |
|---|---:|---:|---:|---:|---:|
| sáng | 7 | 99,7 | 85,7% | 0,2300 ± 0,1544 | 1 |
| tương phản cao | 15 | 74,1 | 66,7% | 0,2137 ± 0,1837 | 6 |
| vừa | 15 | 27,1 | 46,7% | 0,1783 ± 0,2020 | 2 |
| tối | 13 | 15,0 | 30,8% | 0,0941 ± 0,1536 | 1 |

*Bảng 6. Kết quả theo điều kiện ánh sáng.*

### 5.4 Thời gian xử lý

Theo Phần 4, bước Hough và hậu xử lý mất trung bình 3,374 ms/ảnh, không gồm tiền xử lý, Canny và đọc ghi ảnh. Riêng phép biến đổi, Standard Hough mất 2,886 ms và P-Hough mất 2,111 ms (trung bình các median trên 50 ảnh). Theo Phần 3, bản Canny NumPy tự cài mất khoảng 0,2 đến 0,3 giây mỗi ảnh, nên đây là bước chậm nhất của pipeline.

## 6. Đối chiếu với ground truth

![Đối chiếu dự đoán với ground truth](outputs/bao_cao_tong_hop/doi_chieu_ground_truth.png)
*Hình 2. Ba trường hợp: IoU cao nhất (normal_08), trung vị trong các ảnh có đa giác (adverse_01) và thất bại (normal_01). Vàng: ground truth; xanh lá: đa giác dự đoán; đỏ/xanh dương: biên trái/phải; ba đường trắng: các hàng đo MAE 60%, 75%, 90% H.*

### 6.1 Vùng dự đoán hẹp hơn mặt đường

Trên 27 ảnh có đa giác, tách IoU thành độ chính xác pixel (phần đa giác nằm trong mặt đường thật, |P ∩ G| / |P|) và độ phủ pixel (phần mặt đường thật được đa giác phủ, |P ∩ G| / |G|):

| Nhóm | Ảnh có đa giác | Độ chính xác pixel | Độ phủ pixel | Diện tích đa giác (% ảnh) | Diện tích ground truth (% ảnh) |
|---|---:|---:|---:|---:|---:|
| Normal | 9 | 0,647 | 0,446 | 16,0 | 22,9 |
| Curved | 6 | 0,687 | 0,500 | 18,6 | 23,6 |
| Adverse | 12 | 0,661 | 0,322 | 14,4 | 31,6 |
| Toàn tập | 27 | 0,662 | 0,403 | 15,9 | 26,9 |

*Bảng 7. Độ chính xác và độ phủ pixel của đa giác dự đoán.*

Đa giác dự đoán phần lớn nằm trong mặt đường (độ chính xác 0,662) nhưng chỉ phủ khoảng 40% mặt đường thật. IoU thấp chủ yếu do bỏ sót phần mặt đường hai bên, không phải do dự đoán tràn ra ngoài.

### 6.2 Biên dự đoán lệch vào trong lòng đường

Đo sai số có dấu tại ba hàng 60/75/90% H (dương nghĩa là biên dự đoán nằm bên trong vùng đường thật), trên các ảnh mà đường biên phủ đủ ba hàng: biên trái nằm bên trong ở 10/14 ảnh, lệch trung bình 120,9 px; biên phải nằm bên trong ở 9/12 ảnh, lệch trung bình 162,5 px.

Điều này khớp với nhận định của Phần 3 và Phần 4: bộ lọc HSV của Phần 2 được thiết kế để giữ vạch sơn trắng và vàng, nên mép vật lý của đường (nhựa đường giáp cỏ, đất, vỉa hè) thường bị loại trước khi đến Canny. Khi đó Hough thường bám vào vạch kẻ bên trong lòng đường. Ở ảnh normal_01 (Hình 2, dưới cùng), biên trái được đặt đúng lên vạch vàng giữa đường.

### 6.3 Nguyên nhân thiếu đa giác

23/50 ảnh không có đa giác. Trong đó 4 ảnh không tìm được biên nào, 9 ảnh thiếu biên trái, 10 ảnh thiếu biên phải, và 0 ảnh có đủ hai biên nhưng đa giác bị loại. Như vậy mọi trường hợp thiếu đa giác đều do bước Hough không tìm đủ hai biên.

### 6.4 Vì sao chỉ đo được MAE trên ít ảnh

Trong 27 ảnh có đa giác, 10 ảnh đo được MAE. 17 ảnh còn lại không đo được vì: 5 ảnh có đỉnh đa giác nằm dưới hàng 60% H (hai biên giao nhau sớm); 8 ảnh có đáy đa giác nằm trên hàng 90% H (biên bị cắt tại mép trái hoặc phải của khung ảnh); 4 ảnh (normal_10, normal_11, curved_09, adverse_14) có ground truth không chứa pixel đường ở một trong ba hàng đo. Vì vậy MAE 272,40 px chỉ phản ánh 10 ảnh và phải đọc cùng IoU toàn tập.

## 7. Thảo luận

- **Có kết quả không đồng nghĩa với đúng.** Nhóm Adverse có tỉ lệ đa giác cao nhất (60,0%) nhưng IoU trên ảnh có đa giác thấp nhất (0,2760).
- **Khác biệt giữa ba nhóm nhỏ hơn độ lệch chuẩn.** IoU toàn tập của ba nhóm nằm trong khoảng 0,1656 đến 0,1919, trong khi σ từ 0,17 đến 0,20. Với 12 đến 20 ảnh mỗi nhóm, chưa thể kết luận nhóm nào khó hơn.
- **Ánh sáng ảnh hưởng rõ nhất.** Từ ảnh sáng sang ảnh tối, số đoạn P-Hough trung bình giảm từ 99,7 xuống 15,0, tỉ lệ có đa giác giảm từ 85,7% xuống 30,8% và IoU giảm từ 0,2300 xuống 0,0941. Điều này phù hợp với Phần 3: ảnh tối có rất ít điểm cạnh vì ngưỡng V cố định của bộ lọc HSV loại gần hết pixel.
- **Tham số Hough còn dư địa.** Phần 4 khảo sát 27 tổ hợp tham số: cấu hình mặc định (25, 25, 25) cho IoU 0,1743, cấu hình tốt nhất trong lưới (15, 20, 25) cho IoU 0,2579. Lưới chạy trên chính tập đánh giá, không có tập kiểm tra riêng, nên đây chỉ là khảo sát độ nhạy.
- **Mục tiêu của bộ lọc màu và của ground truth khác nhau.** Phần 2 lọc màu vạch làn, trong khi ground truth đánh dấu toàn bộ lòng đường. Phần 2 cũng đã lưu ý không dùng ground truth vùng mặt đường để kết luận độ chính xác phát hiện làn.

## 8. Kết luận và hướng phát triển

Nhóm đã xây dựng hoàn chỉnh pipeline HSV → Canny → Hough, có bộ dữ liệu 50 ảnh kèm ground truth và quy trình đánh giá định lượng tái lập được. Với cấu hình hiện tại, pipeline tạo được vùng đường cho 54,0% số ảnh, IoU trung bình toàn tập 0,1743 và chưa ảnh nào đạt MAE dưới 15 px. Nguyên nhân chính là biên tìm được bám vào vạch sơn bên trong thay vì mép đường, cộng với việc thiếu biên ở ảnh tối.

Các hướng cải thiện, lấy từ đề xuất của các phần trước:

- Chạy thêm nhánh ảnh xám không lọc HSV qua cùng Canny và Hough để kiểm tra việc lọc trắng/vàng có làm mất mép đường hay không, như Phần 3 đề xuất; Phần 4 cũng ghi nhận giới hạn này.
- Dùng ngưỡng tự động theo trung vị hoặc Otsu cho Canny để giảm mất cạnh ở ảnh tối (Phần 3).
- Thay mô hình một đường thẳng mỗi bên bằng mô hình phân đoạn hoặc spline cho khúc cua (Phần 1, Phần 4).
- Chia tập validation và test riêng trước khi chỉnh ngưỡng HSV và tham số Hough (Phần 2, Phần 4).

## 9. Đối chiếu nguồn và tự soát

Mọi con số trong báo cáo được sinh tự động từ dữ liệu bằng `scripts/analyze_tong_hop.py` và `scripts/build_tong_hop_docx.py`, không gõ tay. Script phân tích chạy lại Hough trên 50 ảnh cạnh và tính lại metric từ ground truth, rồi dừng lại nếu bất kỳ giá trị nào lệch với CSV chính thức của Phần 4.

| Nội dung | Nguồn | Cách kiểm tra |
|---|---|---|
| Số ảnh, kích thước, giá trị mask | `dataset/images`, `dataset/ground_truth` | Đọc trực tiếp từng file |
| Bảng 3, 4 | `outputs/hough/hough_group_statistics.csv` | Tính lại từ ground truth, khớp tuyệt đối |
| Bảng 5, 7, mục 6.2–6.4 | `outputs/bao_cao_tong_hop/per_image.csv` | Tính từ kết quả Hough và ground truth |
| Bảng 6 | `outputs/canny/conditions.csv` (Phần 3) ghép với kết quả Hough | Ghép theo tên ảnh |
| Tham số và thời gian | Báo cáo Phần 2, 3, 4; `run_metadata.json`; `hough_method_benchmark.csv` | So với mã nguồn và CSV |
| Notebook tổng hợp | `outputs/pipeline_notebook/results.csv` | So từng ảnh với `hough_50_images_results.csv` |

*Bảng 8. Nguồn số liệu.*

Các điểm chưa thống nhất giữa tài liệu và dữ liệu, phát hiện khi đối chiếu (giữ nguyên tài liệu gốc, chỉ ghi nhận):

- Phần 1 mô tả tên file ground truth dạng `adverse_01_mask.png`, nhưng file thực tế là `dataset/ground_truth/adverse/adverse_01.png`.
- Phần 1 mô tả bước tiền xử lý là "lọc màu mặt đường", trong khi Phần 2 lọc màu vạch sơn trắng và vàng.
- Phần 1 định nghĩa nhóm Normal là ảnh đủ sáng, nhưng nhãn độ sáng của Phần 3 xếp 2 ảnh Normal vào điều kiện tối, và Hình 1 có một ảnh Normal chụp ban đêm.

## Phụ lục A. Kết quả từng ảnh

| Ảnh | Điều kiện | Biên trái | Biên phải | Đa giác | IoU | MAE (px) |
|---|---|:---:|:---:|:---:|---:|---:|
| normal_01 | vừa | Có | Không | Không | 0,000 | – |
| normal_02 | vừa | Có | Không | Không | 0,000 | – |
| normal_03 | sáng | Có | Có | Có | 0,446 | – |
| normal_04 | sáng | Có | Có | Có | 0,314 | – |
| normal_05 | tương phản cao | Có | Có | Có | 0,322 | 219,1 |
| normal_06 | vừa | Có | Có | Có | 0,386 | – |
| normal_07 | tương phản cao | Có | Không | Không | 0,000 | – |
| normal_08 | vừa | Có | Có | Có | 0,608 | – |
| normal_09 | vừa | Không | Không | Không | 0,000 | – |
| normal_10 | tương phản cao | Có | Có | Có | 0,227 | – |
| normal_11 | sáng | Có | Có | Có | 0,206 | – |
| normal_12 | vừa | Có | Có | Có | 0,387 | – |
| normal_13 | tối | Không | Có | Không | 0,000 | – |
| normal_14 | sáng | Có | Có | Có | 0,202 | – |
| normal_15 | tối | Không | Không | Không | 0,000 | – |
| normal_16 | tương phản cao | Không | Có | Không | 0,000 | – |
| normal_17 | tương phản cao | Không | Có | Không | 0,000 | – |
| normal_18 | vừa | Có | Không | Không | 0,000 | – |
| curved_01 | vừa | Không | Có | Không | 0,000 | – |
| curved_02 | vừa | Không | Có | Không | 0,000 | – |
| curved_03 | tối | Có | Không | Không | 0,000 | – |
| curved_04 | sáng | Có | Có | Có | 0,393 | 164,0 |
| curved_05 | tương phản cao | Không | Có | Không | 0,000 | – |
| curved_06 | tối | Có | Không | Không | 0,000 | – |
| curved_07 | tương phản cao | Có | Có | Có | 0,370 | 221,5 |
| curved_08 | vừa | Có | Có | Có | 0,293 | – |
| curved_09 | vừa | Có | Có | Có | 0,321 | – |
| curved_10 | tối | Không | Không | Không | 0,000 | – |
| curved_11 | tương phản cao | Có | Có | Có | 0,542 | – |
| curved_12 | tương phản cao | Có | Có | Có | 0,383 | 183,3 |
| adverse_01 | vừa | Có | Có | Có | 0,323 | 229,6 |
| adverse_02 | tối | Có | Có | Có | 0,129 | 374,2 |
| adverse_03 | vừa | Không | Không | Không | 0,000 | – |
| adverse_04 | tương phản cao | Có | Có | Có | 0,172 | 336,1 |
| adverse_05 | tương phản cao | Có | Có | Có | 0,218 | 344,8 |
| adverse_06 | tối | Có | Có | Có | 0,356 | – |
| adverse_07 | tương phản cao | Có | Có | Có | 0,464 | – |
| adverse_08 | tương phản cao | Có | Có | Có | 0,117 | 432,0 |
| adverse_09 | sáng | Có | Có | Có | 0,050 | – |
| adverse_10 | tương phản cao | Có | Không | Không | 0,000 | – |
| adverse_11 | tối | Có | Không | Không | 0,000 | – |
| adverse_12 | tối | Không | Có | Không | 0,000 | – |
| adverse_13 | sáng | Không | Có | Không | 0,000 | – |
| adverse_14 | tương phản cao | Có | Có | Có | 0,391 | – |
| adverse_15 | tối | Có | Không | Không | 0,000 | – |
| adverse_16 | tối | Không | Có | Không | 0,000 | – |
| adverse_17 | tối | Có | Có | Có | 0,315 | – |
| adverse_18 | vừa | Có | Có | Có | 0,355 | 219,4 |
| adverse_19 | tối | Có | Có | Có | 0,423 | – |
| adverse_20 | vừa | Có | Không | Không | 0,000 | – |

*Bảng A.1. Kết quả của 50 ảnh. Nguồn: outputs/hough/hough_50_images_results.csv.*

## Tài liệu tham khảo

[1] M. D. E. Munajat, D. H. Widyantoro, and R. Munir, "Road Detection System Based on RGB Histogram Filterization and Boundary Classifier," *ICACSIS*, pp. 195-200, 2015.

[2] S. Wang, S. Wu, X. Wang, and Z. Li, "A Canny Operator Road Edge Detection Method Based on Color Features," *Journal of Physics: Conference Series*, vol. 1629, 012018, 2020.

[3] Y. Belhadri, D. Mohamed, B. Alshaqaqi, and A. S. Baquhaizel, "Lane Boundary Detection in Dynamic Environments," *Rev. Roum. Sci. Techn. - Électrotechn. et Énerg.*, vol. 71, no. 2, pp. 299-304, 2026.

[4] J. F. Canny, "A Computational Approach to Edge Detection," *IEEE Trans. Pattern Analysis and Machine Intelligence*, vol. 8, no. 6, pp. 679-698, 1986.

[5] F. Yu et al., "BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning," *CVPR*, 2020.

[6] G. Varma et al., "IDD: A Dataset for Exploring Problems of Autonomous Navigation in Unconstrained Environments," *WACV*, 2019.

[7] Báo cáo Phần 1 đến Phần 4 của nhóm, thư mục `baocao/`.
