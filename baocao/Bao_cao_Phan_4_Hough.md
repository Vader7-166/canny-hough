# BÁO CÁO PHẦN 4 · HOUGH VÀ HẬU XỬ LÝ ROAD BOUNDARY

*Phạm vi phần 4: giữ nguyên ba module trước và dùng đúng ảnh cạnh Canny đã bàn giao*

## 1. Mục tiêu và quan hệ với ba phần trước

Phần 1 định nghĩa road boundary là mép vật lý của toàn bộ lòng đường, không phải vạch chia làn. Ground truth là mask vùng đường 0/255. Các vạch kẻ bên trong không thu hẹp vùng đường. Hough chỉ dùng ảnh cạnh và cấu hình; ground truth không tham gia chọn biên.

Phần 2 hiện có lọc trắng/vàng trong HSV trước Gaussian và chuyển xám. Phần 3 nhận ảnh đó rồi dùng Canny NumPy với ngưỡng (50,150), σ=1,4, nhân 5×5, L2gradient theo quy ước bàn giao. Bản sửa phần 4 không đổi các module này, không tạo lại ảnh đầu vào và không thay nhánh tiền xử lý.

Đầu vào Hough là `outputs/canny/edges/{normal,curved,adverse}/*.png`, uint8 một kênh, 0/255. Ảnh RGB để vẽ lấy từ `outputs/preprocessed_for_canny/resized`; ground truth lấy từ `dataset/ground_truth`, chỉ dùng đánh giá. Batch và notebook Hough chỉ đọc các thư mục này, chỉ ghi kết quả phần 4.

Lọc trắng/vàng có thể xóa mép nhựa đường, lề đất hoặc cỏ không thuộc nhóm màu đó. Đây là giới hạn đầu vào phải ghi nhận: Hough không thể khôi phục cạnh đã mất. Không tự thay pipeline của module trước để che giới hạn này.

## 2. ROI và biến đổi Hough

ROI hình thang cố định theo tỷ lệ, không phải ROI thích ứng hay bộ ước lượng chân trời. Mép trên 0,48H và đáy 0,98H là giả định thực nghiệm của dự án. Chỉ số pixel được chặn trong [0,H−1] và [0,W−1]. Mask áp dụng sau Canny nên phép AND không tự sinh cạnh mới ở đường bao mask; nó chỉ giữ/xóa pixel cạnh đã có.

P-Hough trả đoạn hữu hạn [x1,y1,x2,y2]; Standard Hough trả (rho,theta) của đường vô hạn. P-Hough được dùng vì giao diện đoạn thẳng phù hợp lọc và chọn cụm, không phải vì có bảo đảm luôn nhanh hơn Standard Hough.

| Tham số | Giá trị |
| --- | --- |
| roi_top_width_ratio | 0.35 |
| roi_bottom_width_ratio | 0.98 |
| roi_top_y_ratio | 0.48 |
| roi_bottom_y_ratio | 0.98 |
| rho | 1.0 |
| theta | 0.017453292519943295 |
| threshold | 25 |
| min_line_length | 25.0 |
| max_line_gap | 25.0 |
| min_abs_slope | 0.3 |
| max_abs_slope | 4.0 |
| fit_distance_ratio | 0.02 |

Các giá trị này là cấu hình khởi điểm của nhóm, không được ba bài báo công bố như một bộ tham số dành cho tập dữ liệu hiện tại.

![So sánh hai biến đổi](outputs/hough/figures/figure_2_standard_vs_probabilistic.png)
*Hình 4.1. Đường vô hạn của Standard Hough và đoạn hữu hạn của P-Hough trên cùng ROI.*

## 3. Lọc ứng viên và chọn cụm đồng thuận

Trong tọa độ ảnh y hướng xuống, baseline giả định biên trái có slope âm và biên phải có slope dương. Loại đoạn ngang, quá đứng và sai vị trí: trái có x_mid < 0,65W, phải có x_mid > 0,35W. Đây là giả định cho cảnh đường gần thẳng, không phải quy luật cho mọi khúc cua.

Không gộp mọi đoạn cùng dấu vào một hồi quy. Mỗi đoạn đề xuất mô hình x=k·y+b; một đoạn khác chỉ là inlier nếu cả hai đầu mút cách mô hình không quá 0,02W pixel theo phương vuông góc. Chọn mô hình có tổng chiều dài inlier lớn nhất; hòa thì chọn sai số nhỏ nhất, với thứ tự đoạn chuẩn hóa để kết quả có tính xác định.

Chọn cụm đồng thuận xử lý việc trộn các cấu trúc khác vị trí. Nó chưa phân biệt ngữ nghĩa mép đường với vạch làn, thân xe hoặc bóng: một cụm nhiễu nhất quán vẫn có thể được chọn.

## 4. WLS và hình học đầu ra

Chỉ fit đầu mút của các inlier bằng WLS x=k·y+b. Mỗi đầu mút nhận trọng số bằng chiều dài đoạn chứa nó:

$$k = \frac{\sum_i w_i(y_i-\bar y)(x_i-\bar x)}{\sum_i w_i(y_i-\bar y)^2}, \qquad b=\bar x-k\bar y$$

Kiểm tra lại dấu và khoảng độ dốc sau fit; không xuất đường đảo hướng. Mỗi đường được cắt theo khung ảnh bằng cv2.clipLine thay vì ép x vào biên ảnh làm đổi hướng đường.

Đa giác chỉ dùng khoảng y chung của hai đường. Nếu hai đường giao nhau trong khoảng đang xét, đỉnh vùng đường được dời xuống dưới giao điểm, sao cho biên trái luôn nằm trước biên phải và có bề rộng dương. Nếu thứ tự sai ở đáy hoặc không còn diện tích chung thì polygon=None. Hai đường xuất cùng polygon được rút về cùng miền y. Cách này ngăn đa giác hình nơ, không chứng minh biên được chọn là đúng về ngữ nghĩa.

![Các bước Hough](outputs/hough/figures/figure_1_hough_pipeline_stages.png)
*Hình 4.2. Canny → ROI → ứng viên slope → consensus/WLS → vùng đường hợp lệ.*

## 5. Quy ước đánh giá

IoU chính tính trên toàn ảnh và trung bình trên tất cả ảnh. Thiếu hoặc sai hình học polygon được xem như dự đoán mask rỗng, không bị loại khỏi mẫu số. Không cắt ground truth theo ROI để tăng điểm. IoU trên nhóm có polygon chỉ là thống kê phụ, ghi riêng trong CSV.

MAE tuân theo phần 1: lấy x trái nhất/phải nhất của ground truth tại ba hàng 60%,75%,90% H. Chỉ tính MAE ảnh nếu có cả hai đường, polygon hợp lệ, đủ ground truth và đủ miền y tại cả ba hàng. Không ngoại suy qua phần đã cắt và không thay lỗi thiếu bằng 0. Luôn báo N đo MAE bên cạnh trung bình ± độ lệch chuẩn (ddof=0).

SR15 là số ảnh có MAE < 15px chia cho toàn bộ số ảnh. Thiếu biên hoặc thiếu mặt cắt tính là không thành công. Các tỷ lệ left/right/both/polygon available chỉ đo có đầu ra, không được gọi là độ chính xác.

| Nhóm | N | Có polygon (%) | IoU toàn tập ± σ | N đo MAE | MAE ± σ (px) | SR15 (%) |
| --- | --- | --- | --- | --- | --- | --- |
| normal | 18 | 50.0 | 0.1721 ± 0.1931 | 1 | 219.13 ± 0.00 | 0.0 |
| curved | 12 | 50.0 | 0.1919 ± 0.1998 | 3 | 189.61 ± 23.89 | 0.0 |
| adverse | 20 | 60.0 | 0.1656 ± 0.1680 | 6 | 322.67 ± 75.95 | 0.0 |
| all | 50 | 54.0 | 0.1743 ± 0.1855 | 10 | 272.40 ± 86.54 | 0.0 |

Số liệu trên là kết quả mô hình đường thẳng với cấu hình cố định, không phải kết quả tối ưu hoặc chứng minh tổng quát hóa. MAE có thể dựa trên một tập con nhỏ và không được đọc tách khỏi coverage, IoU toàn tập và SR15.

## 6. Khảo sát tham số Hough trên cùng đầu vào

Mọi cấu hình dưới đây dùng nguyên cùng ảnh cạnh Canny đã bàn giao. Chỉ thay tham số P-Hough; không chạy lại Canny, đổi ngưỡng HSV hoặc tạo nhánh đầu vào khác. Vì vậy đây là khảo sát phần 4, không phải kết luận cải tiến module tiền xử lý/Canny.

Lưới 27 tổ hợp: threshold ∈ {15,25,35}, minLineLength ∈ {20,25,45}, maxLineGap ∈ {15,25,35}. Lưới có cấu hình mặc định (25,25,25). Bảng dưới gồm mặc định và tối đa bốn cấu hình có IoU toàn tập cao nhất trong khảo sát; toàn bộ dữ liệu ở hough_parameters_grid.csv.

| threshold | minLineLength | maxLineGap | IoU toàn tập | N đo MAE | MAE (px) |
| --- | --- | --- | --- | --- | --- |
| 25 | 25 | 25 | 0.1743 | 10 | 272.40 |
| 15 | 20 | 25 | 0.2579 | 14 | 239.86 |
| 15 | 25 | 25 | 0.2506 | 13 | 243.68 |
| 15 | 20 | 35 | 0.2440 | 13 | 242.64 |
| 15 | 25 | 35 | 0.2422 | 12 | 252.36 |

Lưới chạy trên cùng tập ảnh để khảo sát độ nhạy, không có tập held-out. Không dùng nó tự động thay mặc định hay tuyên bố cấu hình “tối ưu”. Muốn chọn tham số cần chia validation/test độc lập trước khi hiệu chỉnh.

![Độ nhạy tham số](outputs/hough/figures/figure_3_parameter_grid_sensitivity.png)
*Hình 4.3. IoU toàn tập trên lưới chứa cả cấu hình mặc định.*

## 7. Phạm vi đo thời gian

Runtime Hough trung bình: 3.374 ms/ảnh. Chỉ gồm ROI, P-Hough, consensus, WLS và tạo polygon; không gồm tiền xử lý, Canny hoặc đọc/ghi ảnh. Không quy đổi thành FPS của toàn pipeline.

Benchmark riêng từng biến đổi: mỗi ảnh một lượt warmup cho mỗi phương pháp, sau đó median năm lượt trên cùng ROI; bảng là trung bình các median ảnh. Đếm đường Standard và đoạn P-Hough không phải phép so sánh độ chính xác tương đương.

| Phương pháp | TB median mỗi ảnh (ms) | Số ảnh |
| --- | --- | --- |
| Standard Hough | 2.886 | 50 |
| P-Hough | 2.111 | 50 |

Môi trường đo: Python 3.14.3, OpenCV 4.13.0, NumPy 2.5.2; Windows-11-10.0.26200-SP0. Runtime phụ thuộc máy và tải hệ thống; không lấy tốc độ từ bài báo để gán cho triển khai này.

## 8. Ca lỗi và giới hạn còn lại

![Ví dụ có ground truth](outputs/hough/figures/figure_4_group_examples.png)
*Hình 4.4. Biên dự đoán và polygon; đường bao ground truth màu vàng.*

- Mô hình một đường thẳng mỗi bên không mô tả đầy đủ khúc cua; baseline có thể bỏ biên đổi dấu slope hoặc gần đứng.
- ROI cố định có thể loại mép xa, đường lệch tâm hoặc đoạn cua ra ngoài hình thang.
- Consensus chỉ loại các đoạn không nhất quán, không tự phân loại road/non-road. Vạch làn, xe đỗ, rào chắn và bóng chéo vẫn có thể tạo cụm sai.
- Tiền xử lý trắng/vàng hiện có có thể xóa mép vật lý trước Canny; ngưỡng Canny cố định còn có thể bỏ mép tương phản thấp. Phần 4 chỉ ghi nhận giới hạn, không thay hoặc chỉnh lại hai module đó.
- Mặt cắt ground truth chạm khung ảnh và các vật cản phản ánh quy ước mask có sẵn. Không thay nhãn để phù hợp đường dự đoán.
- Cần tập validation/test độc lập và, nếu mở rộng phạm vi, mô hình phân đoạn theo chiều cao hoặc đường cong. Các kỹ thuật này không được giả định là đã có trong bản hiện tại.

## 9. Tái lập và bàn giao

```text
python scripts/generate_hough_assets.py
python scripts/build_hough_report.py
python scripts/compile_latex_pdf.py
python -m unittest discover -s tests -v
```

Lệnh đầu chỉ đọc đầu ra Canny hiện có, đánh giá Hough trên toàn bộ ảnh và xuất hình/CSV trong outputs/hough. Không chạy lại tiền xử lý hoặc Canny. Hough_Colab.ipynb gọi cùng hough.py và evaluation.py; có thể tạo lại notebook Hough bằng scripts/build_hough_notebook.py. Hai notebook và các tài liệu của module trước được giữ nguyên.

CSV và notebook chạy được trên Colab khi cài các thư viện NumPy, Pillow, OpenCV, pandas và matplotlib. Bước xuất PDF hiện dùng Microsoft Word trên Windows qua render_docx_with_word.ps1; không yêu cầu tải KaTeX/CDN. DOCX và PDF cùng đọc nguồn Markdown, không có bản HTML chứa số liệu độc lập.

Giao diện: detect_road_boundaries(edges, config) trả HoughResult với left_line/right_line hữu hạn hoặc None; polygon (4,2) hợp lệ hoặc None; raw_lines; left_segments/right_segments là các ứng viên sau lọc slope, không phải toàn bộ đều là inlier của fit; roi_mask. evaluation.evaluate_road_result và summarize_records dùng chung cho batch và notebook.

Các CSV hough_50_images_results, hough_group_statistics, hough_parameters_grid và hough_method_benchmark cùng run_metadata.json nằm trong outputs/hough. Metadata ghi đường dẫn đầu vào chỉ đọc và cấu hình tiền xử lý được bàn giao, không gán cho Hough trách nhiệm sinh các dữ liệu đó. Mã kiểm thử hồi quy bao phủ đảo hướng khi gộp đoạn, outlier khác vị trí, đa giác tự cắt, miền y chung, thiếu biên và mẫu số metric.

## Tài liệu tham khảo

[1] Munajat et al. (2015), Road Detection System based on RGB Histogram Filterization and Boundary Classifier, ICACSIS, 195–200. Tham khảo mục tiêu road/non-road và giới hạn hình học; không tuyên bố đã tái hiện toàn bộ bộ phân loại/spline của bài.

[2] Wang et al. (2020), A Canny operator road edge detection method based on color features, Journal of Physics: Conference Series 1629, 012018. Tham khảo chuỗi HSV/Canny/Hough; ngưỡng HSV và Hough trong dự án là thực nghiệm.

[3] Belhadri et al. (2026), Lane Boundary Detection in Dynamic Environments, RRST Électrotechn. et Énerg. 71, 299–304. Bài nghiên cứu lane boundary; không dùng nhãn road-area để chứng minh cải thiện lane boundary.
