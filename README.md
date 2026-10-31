# Phát hiện biên đường bằng Canny + Hough

Đồ án xử lý ảnh: tìm **hai biên trái/phải của mặt đường** (road boundary, tức ranh giới lòng đường với vỉa hè, cỏ, dải phân cách; không phải vạch chia làn) trong ảnh camera hành trình bằng xử lý ảnh cổ điển, không dùng học sâu. Pipeline là biến thể của Belhadri et al. (2026): chèn bước lọc màu HSV trước Canny.

```
Ảnh gốc 1280×720
  ─► [P2] Resize 960×540 → HSV → lọc vạch trắng/vàng → Gaussian → ảnh xám
  ─► [P3] Canny tự cài (NumPy) → ảnh cạnh 0/255
  ─► [P4] ROI hình thang → HoughLinesP → lọc độ dốc, tách trái/phải → hồi quy WLS → 2 biên + đa giác mặt đường
  ─► [P5] Ghép pipeline, đo IoU / MAE, phân tích lỗi, demo   (chưa làm)
```

## Tiến độ

Nhóm 5 người, mỗi người một phần (xem `Phân việc.jpg`).

| Phần | Nội dung | Trạng thái | Code | Báo cáo |
|---|---|---|---|---|
| 1. Đề tài & dữ liệu | 50 ảnh, ground truth, định nghĩa IoU/MAE | ✅ Xong | — | `baocao/Bao_cao_Phan_1_De_tai_va_du_lieu.docx` |
| 2. Tiền xử lý | Resize, HSV, lọc màu, Gaussian | ✅ Xong | `preprocessing.py` | `baocao/Bao_cao_Phan_2_Tien_xu_ly_HSV.{md,docx}` |
| 3. Canny | 4 bước Canny, quét ngưỡng theo điều kiện sáng | ✅ Xong | `canny.py` | `baocao/Bao_cao_Phan_3_Canny.{md,docx}` |
| 4. Hough & hậu xử lý | ROI, P-Hough, tách trái/phải, gộp biên | ✅ Xong | `hough.py` | `baocao/Bao_cao_Phan_4_Hough.pdf`, `baocao/Bao_cao_Phan_4_Hough_Hau_xu_ly.pdf` |
| 5. Tích hợp & đánh giá | Pipeline end-to-end, metric toàn tập, so sánh có/không lọc màu, phân tích lỗi, demo | ⏳ Chưa bắt đầu | — | — |

## Cấu trúc thư mục

```
├── preprocessing.py        # P2: tiền xử lý HSV (NumPy + Pillow), có CLI batch
├── canny.py                # P3: Canny tự cài bằng NumPy (Gauss, Sobel, NMS, ngưỡng kép + hysteresis)
├── hough.py                # P4: ROI + HoughLinesP + hậu xử lý (OpenCV)
├── tests/                  # 18 unittest: preprocessing (4), canny (5), hough (9)
├── Tien_xu_ly_HSV_Colab.ipynb, Canny_Colab.ipynb, Hough_Colab.ipynb
│                           # Notebook từng phần; tự định nghĩa lại hàm (bản sao của các file .py)
├── scripts/                # Sinh notebook, hình, báo cáo .docx/.pdf (không thuộc pipeline)
├── dataset/
│   ├── images/{normal,curved,adverse}/        # 50 ảnh .jpg 1280×720 (18 / 12 / 20), từ BDD100K và IDD
│   ├── ground_truth/{normal,curved,adverse}/  # 50 mask .png cùng tên: 255 = lòng đường, 0 = nền
│   └── visualizations/                        # mỗi nhóm một ảnh lưới: ảnh gốc tô mask ground truth
├── outputs/
│   ├── preprocessed_for_canny/  # đầu ra P2 (canny_input/ là đầu vào P3)
│   ├── preprocessing_figures/   # hình + thống kê P2
│   ├── canny/                   # edges/ (đầu vào P4), figures/, các CSV thí nghiệm
│   ├── hough/                   # visualizations/, figures/, các CSV kết quả
│   └── docx_render/             # bản PDF render từ báo cáo P2
├── baocao/                 # Báo cáo từng phần + hough_road_boundary_tom_tat.md
├── canny_road_boundary_tom_tat.pdf, hough_road_boundary_tom_tat.pdf   # tài liệu tóm tắt cho người làm P3, P4
└── Article/                # 3 bài báo tham khảo (Belhadri 2026, Wang 2020, Munajat 2015)
```

## Chi tiết từng module

### Phần 1 — Dữ liệu và metric
- 3 nhóm: **normal** (đường thẳng, đủ sáng), **curved** (khúc cua), **adverse** (đêm, chói, bóng đổ, ướt, tuyết, bị che).
- **IoU** giữa đa giác mặt đường dự đoán và mask ground truth.
- **MAE** (pixel): sai lệch vị trí biên trái/phải tại 3 hàng y = 60%, 75%, 90% chiều cao ảnh, trung bình trên 2K = 6 giá trị.

### Phần 2 — `preprocessing.py`
- `preprocess_rgb(rgb, PreprocessConfig)` → `PreprocessResult` (resized, hsv, color_mask, color_filtered, gaussian, `canny_input`).
- HSV theo quy ước OpenCV (H ∈ [0,179]). Vạch trắng: S ≤ 80, V ≥ 170. Vạch vàng: H 15–40, S ≥ 70, V ≥ 90. Gaussian radius 1.2.
- Đầu ra: ảnh xám uint8 960×540 tại `outputs/preprocessed_for_canny/canny_input/`.

### Phần 3 — `canny.py`
- `canny_edges(img, low, high, sigma, size=5, l2=True)` → ảnh uint8 0/255. So với `cv2.Canny`, IoU ảnh cạnh trung bình là 0.952.
- Ảnh cạnh bàn giao (`outputs/canny/edges/`) tạo bằng bản NumPy với σ = 1.4, nhân 5×5, ngưỡng (50, 150).
- Các thí nghiệm quét tham số (σ, nhân, cặp ngưỡng, median, Otsu) chạy bằng `cv2.Canny` cho nhanh. Gợi ý: ảnh sáng/tương phản cao dùng σ = 2.0, nhân 7×7; ảnh tối dùng σ = 1.0 và ngưỡng thấp.
- Bản NumPy mất ~0.2–0.3 s/ảnh, không đạt thời gian thực (mục tiêu < 50 ms/khung).

### Phần 4 — `hough.py`
- `detect_road_boundaries(edges, HoughConfig)` → `HoughResult` (left_line, right_line, polygon, raw_lines, left/right_segments, roi_mask).
- ROI hình thang: cạnh trên ở 0.48H rộng 0.35W, cạnh dưới ở 0.98H rộng 0.98W.
- HoughLinesP: rho = 1, theta = 1°, threshold = 25, minLineLength = 25, maxLineGap = 25.
- Lọc |độ dốc| trong [0.30, 4.00]; trái: m < 0 và trung điểm < 0.65W; phải: m > 0 và trung điểm > 0.35W.
- Gộp mỗi bên bằng hồi quy `x = k·y + b` có trọng số theo độ dài đoạn; đa giác chỉ có khi tìm được cả hai biên.
- `draw_hough_visualization(...)` vẽ kết quả lên ảnh.

## Kết quả hiện có

**Canny**: đánh giá bằng chỉ số tạm *recall*, là tỉ lệ điểm trên biên mask lòng đường có cạnh Canny trong bán kính 5 px. Kết quả phụ thuộc mạnh vào điều kiện sáng: với ngưỡng (50, 150), recall là 0.632 ở ảnh sáng và 0.090 ở ảnh tối. Trên chỉ số này, ảnh **không** lọc HSV có recall cao hơn: median 0.431 so với 0.284, Otsu 0.390 so với 0.309, cố định (50, 150) 0.314 so với 0.277. Báo cáo P3 cho rằng nguyên nhân là lọc màu giữ vạch sơn nhưng làm mất mép lòng đường, giải thích này chưa được kiểm chứng riêng. Lọc HSV có giúp pipeline cuối hay không vẫn phải chờ Phần 5 chạy cả hai nhánh qua Hough.

**Hough** với cấu hình mặc định trên 50 ảnh (`outputs/hough/hough_group_statistics.csv`). IoU và MAE chỉ tính trên các ảnh tìm được cả hai biên. Nếu tính cả ảnh thất bại (IoU = 0) thì IoU toàn tập là 0.170.

| Nhóm | Số ảnh | Tìm được biên trái | Tìm được biên phải | Tìm được cả hai | IoU | MAE (px) | Thời gian |
|---|---|---|---|---|---|---|---|
| normal | 18 | 72.2% | 66.7% | 50.0% | 0.299 | 213.7 | 8.7 ms |
| curved | 12 | 66.7% | 75.0% | 50.0% | 0.381 | 177.2 | 7.7 ms |
| adverse | 20 | 80.0% | 75.0% | 60.0% | 0.292 | 279.7 | 6.9 ms |
| **Tất cả** | 50 | 74.0% | 72.0% | **54.0%** | **0.314** | 234.9 | 7.8 ms |

Thời gian chỉ tính bước Hough và phụ thuộc máy chạy.

## Chạy thử

```bash
pip install numpy pillow opencv-python pandas matplotlib
python preprocessing.py --input dataset/images --output outputs/preprocessed_for_canny --save-intermediate
python -m unittest discover tests
```

Để có ảnh cạnh và kết quả Hough, chạy lần lượt `Tien_xu_ly_HSV_Colab.ipynb` → `Canny_Colab.ipynb` → `Hough_Colab.ipynb`. Notebook P2 và P3 tìm dự án ở thư mục hiện tại hoặc `/content/Bai_Tap_Lon`, `/content/drive/MyDrive/Bai_Tap_Lon` trên Colab. Notebook P4 dùng thư mục hiện tại hoặc thư mục cha.

## Vấn đề tồn đọng / việc cần làm

- **Phần 5 chưa có**: chưa có script ghép end-to-end từ ảnh gốc đến biên đường, chưa có demo. Chưa so sánh có/không lọc màu trên IoU/MAE, chưa tính độ lệch chuẩn hay tỉ lệ thành công SR như Phần 1 đề ra. Các thẻ phụ phân tích lỗi (shadow, lowlight, occlusion…) mới được định nghĩa trong báo cáo P1, chưa có file gán thẻ.
- **Tham số Hough mặc định không phải bộ tốt nhất**: trong `hough_parameters_grid.csv`, bộ (threshold 15, minLineLength 20, maxLineGap 25) tìm được cả hai biên ở 82% số ảnh, IoU toàn tập 0.237. Bộ mặc định (25, 25, 25) chỉ đạt 54% và 0.170. Đổi lại, MAE của bộ (15, 20, 25) hơi cao hơn: 249.8 px so với 234.9 px. Bộ (25, 25, 25) cũng không nằm trong lưới quét, vì minLineLength chỉ thử 20/30/45.
- **Bảng 4.1 trong `Bao_cao_Phan_4_Hough.pdf` không khớp CSV mà nó trích dẫn**. Ví dụ, dòng (15, 20, 25) ghi 56% và IoU 0.1742, trong khi CSV ghi 82% và 0.2374. Chạy lại `hough.py` thì ra đúng số trong CSV, nên CSV mới là số đúng.
- **Không có script sinh CSV/hình của Hough**: `baocao/hough_road_boundary_tom_tat.md` nhắc `scripts/generate_hough_assets.py` nhưng file không tồn tại. `Hough_Colab.ipynb` chỉ hiển thị bảng, không ghi CSV.
- MAE còn lớn (~235 px trên ảnh rộng 960 px). Có 23/50 ảnh không ra đa giác mặt đường: 19 ảnh chỉ tìm được một biên, 4 ảnh không tìm được biên nào.
- Notebook chép lại code thay vì import các file `.py`, nên sửa một bên phải sửa cả bên kia.
- Chưa có `requirements.txt`; `tests/test_hough.py` cần `opencv-python`.
- Canny NumPy chậm; nếu cần thời gian thực thì chuyển sang `cv2.Canny`.
