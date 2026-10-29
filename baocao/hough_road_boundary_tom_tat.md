# Phần Hough trong phát hiện biên đường

*Bản tóm tắt đồng bộ với mã và số liệu hiện tại; xem báo cáo phần 4 để biết đầy đủ quy ước.*

## 1. Mục tiêu và đầu vào

Road boundary là mép vật lý toàn bộ lòng đường theo phần 1, không phải vạch chia làn. Phần 4 nhận nguyên ảnh cạnh Canny NumPy 0/255 từ outputs/canny/edges. Giữ nguyên tiền xử lý HSV trắng/vàng, Canny, hai notebook và dữ liệu bàn giao của ba module trước; không chạy lại hoặc thay nhánh đầu vào.

## 2. Thuật toán đã triển khai

ROI hình thang cố định theo tỷ lệ → P-Hough → lọc slope/vị trí → chọn cụm có hai đầu mút gần cùng mô hình → WLS x(y) theo chiều dài đoạn → kiểm tra hướng → cắt khung ảnh → tạo polygon trên miền y chung, dưới giao điểm. Không còn miền hợp lệ thì polygon=None. Không dùng ground truth chọn biên.

Cấu hình khởi điểm: threshold=25, minLineLength=25, maxLineGap=25; dung sai consensus=0,02W. ROI top_y=0,48H, bottom_y=0,98H, top_width=0,35W, bottom_width=0,98W. Đây không phải ROI thích ứng hoặc bộ tham số tối ưu đã được chứng minh.

## 3. Đọc kết quả đúng cách

IoU chính trên toàn ảnh/toàn tập, thất bại là mask rỗng. MAE cần cả hai biên tại đủ 60%,75%,90% H và luôn ghi N đo được. SR15 là MAE<15px trên toàn bộ ảnh. Có đầu ra không đồng nghĩa biên đúng.

| Nhóm | N | Có polygon (%) | IoU toàn tập ± σ | N đo MAE | MAE ± σ (px) | SR15 (%) |
| --- | --- | --- | --- | --- | --- | --- |
| normal | 18 | 50.0 | 0.1721 ± 0.1931 | 1 | 219.13 ± 0.00 | 0.0 |
| curved | 12 | 50.0 | 0.1919 ± 0.1998 | 3 | 189.61 ± 23.89 | 0.0 |
| adverse | 20 | 60.0 | 0.1656 ± 0.1680 | 6 | 322.67 ± 75.95 | 0.0 |
| all | 50 | 54.0 | 0.1743 ± 0.1855 | 10 | 272.40 ± 86.54 | 0.0 |

## 4. Giới hạn và phạm vi sửa

Tiền xử lý trắng/vàng có thể loại mất mép vật lý không mang màu vạch làn. Hough không thể khôi phục cạnh đã xóa; đây là giới hạn cần ghi rõ, không tự sửa module trước trong phần 4.

Lưới 27 tổ hợp chứa mặc định, dùng cùng ảnh Canny để khảo sát độ nhạy Hough. Chưa có held-out để kết luận tối ưu/tổng quát hóa. Hough đường thẳng và quy tắc dấu slope vẫn hạn chế ở đường cong; cụm nhất quán vẫn có thể là vạch làn hoặc vật thể sai. Sửa hình học không đồng nghĩa giải quyết ngữ nghĩa road boundary.

## 5. Chạy lại

```text
python scripts/generate_hough_assets.py
python scripts/build_hough_report.py
python scripts/compile_latex_pdf.py
```

Runtime Hough không gồm Canny/tiền xử lý/I/O; không gọi FPS của riêng bước này là tốc độ toàn pipeline. Báo cáo đầy đủ có benchmark warmup/median trên cùng ROI và thông tin môi trường.

## 6. Sáu slide gợi ý

1. Road boundary khác lane boundary; ground truth và giao diện ba phần trước.
2. ROI cố định, Standard Hough và P-Hough; lý do chọn đầu ra đoạn hữu hạn.
3. Vì sao không gộp mọi đoạn cùng dấu; consensus trước WLS.
4. Kiểm tra hướng, cắt khung ảnh và loại polygon tự cắt.
5. IoU toàn tập, coverage, N đo MAE và SR15 trên nguyên đầu ra Canny; không gọi availability là accuracy.
6. Ca lỗi đường cong/vạch làn/thiếu tương phản và giới hạn tập dữ liệu nhỏ.

Nguồn: ba bài Munajat 2015, Wang 2020, Belhadri 2026 trong Article; phân biệt rõ bài lane của Belhadri với mục tiêu road của dự án.
