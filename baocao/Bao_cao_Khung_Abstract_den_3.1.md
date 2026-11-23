# PHÁT HIỆN LÒNG ĐƯỜNG VÀ BIÊN ĐƯỜNG TRONG ẢNH CAMERA HÀNH TRÌNH BẰNG XỬ LÝ ẢNH CỔ ĐIỂN

*Bản khung báo cáo: Abstract, Introduction, Fundamentals, 3.1 Tiền xử lý, Conclusion, References. Nội dung theo pipeline hiện tại (v3, `road_pipeline.py`).*

## Abstract

Báo cáo này giải bài toán xác định vùng lòng đường và hai biên trái/phải của mặt đường trong ảnh camera hành trình. Biên ở đây là mép vật lý của mặt đường (vỉa hè, lề cỏ, dải phân cách), không phải vạch kẻ làn. Nhóm chỉ dùng kỹ thuật xử lý ảnh cổ điển, không dùng học sâu.

Phiên bản đầu tiên của nhóm đi theo hướng của các bài báo tham khảo: lọc màu trắng/vàng trong HSV, phát hiện cạnh bằng Canny, tìm đường thẳng bằng biến đổi Hough, rồi lấy đa giác giữa hai đường làm vùng đường. Cách này đạt IoU trung bình 0,174, vì Hough bám vào vạch kẻ làn thay vì mép đường. Pipeline hiện tại gồm năm bước. Đầu tiên, ảnh tối được chuẩn hóa tương phản bằng gamma và CLAHE trên kênh độ sáng L. Tiếp theo, pipeline ước lượng điểm tụ bằng Canny + Hough và ước lượng mép capo theo màu. Hai thông tin hình học này được dùng để khởi tạo tự động thuật toán phân vùng GrabCut. Mặt nạ GrabCut sau đó được hậu xử lý: cắt theo mép capo cong, đóng hình thái và cắt theo các tia biên do Canny + Hough tìm được. Cuối cùng, Canny + Hough chạy trên viền mặt nạ để lấy hai biên trái/phải.

Dữ liệu gồm 50 ảnh 1280 × 720 chọn từ BDD100K và IDD, chia thành ba nhóm: bình thường (18), đường cong (12) và điều kiện bất lợi (20). Mỗi ảnh có mặt nạ ground truth của toàn bộ lòng đường. Trên 50 ảnh, pipeline hiện tại đạt IoU 0,689 và F1 0,807 (precision 0,821, recall 0,830). Có 25/50 ảnh đạt IoU ≥ 0,7. Riêng nhóm ảnh tối, IoU tăng từ 0,094 lên 0,622. Sai số vị trí biên (MAE) còn 97,3 px, và tốc độ khoảng 6,2 s/ảnh trên CPU. Đây là hai hạn chế chính.

## 1. Introduction

### 1.1 Bối cảnh và vấn đề

Hệ thống hỗ trợ lái xe và xe tự hành cần biết mặt đường kết thúc ở đâu để giữ xe trong phần có thể đi được. Vạch kẻ làn không trả lời được câu hỏi này. Vạch có thể mờ, bị xe che, bị tuyết phủ, hoặc không có trên đường nhỏ. Kể cả khi vạch rõ ràng, nó chỉ chia làn bên trong lòng đường chứ không đánh dấu mép đường. Vì vậy, đề tài của nhóm đặt mục tiêu tìm **mép vật lý của lòng đường** và **vùng lòng đường** nằm giữa hai mép đó.

Các bài báo tham khảo của nhóm đều dùng chuỗi xử lý cổ điển gồm chuyển màu, phát hiện cạnh và biến đổi Hough. Belhadri và cộng sự [1] chuyển ảnh sang xám, tăng tương phản, chạy Canny [8] rồi biến đổi Hough [9] để tìm biên làn. Wang và cộng sự [2] lọc màu trong HSV trước Canny. Munajat và cộng sự [3] chỉ ra rằng ngưỡng Canny cố định hoạt động kém khi ánh sáng thay đổi. Các phương pháp này nhanh và dễ giải thích, nhưng được thiết kế cho **vạch kẻ làn**, nên không trực tiếp giải được bài toán **mép đường** của nhóm.

### 1.2 Mục tiêu, phạm vi và dữ liệu

- **Đầu vào:** ảnh màu từ camera hành trình, kích thước 1280 × 720.
- **Đầu ra:** mặt nạ nhị phân của lòng đường, và hai đường biên trái/phải (nếu tìm được).
- **Ràng buộc:** chỉ dùng xử lý ảnh cổ điển; ground truth chỉ dùng để đánh giá, không dùng trong pipeline.
- **Dữ liệu:** 50 ảnh chọn có chủ đích từ BDD100K [12] và IDD [13], gồm 18 ảnh bình thường, 12 ảnh đường cong và 20 ảnh điều kiện bất lợi (mưa, đêm, ngược sáng, đông xe). Mỗi ảnh có một mặt nạ ground truth 0/255. Mặt nạ bao trọn lòng đường, kể cả vạch sơn và phần mặt đường bị xe che. Theo độ sáng vùng lòng đường, có 7 ảnh sáng, 15 ảnh tương phản cao, 15 ảnh mức vừa và 13 ảnh tối.
- **Chỉ số đánh giá:** IoU và F1 giữa mặt nạ dự đoán và ground truth, theo cách đánh giá phát hiện đường ở mức pixel [14], tính trên toàn bộ 50 ảnh; ảnh không có dự đoán tính IoU = 0. Với biên đường: MAE theo pixel tại ba hàng 60%, 75% và 90% chiều cao ảnh, và SR15 là tỉ lệ ảnh có MAE < 15 px.

### 1.3 Những gì đã làm được

Nhóm phát triển pipeline qua ba phiên bản và đo từng phiên bản trên cùng 50 ảnh:

| Phiên bản | Cách làm chính | IoU | F1 |
|---|---|---:|---:|
| Bản gốc | Lọc vạch trắng/vàng HSV → Canny (50, 150) → Hough → đa giác | 0,174 | 0,256 |
| Notebook | Mặt nạ màu mặt đường theo S, V → Canny → Hough | 0,559 | 0,707 |
| v2 | Chuẩn hóa tương phản → GrabCut → Canny + Hough trên viền mặt nạ | 0,671 | 0,795 |
| **v3 (hiện tại)** | v2 + hậu xử lý mặt nạ | **0,689** | **0,807** |

*Bảng 1. Các phiên bản pipeline trên 50 ảnh.*

Hai phân tích định hướng cho các thay đổi:

- **Biểu diễn bằng hai đường thẳng có trần.** Ngay cả khi lấy hai đường thẳng khớp với mép trái/phải của chính ground truth, IoU cũng chỉ đạt khoảng 0,68–0,69, vì ground truth có khúc cua, có xe che và có vùng đường rộng hết khung ảnh. Bản notebook chỉ hơn một baseline tầm thường rất ít (0,559 so với 0,558). Baseline đó tô kín cả dải ảnh từ đường chân trời tới mép capo, không cần biên trái/phải. Từ đó nhóm chuyển đầu ra sang dạng **vùng** bám theo mép thật.
- **Ảnh tối mất cạnh ngay từ đầu.** Với ngưỡng V cố định của bộ lọc HSV và ngưỡng Canny cố định (50, 150), chỉ 8,8% điểm trên biên lòng đường của ảnh tối có cạnh đi qua. Ở ảnh sáng, tỉ lệ này là 63%. Từ đó nhóm thêm bước chuẩn hóa tương phản (mục 3.1) và ngưỡng Canny thích nghi.

### 1.4 Điểm mới

So với các bài báo tham khảo và bản gốc của nhóm, pipeline hiện tại có các điểm khác sau:

- **Chuẩn hóa tương phản có điều kiện trên kênh L của Lab.** Gamma tự động và CLAHE chỉ áp cho ảnh tối và chỉ trên kênh độ sáng, giữ nguyên màu. Tỉ lệ điểm biên có cạnh đi qua tăng từ 0,276 lên 0,810 trên toàn tập, và từ 0,088 lên 0,736 ở ảnh tối (khi đi kèm ngưỡng Canny thích nghi).
- **Ngưỡng Canny theo phân vị gradient** thay cho ngưỡng cố định, nên ngưỡng tự co giãn theo độ tương phản của từng ảnh. Đây là cách xử lý điểm yếu mà Munajat và cộng sự [3] nêu ra.
- **Phân vùng lòng đường bằng GrabCut [10] khởi tạo tự động từ hình học của cảnh.** Phía trên đường chân trời (qua điểm tụ) và phía dưới mép capo được đánh dấu là nền. Cửa sổ ngay trước xe là đường. Nêm phối cảnh từ điểm tụ là "có thể là đường". Không cần người dùng khoanh vùng như GrabCut gốc.
- **Canny + Hough đổi vai trò.** Chúng không còn tạo trực tiếp đa giác vùng đường. Thay vào đó, chúng được dùng để ước lượng điểm tụ, cắt phần mặt nạ tràn ra vỉa hè theo tia biên và tìm hai biên từ viền mặt nạ.
- **Đánh giá có kiểm soát.** Mỗi cải tiến được tắt riêng để đo đóng góp (ablation). Ngưỡng của bước hậu xử lý được chọn trên nửa ảnh có chỉ số chẵn, rồi kiểm tra trên nửa còn lại (IoU 0,690 và 0,688).

### 1.5 Cấu trúc báo cáo

Mục 2 trình bày kiến thức nền về ảnh số và các mô hình màu, sau đó mô tả tổng thể pipeline. Mục 3 trình bày các kỹ thuật theo thứ tự trong pipeline, bắt đầu bằng tiền xử lý ảnh đầu vào (3.1). Phần Conclusion tóm tắt kết quả, hạn chế và hướng phát triển.

## 2. Fundamentals

### 2.1 Lý thuyết ảnh kỹ thuật số

**Ảnh số là một hàm rời rạc.** Một ảnh xám là hàm hai biến f(x, y), trong đó giá trị f tại tọa độ (x, y) là cường độ sáng. Để máy tính xử lý được, hàm liên tục này được **lấy mẫu** theo không gian thành một lưới M hàng × N cột, và **lượng tử hóa** giá trị thành một số hữu hạn mức [4]. Mỗi phần tử của lưới là một **pixel**. Với ảnh 8 bit, mỗi pixel nhận 2⁸ = 256 mức, từ 0 (đen) đến 255 (trắng). Do đó, ảnh xám là một ma trận M × N số nguyên, còn ảnh màu là một mảng M × N × 3, mỗi pixel có ba giá trị.

**Hệ tọa độ.** Gốc tọa độ đặt ở góc trên bên trái. Trục x hướng sang phải theo cột, trục y hướng **xuống dưới** theo hàng. Quy ước này ảnh hưởng tới cách đọc độ dốc: trong ảnh camera hành trình, biên trái của lòng đường đi lên về phía điểm tụ nên có độ dốc âm, còn biên phải có độ dốc dương. Đường chân trời có y nhỏ, nắp capo có y lớn.

**Độ phân giải.** Ảnh gốc của bộ dữ liệu có 1280 × 720 = 921.600 pixel. Pipeline đưa mọi ảnh về 960 × 540 (mục 3.1). Mọi tham số tính theo pixel trong báo cáo đều hiểu theo kích thước này.

**Lân cận và liên thông.** Mỗi pixel có 4 lân cận theo cạnh (trên, dưới, trái, phải) hoặc 8 lân cận nếu tính cả đường chéo. Hai pixel thuộc cùng một **thành phần liên thông** nếu nối được với nhau qua một chuỗi pixel lân cận cùng nhãn. Pipeline dùng khái niệm này để chỉ giữ vùng đường liền khối chứa cửa sổ ngay trước xe. Phép cắt đồ thị trong GrabCut cũng xây dựng trên quan hệ lân cận giữa các pixel.

**Histogram.** Histogram của một ảnh xám đếm số pixel ở mỗi mức sáng 0–255. Ảnh tối có histogram dồn về phía 0. Ảnh tương phản thấp có histogram hẹp. Các phép tăng tương phản ở mục 3.1 (gamma, CLAHE) đều là phép biến đổi histogram.

**Gradient và cạnh.** Cạnh là nơi cường độ thay đổi đột ngột. Trên ảnh rời rạc, đạo hàm theo x và y được xấp xỉ bằng các phép chập với nhân nhỏ, ví dụ nhân Sobel 3 × 3. Từ đó tính được độ lớn gradient và hướng gradient của từng pixel. Đây là cơ sở của bộ dò cạnh Canny [8].

### 2.2 Kiểu ảnh và mô hình màu

![Các kiểu ảnh và kênh màu](outputs/khung_bao_cao/fig_2_2_kieu_anh.png)

*Hình 1. Ảnh normal_11 dưới các dạng biểu diễn dùng trong pipeline. Kênh H không ổn định ở các vùng gần như không màu như mặt đường và nắp capo. Kênh S tách rõ mặt đường nhựa (S thấp, tối) khỏi vạch vàng và bầu trời.*

**Ảnh màu RGB.** Mỗi pixel có ba giá trị đỏ (R), lục (G) và lam (B), mỗi giá trị 0–255, theo mô hình màu cộng. Đây là định dạng ảnh camera trả về. Nhược điểm của RGB là cả ba kênh đều phụ thuộc vào độ sáng. Khi một vùng mặt đường đi từ nắng vào bóng râm, cả R, G, B cùng giảm, dù màu của vật liệu không đổi. Vì vậy, khó đặt ngưỡng màu ổn định trực tiếp trên RGB.

**Ảnh xám.** Ảnh xám chỉ có một kênh cường độ, tính bằng tổng có trọng số theo chuẩn ITU-R BT.601 [17]: Gray = 0,299R + 0,587G + 0,114B. Trọng số phản ánh độ nhạy khác nhau của mắt người với ba màu. Ảnh xám là đầu vào của các phép tính gradient, nên Canny chạy trên ảnh xám.

**Ảnh nhị phân.** Mỗi pixel chỉ nhận một trong hai giá trị, trong pipeline là 0 và 255. Pipeline tạo ra ba loại ảnh nhị phân: **ảnh cạnh** (đầu ra của Canny, 255 tại pixel cạnh), **mặt nạ màu** (255 tại pixel thỏa điều kiện màu) và **mặt nạ lòng đường** (255 tại pixel thuộc lòng đường, Hình 1, ô cuối). Ground truth cũng là ảnh nhị phân, nên đánh giá IoU là phép so sánh hai ảnh nhị phân.

**Mô hình màu HSV.** HSV biểu diễn màu bằng sắc độ H (hue: loại màu), độ bão hòa S (saturation: màu đậm hay nhạt) và độ sáng V (value) [5]. Với R, G, B chuẩn hóa về [0, 1], đặt max = max(R, G, B) và min = min(R, G, B):

- V = max.
- S = (max − min) / max, và S = 0 khi max = 0.
- H là góc trên vòng màu, xác định theo kênh nào đạt max. Ví dụ, khi R là lớn nhất thì H = 60° × (G − B) / (max − min).

Pipeline dùng quy ước của OpenCV [16]: H thuộc [0, 179] (góc chia đôi), S và V thuộc [0, 255].

**Vì sao dùng HSV.** HSV tách thông tin màu (H, S) khỏi độ sáng (V), nên ngưỡng màu ít bị ảnh hưởng hơn khi ánh sáng thay đổi. Trong dự án, HSV được dùng ở bốn chỗ (ba chỗ trong pipeline hiện tại và một chỗ ở bản gốc), đều dựa vào tính chất này:

- **Vạch sơn trắng** có S thấp và V cao. **Vạch vàng** có H trong khoảng 15–40 và S cao. Hai điều kiện này dùng để tìm vạch sơn và gộp vạch vào vùng mặt đường khi quét tìm mép capo.
- **Mặt đường nhựa** gần như không có màu, nên S thấp và V ở mức trung bình (Hình 1, kênh S). Pipeline lấy trung vị và độ phân tán của S, V trong cửa sổ ngay trước xe làm mô hình màu mặt đường. Mô hình này dùng để quét từng hàng xuống dưới và tìm mép capo, nơi màu đổi khác mặt đường.
- **Kênh V** đóng vai trò ảnh độ sáng để Canny + Hough tìm các đoạn thẳng và ước lượng điểm tụ.
- **Bản gốc** dùng HSV để **lọc giữ vạch trắng/vàng** trước Canny, theo Wang và cộng sự [2].

**Giới hạn của HSV và lý do dùng thêm Lab.** Kênh H mất ổn định khi S thấp: mặt đường và nắp capo trong Hình 1 có H thay đổi lộn xộn. Thí nghiệm của nhóm cho thấy bằng chứng màu theo S, V **không đủ để tách mặt đường** khỏi cây, nhà và bầu trời. Chọn biên theo mặt nạ S-V chỉ đạt IoU khoảng 0,556, ngang với baseline tô kín dải ảnh. Lọc giữ vạch trắng/vàng còn xóa luôn mép vật lý của đường, nên bản gốc chỉ đạt IoU 0,174. Vì vậy, pipeline hiện tại **không dùng HSV để quyết định vùng đường**. Ở những chỗ cần tách độ sáng khỏi màu một cách chính xác hơn, pipeline dùng **CIELAB (Lab)** [18]: L là độ sáng, a và b là hai trục màu (lục–đỏ, lam–vàng), và khoảng cách Euclid trong Lab (ΔE) xấp xỉ khác biệt màu mà mắt người cảm nhận. Lab được dùng khi chuẩn hóa tương phản (chỉ sửa L, mục 3.1) và khi so sánh màu hai bên một biên ở bước hậu xử lý.

### 2.3 Mô tả pipeline

![Sơ đồ pipeline](outputs/khung_bao_cao/fig_2_3_pipeline.png)

*Hình 2. Pipeline hiện tại (v3). Vùng đường để đánh giá là mặt nạ sau bước hậu xử lý; hai biên trái/phải chỉ dùng để vẽ và đo MAE.*

| Bước | Đầu vào | Kỹ thuật | Đầu ra |
|---|---|---|---|
| 1. Resize | Ảnh RGB 1280 × 720 | Nội suy Lanczos | Ảnh RGB 960 × 540 |
| 2. Chuẩn hóa tương phản | Ảnh RGB 960 × 540 | Lab, gamma, CLAHE trên L; chỉ ảnh tối | Ảnh RGB đã chuẩn hóa |
| 3. Ước lượng hình học | Ảnh RGB 960 × 540 | Canny + Hough trên kênh V; quét hàng theo màu S, V | Điểm tụ (x, y), hàng mép capo |
| 4. Phân vùng | Ảnh đã chuẩn hóa, điểm tụ, mép capo | GrabCut khởi tạo hình học, 3 lần chạy lấy đa số | Mặt nạ lòng đường thô |
| 5. Hậu xử lý | Mặt nạ thô, ảnh đã chuẩn hóa | Cắt theo mép capo cong, đóng hình thái, cắt theo tia Canny + Hough | Mặt nạ lòng đường cuối |
| 6. Hai biên | Mặt nạ cuối | Canny trên mặt nạ, Hough, hồi quy WLS | Hai đường biên trái/phải |
| 7. Đánh giá | Mặt nạ cuối, hai biên, ground truth | IoU, F1, MAE, SR15 | Bảng kết quả |

*Bảng 2. Các bước của pipeline, đầu vào và đầu ra của mỗi bước.*

**Bước 1–2: tiền xử lý.** Đưa ảnh về một kích thước chung. Với ảnh tối, kéo giãn độ sáng để các bước sau còn thấy được cạnh và khác biệt màu. Chi tiết ở mục 3.1.

**Bước 3: ước lượng hình học của cảnh.** Do phối cảnh, hai mép đường song song ngoài thực tế chụm lại tại **điểm tụ** gần đường chân trời. Pipeline chạy Canny trên kênh V, dùng Hough để lấy các đoạn thẳng, rồi cho mỗi cặp đoạn nghiêng trái và nghiêng phải bỏ phiếu tại giao điểm của chúng. Ô có nhiều phiếu nhất là điểm tụ. Điểm tụ là ràng buộc hình học thường dùng trong phát hiện đường [11]; Kong và cộng sự ước lượng nó bằng bỏ phiếu theo hướng texture, còn nhóm dùng cách đơn giản hơn dựa trên các đoạn Hough. Hàng ngang qua điểm tụ được coi là đường chân trời. **Mép capo** là hàng thấp nhất còn giống màu mặt đường khi quét từ cửa sổ trước xe xuống dưới.

**Bước 4: phân vùng bằng GrabCut.** GrabCut [10] học hai mô hình màu (hỗn hợp Gauss) cho "đường" và "nền", rồi tách hai vùng bằng phép cắt đồ thị. Phép cắt đồ thị cân bằng hai yêu cầu: mỗi pixel hợp với mô hình màu của nhãn nó nhận, và ranh giới đi qua chỗ đổi màu mạnh. Bản đánh dấu ban đầu được tạo tự động từ bước 3. Đây là bước quyết định vùng đường.

**Bước 5: hậu xử lý mặt nạ.** Bước này sửa ba lỗi điển hình của GrabCut. Mặt nạ lấn xuống nắp capo thì cắt theo một parabol khớp với mép capo. Mặt nạ bị khuyết chỗ xe che thì lấp bằng phép đóng hình thái. Mặt nạ tràn sang vỉa hè cùng màu nhựa đường thì cắt theo tia biên đi qua điểm tụ. Tia biên được tìm bằng Canny với ngưỡng theo phân vị gradient, sau đó dùng Hough.

**Bước 6: hai biên trái/phải.** Canny chạy trên mặt nạ để lấy đường viền vùng đường. Chỉ giữ phần viền sát mép trái và mép phải, rồi dùng Hough và hồi quy có trọng số (WLS) để khớp một đường thẳng cho mỗi bên.

Trong pipeline, Canny được gọi ba lần với ba đầu vào khác nhau: kênh V (bước 3), ảnh xám đã chuẩn hóa (bước 5) và mặt nạ lòng đường (bước 6). Mỗi lần, đầu ra của Canny được đưa vào Hough.

## 3. Techniques

### 3.1 Tiền xử lý ảnh đầu vào: resize và chuẩn hóa tương phản

**3.1.1 Resize.** Mọi ảnh được đưa về 960 × 540 bằng nội suy Lanczos [19]. Lý do chọn kích thước này:

- **Giữ tỉ lệ 16:9** của ảnh gốc 1280 × 720 (hệ số 0,75 theo cả hai chiều), nên độ dốc của đường thẳng và hình dạng phối cảnh không bị méo. Nếu resize sang tỉ lệ khác, độ dốc của biên đường sẽ đổi và làm sai các ngưỡng độ dốc ở bước Hough.
- **Giảm 43,75% số pixel** (từ 921.600 xuống 518.400). Điều này quan trọng vì Canny tự cài bằng NumPy và GrabCut là các bước tốn thời gian nhất.
- **Thống nhất hệ tọa độ.** Mọi tham số tính bằng pixel (kích thước nhân, độ dài đoạn Hough, kích thước phần tử cấu trúc) dùng chung cho cả tập.

Lanczos là nội suy dùng hàm sinc có cửa sổ. Khi thu nhỏ ảnh, nó ít gây răng cưa và giữ cạnh sắc hơn nội suy song tuyến tính. Ground truth được resize về cùng kích thước bằng nội suy láng giềng gần nhất, để mặt nạ vẫn chỉ có hai giá trị 0 và 255.

**3.1.2 Vì sao cần chuẩn hóa tương phản.** Ở bản gốc, ngưỡng V cố định của bộ lọc HSV và ngưỡng Canny cố định (50, 150) gần như xóa hết thông tin của ảnh tối. Trong 13 ảnh tối, chỉ 8,8% điểm trên biên lòng đường có cạnh đi qua, chỉ 4/13 ảnh tạo được vùng đường, và IoU trung bình của nhóm này là 0,094. Ảnh tối có histogram dồn về vùng gần 0: các mức xám của mặt đường, lề đường và cây cối nằm sát nhau, nên gradient nhỏ và khác biệt màu cũng nhỏ.

**3.1.3 Chuyển sang Lab và chỉ sửa kênh L.** Ảnh RGB được chuyển sang Lab. Các phép biến đổi chỉ áp lên kênh độ sáng L. Hai kênh màu a, b giữ nguyên, sau đó ảnh được chuyển ngược về RGB. Nếu tăng sáng trực tiếp trên từng kênh R, G, B, tỉ lệ giữa ba kênh sẽ thay đổi và màu bị lệch. Khi đó GrabCut ở bước sau, vốn dựa vào màu, sẽ học sai mô hình màu mặt đường.

**3.1.4 Điều kiện áp dụng.** Bước chuẩn hóa **chỉ chạy khi trung vị L của nửa dưới ảnh nhỏ hơn 50** (trên thang 0–255). Pipeline xét nửa dưới vì đó là vùng có mặt đường. Nếu xét cả ảnh, một bầu trời sáng có thể khiến một ảnh có mặt đường rất tối vẫn bị coi là đủ sáng. Ngưỡng 50 được chọn bằng thực nghiệm. Áp CLAHE cho mọi ảnh làm IoU phân vùng giảm ở ảnh sáng, vì CLAHE làm đổi màu cục bộ. Với ngưỡng 50, có 13/50 ảnh được chuẩn hóa, trong đó 10 ảnh thuộc nhóm tối.

**3.1.5 Hiệu chỉnh gamma.** Phép biến đổi gamma ánh xạ mỗi mức sáng theo công thức L' = 255 × (L / 255)^γ. Với γ < 1, vùng tối được kéo giãn mạnh hơn vùng sáng. Pipeline tự chọn γ để đưa trung vị độ sáng của ảnh, m (chuẩn hóa về [0, 1]), lên mức mục tiêu 0,45:

γ = ln(0,45) / ln(m), giới hạn trong khoảng [1/2,5; 1] = [0,4; 1].

Giới hạn dưới 0,4 tránh kéo sáng quá mức, vì khi đó nhiễu cảm biến trong vùng tối bị khuếch đại thành vân hạt. Giới hạn trên 1 bảo đảm phép biến đổi không làm ảnh tối đi. Phép biến đổi được cài bằng bảng tra (LUT) 256 phần tử nên rất nhanh. Trong Hình 3, ảnh curved_06 có trung vị L rất thấp nên γ chạm giới hạn 0,40, còn normal_15 có γ = 0,49.

**3.1.6 CLAHE.** Gamma là phép biến đổi **toàn cục**: mọi pixel cùng mức sáng được ánh xạ như nhau. Vì vậy, gamma không làm tăng khác biệt giữa hai vùng có độ sáng gần nhau ở hai nơi khác nhau trong ảnh. Cân bằng histogram thích nghi (AHE) [7] giải quyết điều này bằng cách cân bằng histogram **riêng cho từng vùng nhỏ**. Nhược điểm của AHE là nó khuếch đại mạnh nhiễu ở các vùng gần như đồng nhất, chẳng hạn mặt đường trơn. **CLAHE** (Contrast Limited AHE) [6] thêm hai cơ chế:

- **Giới hạn tương phản.** Histogram của mỗi vùng bị cắt ở một ngưỡng (clip limit). Phần vượt ngưỡng được chia đều lại cho mọi mức. Nhờ vậy, độ dốc của hàm ánh xạ bị giới hạn, và nhiễu không bị khuếch đại vô hạn.
- **Nội suy giữa các ô.** Ảnh được chia thành lưới ô. Mỗi ô có một hàm ánh xạ riêng. Giá trị của mỗi pixel được nội suy song tuyến tính từ hàm ánh xạ của các ô lân cận, để không lộ đường ranh giới giữa các ô.

Pipeline dùng CLAHE của OpenCV với clip limit 2,0 và lưới 8 × 8 ô, áp lên kênh L sau bước gamma. Gamma đưa độ sáng chung về mức hợp lý trước, sau đó CLAHE tăng tương phản cục bộ.

![Chuẩn hóa tương phản](outputs/khung_bao_cao/fig_3_1_chuan_hoa.png)

*Hình 3. Chuẩn hóa tương phản trên hai ảnh tối. Histogram kênh L của ảnh gốc (xám) dồn về gần 0. Gamma (lam) dịch histogram sang phải. CLAHE (đỏ) trải rộng thêm histogram và làm rõ mép đường cùng vạch sơn. Ở curved_06, CLAHE cũng làm lộ nhiễu hạt ở vùng trời.*

**3.1.7 Kết quả.** Chuẩn hóa tương phản được đánh giá theo hai cách.

Cách thứ nhất là đo trực tiếp lượng cạnh giữ được. Chỉ số recall là tỉ lệ điểm trên biên lòng đường (lấy từ ground truth) có cạnh Canny trong bán kính 5 px. Density là tỉ lệ pixel cạnh trên toàn ảnh.

| Điều kiện | Lọc HSV + Canny (50, 150) | Ảnh xám + Canny (50, 150) | Chuẩn hóa + Canny (50, 150) | Chuẩn hóa + Canny ngưỡng phân vị |
|---|---:|---:|---:|---:|
| sáng | 0,632 / 0,031 | 0,679 / 0,035 | 0,835 / 0,052 | 0,861 / 0,062 |
| tương phản cao | 0,368 / 0,026 | 0,592 / 0,041 | 0,778 / 0,066 | 0,836 / 0,074 |
| vừa | 0,182 / 0,019 | 0,379 / 0,024 | 0,703 / 0,046 | 0,824 / 0,063 |
| **tối** | **0,088 / 0,010** | 0,214 / 0,018 | 0,457 / 0,035 | **0,736 / 0,058** |
| tất cả | 0,276 / 0,021 | 0,442 / 0,029 | 0,680 / 0,050 | 0,810 / 0,065 |

*Bảng 3. Recall / density của ảnh cạnh theo điều kiện ánh sáng. Nguồn: `outputs/pipeline_v2/canny_contrast_by_condition.csv`.*

So hai cột ở giữa (cùng ngưỡng Canny cố định), riêng chuẩn hóa tương phản đã nâng recall ở ảnh tối từ 0,214 lên 0,457, và trên toàn tập từ 0,442 lên 0,680. Khi kết hợp thêm ngưỡng Canny theo phân vị gradient, recall ở ảnh tối đạt 0,736. Mật độ cạnh của ảnh sáng so với ảnh tối giảm từ 3,1 lần xuống khoảng 1,07 lần, tức là lượng cạnh gần như không còn phụ thuộc ánh sáng. Đổi lại, density tăng từ 0,021 lên 0,065: số cạnh nhiễu (cây, nhà, xe) cũng tăng.

Cách thứ hai là đo trên kết quả cuối, bằng cách tắt riêng bước chuẩn hóa và giữ nguyên các bước khác (ablation trên v2). IoU toàn tập giảm từ 0,671 xuống 0,658. Toàn bộ mức giảm này nằm ở nhóm ảnh tối (0,612 xuống 0,562); ba nhóm còn lại không đổi, đúng như thiết kế. Ở pipeline v3 hiện tại, IoU nhóm tối đạt 0,622, so với 0,094 của bản gốc.

**3.1.8 Hạn chế.**

- **CLAHE khuếch đại nhiễu ở ảnh rất tối.** Nhiễu hạt thấy rõ ở vùng trời của curved_06 (Hình 3), và đó là một nguồn cạnh nhiễu cho Canny.
- **Ngưỡng 50 được chọn trên chính 50 ảnh của bộ dữ liệu**, chưa kiểm tra trên một tập độc lập.
- **Một số ảnh tối vẫn kém sau chuẩn hóa.** adverse_15 (ngược sáng) và curved_10 chỉ đạt IoU khoảng 0,38–0,39. Ở những ảnh này, chuẩn hóa độ sáng không đủ để tách được màu mặt đường khỏi nền.

*Các mục 3.2 trở đi (ước lượng hình học, GrabCut, hậu xử lý, Canny + Hough tìm biên, đánh giá) do các thành viên khác trình bày.*

## Conclusion

Báo cáo giải bài toán xác định vùng lòng đường và hai biên vật lý của mặt đường trong ảnh camera hành trình, chỉ bằng xử lý ảnh cổ điển. Bản gốc theo hướng của các bài báo tham khảo (lọc vạch HSV, Canny, Hough, đa giác giữa hai đường) chỉ đạt IoU 0,174. Có hai nguyên nhân chính. Thứ nhất, phương pháp được thiết kế cho vạch kẻ làn chứ không cho mép đường. Thứ hai, ảnh tối gần như mất hết cạnh khi dùng ngưỡng cố định.

Pipeline hiện tại khắc phục cả hai. Ảnh tối được chuẩn hóa tương phản trên kênh L. Vùng đường được phân vùng bằng GrabCut khởi tạo tự động từ điểm tụ và mép capo, rồi được hậu xử lý. Canny + Hough được dùng để ước lượng điểm tụ, cắt vùng tràn và tìm hai biên. Kết quả trên 50 ảnh: IoU 0,689, F1 0,807, 25/50 ảnh đạt IoU ≥ 0,7, IoU nhóm tối tăng từ 0,094 lên 0,622.

Các hạn chế còn lại:

- Chưa đạt IoU 0,7 trên toàn tập. Nhóm curved và adverse còn 0,668 và 0,627.
- Sai số vị trí biên còn lớn (MAE 97,3 px, SR15 8%).
- Pipeline chậm, khoảng 6,2 s/ảnh trên CPU, chủ yếu do ba lần chạy GrabCut và Canny tự cài bằng NumPy.
- Chưa có tập kiểm tra độc lập; tham số được chọn trên chính 50 ảnh.
- GrabCut dựa vào màu nên vẫn tràn sang vỉa hè hoặc bãi đỗ cùng màu nhựa đường.

Hướng phát triển:

- Đưa ảnh cạnh Canny vào chi phí cắt của GrabCut để chặn vùng tràn.
- Cải thiện bộ dò mép capo. Nếu biết đúng mép capo, IoU có thể lên khoảng 0,73 (đo bằng ground truth).
- Thử mô hình màu bất biến bóng đổ [15] cho ảnh tương phản cao.
- Tách riêng tập validation và tập test trước khi tinh chỉnh thêm tham số.

## References

[1] Y. Belhadri, D. Mohamed, B. Alshaqaqi, and A. S. Baquhaizel, "Lane Boundary Detection in Dynamic Environments," *Rev. Roum. Sci. Techn. – Électrotechn. et Énerg.*, vol. 71, no. 2, pp. 299–304, 2026.

[2] S. Wang, S. Wu, X. Wang, and Z. Li, "A Canny Operator Road Edge Detection Method Based on Color Features," *Journal of Physics: Conference Series*, vol. 1629, 012018, 2020.

[3] M. D. E. Munajat, D. H. Widyantoro, and R. Munir, "Road Detection System Based on RGB Histogram Filterization and Boundary Classifier," in *Proc. ICACSIS*, pp. 195–200, 2015.

[4] R. C. Gonzalez and R. E. Woods, *Digital Image Processing*, 4th ed. Pearson, 2018.

[5] A. R. Smith, "Color Gamut Transform Pairs," *ACM SIGGRAPH Computer Graphics*, vol. 12, no. 3, pp. 12–19, 1978.

[6] K. Zuiderveld, "Contrast Limited Adaptive Histogram Equalization," in *Graphics Gems IV*, Academic Press, pp. 474–485, 1994.

[7] S. M. Pizer et al., "Adaptive Histogram Equalization and Its Variations," *Computer Vision, Graphics, and Image Processing*, vol. 39, no. 3, pp. 355–368, 1987.

[8] J. F. Canny, "A Computational Approach to Edge Detection," *IEEE Trans. Pattern Analysis and Machine Intelligence*, vol. 8, no. 6, pp. 679–698, 1986.

[9] R. O. Duda and P. E. Hart, "Use of the Hough Transformation to Detect Lines and Curves in Pictures," *Communications of the ACM*, vol. 15, no. 1, pp. 11–15, 1972.

[10] C. Rother, V. Kolmogorov, and A. Blake, "GrabCut: Interactive Foreground Extraction Using Iterated Graph Cuts," *ACM Trans. Graphics (SIGGRAPH)*, vol. 23, no. 3, pp. 309–314, 2004.

[11] H. Kong, J.-Y. Audibert, and J. Ponce, "Vanishing Point Detection for Road Detection," in *Proc. CVPR*, pp. 96–103, 2009.

[12] F. Yu et al., "BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning," in *Proc. CVPR*, 2020.

[13] G. Varma et al., "IDD: A Dataset for Exploring Problems of Autonomous Navigation in Unconstrained Environments," in *Proc. WACV*, 2019.

[14] J. Fritsch, T. Kühnl, and A. Geiger, "A New Performance Measure and Evaluation Benchmark for Road Detection Algorithms," in *Proc. ITSC*, pp. 1693–1700, 2013.

[15] J. M. Álvarez and A. M. López, "Road Detection Based on Illuminant Invariance," *IEEE Trans. Intelligent Transportation Systems*, vol. 12, no. 1, pp. 184–193, 2011.

[16] G. Bradski, "The OpenCV Library," *Dr. Dobb's Journal of Software Tools*, 2000.

[17] ITU-R, *Recommendation BT.601-7: Studio Encoding Parameters of Digital Television for Standard 4:3 and Wide-Screen 16:9 Aspect Ratios*, 2011.

[18] CIE, *Colorimetry*, 3rd ed., CIE Publication 15:2004, 2004.

[19] C. E. Duchon, "Lanczos Filtering in One and Two Dimensions," *Journal of Applied Meteorology*, vol. 18, no. 8, pp. 1016–1022, 1979.
