# Kiến trúc và cách đọc code

Luồng local: browser → FastAPI → hàng đợi một worker → PDF/text hoặc OCR → kết quả theo trang → đối chiếu → xuất file.

Ứng dụng HTML/CSS/JS không có dependency build Node. Server đọc file local, lưu JSON atomic theo từng trang. Mỗi upload có UUID độc lập; tài liệu gốc có SHA-256 để theo dõi nguồn. Original, crops, edits được lưu trong workspace.

## Lựa chọn kỹ thuật

- Native PDF text trước: giữ nguyên dấu/ký tự khi font mapping tốt, tránh đưa 255 trang qua model.
- EasyOCR CRAFT: detector sẵn có chạy CPU, hỗ trợ vùng chữ; recognizer Latin cho chữ in Việt/Anh.
- VietOCR VGG19-BN + Transformer: pretrained tiếng Việt, fine-tune có giám sát; kích thước phù hợp triển khai CPU hơn VLM nhiều tỷ tham số.
- Training dùng mô hình VietOCR nhưng viết loop riêng: masks boolean chính xác, gradient accumulation, AMP, clip gradients, validation CER, checkpoint RNG.
- Giao diện manual crop là phương án xử lý detector bỏ sót chữ viết tay; không coi detector phổ thông là giải pháp hoàn hảo cho handwriting.
- Một worker/model lock: tránh hai OCR jobs đồng thời làm đầy RAM trên máy 16 GB.
- Torch/EasyOCR chỉ import khi cần OCR; PDF text không cần tải model.
- Model được tải bằng script riêng; chạy app không âm thầm tải Internet.

## Hợp đồng dữ liệu

Document: schema_version, document_id, filename, pages.
Page: number, width/height (pixel ảnh preview), method, blocks, text, edited_text, reviewed, error, seconds.
Block: id, bbox=[x0,y0,x1,y1] cùng hệ tọa độ ảnh trang, text, original_text, source, score, reviewed.

PDF text bbox được biến đổi qua rotation matrix rồi scale để khớp ảnh render. EXIF orientation ảnh được áp dụng trước detector.

## Đọc code theo thứ tự

1. Chạy native PDF demo và xem `result.json`.
2. `core.py`: document/page/block, chọn trang, chunk.
3. `pipeline.py`: quyết định native/OCR và lỗi từng trang.
4. `engines.py` + `recognizer.py`: detection, crop, decode.
5. `server.py` + `static/app.js`: upload, polling, edit và export.
6. `training/prepare.py`: splits, provenance, vocabulary.
7. `training/train.py`: forward/loss/backward/step/validation/resume.
8. `training/evaluate.py`: đọc predictions và phân loại lỗi.

## Kiểm soát phạm vi

App local không có auth, không deploy public. Không có NER, chatbot, dịch, tự sửa nội dung bằng LLM hay huấn luyện detector. Detector không được fine-tune trong notebook.
Các API tại `/docs` để tích hợp thử nghiệm local. Dùng một process; chạy nhiều worker sẽ tạo nhiều hàng đợi/model độc lập và không phù hợp thiết kế này.

Handwriting: các hộp từ gần nhau được ghép theo độ chồng theo chiều dọc và khoảng cách ngang, giữ riêng cột khi chọn 2 cột. Heuristic dành cho dòng tương đối ngang; với chữ nghiêng mạnh hoặc bố cục phức tạp, dùng khoanh vùng từng dòng và kiểm tra thủ công.
