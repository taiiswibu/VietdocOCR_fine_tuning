# Hướng dẫn Kaggle T4

## Thiết lập

1. Tạo Dataset private chứa `vietdoc-ocr-complete.zip`.
2. Tạo Notebook mới → Import `notebooks/01_kaggle_finetune.ipynb`.
3. Add Input dataset source; chọn T4; bật Internet.
4. Hugging Face: chấp nhận điều kiện truy cập `5CD-AI/Viet-Handwriting-OCR-v2`.
5. Kaggle Secrets: thêm `HF_TOKEN`, cấp quyền notebook. Không dán token vào cell, Git hoặc ảnh chụp.
6. Chạy notebook từ đầu. Sau khi thay torch/numpy/Pillow, restart kernel nếu phiên đã import chúng.

Script baseline tải từ Hugging Face revision cố định. Với train dataset, `prepare` ghi commit revision hiện tại vào `preparation.json` để tái lập.

## Lệnh tương đương ngoài notebook

```bash
python scripts/download_models.py --base
python -m training.prepare --accept-noncommercial
python -m training.train --data data/handwriting --output runs/handwriting --epochs 8 --batch-size 8 --accum 4
python -m training.evaluate --model models/base --manifest data/handwriting/test.jsonl --output runs/baseline-test --device cuda:0
python -m training.evaluate --model runs/handwriting --manifest data/handwriting/test.jsonl --output runs/finetuned-test --device cuda:0
python scripts/compare_results.py runs/baseline-test/metrics.json runs/finetuned-test/metrics.json
python scripts/package_model.py runs/handwriting --output /kaggle/working/handwriting-model.zip
```

T4 dùng FP16 (không yêu cầu BF16/FlashAttention/vLLM). Một GPU, không tự cộng VRAM hai T4.

## Kiểm tra pipeline nhanh trước một lần train dài

Sau chuẩn bị dữ liệu, chạy:

```bash
python -m training.train --output runs/handwriting --epochs 8 --batch-size 8 --accum 4 --max-updates 2
```

Lệnh này tạo `last.pt`, **chưa tạo bản final**. Sau đó tiếp tục:

```bash
python -m training.train --output runs/handwriting --epochs 8 --batch-size 8 --accum 4 --resume runs/handwriting/last.pt
```

Resume giữ nguyên batch-size, accum, epochs, seed, data/config. Checkpoint có model, optimizer, scheduler, GradScaler, RNG, epoch và batch kế tiếp. Dữ liệu, config và hyperparameters quyết định lịch learning rate; không đổi giữa run.

## Giữ tiến độ khi hết phiên

- Save Version/Output chứa `data/handwriting/`, `runs/handwriting/`, `models/base/`.
- Phiên sau Add Input output trước, chép ba thư mục về cùng vị trí trong project ở `/kaggle/working`.
- Giữ cả `model.pth` tốt nhất, `last.pt`, `config.yml`, `metrics.jsonl`, `run.json`.
- Nếu đổi output directory, chép cả thư mục run trước. Không chỉ lấy last.pt rồi làm mất model tốt nhất trước đó.
- Checkpoint định kỳ giảm công chạy lại nhưng không đảm bảo file `/kaggle/working` tồn tại sau mọi loại reset. Cần lưu output chủ động.
- Thời gian toàn bộ training chưa được đo trên T4; lấy thời gian 100 updates đầu để ước lượng, quản lý quota theo UI tài khoản.

## Chuyển về laptop

Giải nén `handwriting-model.zip`; bên trong có thư mục `handwriting`. Đặt vào `vietdoc-ocr/models/handwriting/`.
Sao lưu baseline cũ. Không chạy lại downloader lên model fine-tuned. Restart app và thử 1–3 ảnh/crop.

Giữ các báo cáo test và log. Chỉ viết số CER/cải thiện vào CV sau khi notebook đã chạy và kiểm tra đủ số mẫu.
