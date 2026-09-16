"""VietOCR compatibility layer. No weight download and no network at import time."""
from pathlib import Path
import json
import math
import os
import torch
import numpy as np
from PIL import Image


def load_config(path):
    import yaml
    config=yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    config["cnn"]["pretrained"]=False  # OCR checkpoint includes CNN; avoid hidden ImageNet download.
    return config


def create_model(config, device="cpu"):
    from vietocr.tool.translate import build_model
    config={**config,"device":device}
    model,vocab=build_model(config)
    return model,vocab


def load_weights(model,path):
    import torch
    value=torch.load(path,map_location="cpu",weights_only=True)
    state=value.get("model",value.get("state_dict",value))
    model.load_state_dict(state,strict=True)


def image_array(image, config):
    cfg=config["dataset"]
    w=max(cfg["image_min_width"],min(cfg["image_max_width"],math.ceil(cfg["image_height"]*image.width/image.height/10)*10))
    image=image.convert("RGB").resize((w,cfg["image_height"]),Image.Resampling.LANCZOS)
    return np.asarray(image,dtype=np.float32).transpose(2,0,1)/255.0

def preprocess_handwriting_line(image, mode="off"):
    """
    Tiền xử lý nhẹ trước khi đưa crop một dòng vào VietOCR.

    mode:
      off    : giữ ảnh gốc
      light  : deskew + contrast + sharpen
      degrid : light + loại bớt các đường ô ly dài
    """
    import cv2
    import numpy as np
    from PIL import Image

    mode = str(mode).lower().strip()

    if mode in ("", "off", "0", "false", "none"):
        return image.convert("RGB")

    rgb = np.asarray(image.convert("RGB"))

    # -------------------------------------------------
    # 1. RGB -> grayscale
    # -------------------------------------------------
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    h, w = gray.shape[:2]

    # -------------------------------------------------
    # 2. Deskew nhẹ
    # Chỉ xoay nếu phát hiện góc lệch đáng kể nhưng < 5 độ
    # -------------------------------------------------
    edges = cv2.Canny(gray, 50, 150)

    min_length = max(20, int(w * 0.25))

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(15, int(w * 0.08)),
        minLineLength=min_length,
        maxLineGap=8,
    )

    angles = []

    if lines is not None:
        for line in lines[:, 0]:
            x1, y1, x2, y2 = line

            angle = np.degrees(
                np.arctan2(y2 - y1, x2 - x1)
            )

            # Chỉ lấy những đường gần nằm ngang
            if abs(angle) <= 7:
                angles.append(angle)

    if angles:
        angle = float(np.median(angles))

        # Không xoay vì sai lệch quá nhỏ.
        if 0.7 <= abs(angle) <= 5.0:
            center = (w / 2.0, h / 2.0)

            matrix = cv2.getRotationMatrix2D(
                center,
                angle,
                1.0,
            )

            gray = cv2.warpAffine(
                gray,
                matrix,
                (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255,
            )

    # -------------------------------------------------
    # 3. Nếu là giấy ô ly → giảm các line dài
    # -------------------------------------------------
    if mode == "degrid":

        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            11,
        )

        # Chỉ bắt các line RẤT DÀI.
        # Làm bảo thủ để tránh xóa nét chữ.
        horizontal_length = max(
            30,
            int(w * 0.35),
        )

        vertical_length = max(
            15,
            int(h * 0.80),
        )

        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (horizontal_length, 1),
        )

        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, vertical_length),
        )

        horizontal = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            horizontal_kernel,
        )

        vertical = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            vertical_kernel,
        )

        grid_mask = cv2.bitwise_or(
            horizontal,
            vertical,
        )

        # Điền nền vào vị trí grid line.
        gray = cv2.inpaint(
            gray,
            grid_mask,
            1,
            cv2.INPAINT_TELEA,
        )

    # -------------------------------------------------
    # 4. Tăng contrast cục bộ
    # -------------------------------------------------
    clahe = cv2.createCLAHE(
        clipLimit=1.6,
        tileGridSize=(8, 8),
    )

    enhanced = clahe.apply(gray)

    # -------------------------------------------------
    # 5. Sharpen nhẹ
    # -------------------------------------------------
    blurred = cv2.GaussianBlur(
        enhanced,
        (0, 0),
        0.7,
    )

    enhanced = cv2.addWeighted(
        enhanced,
        1.35,
        blurred,
        -0.35,
        0,
    )

    # VietOCR vẫn nhận RGB 3 channel
    rgb_out = cv2.cvtColor(
        enhanced,
        cv2.COLOR_GRAY2RGB,
    )

    return Image.fromarray(rgb_out)

def forward_train(model,images,tokens,padding_mask,source_widths=None):
    """Correct boolean attention masks (upstream old VietOCR casts padding to 0/1)."""
    import torch
    t=model.transformer
    src=model.cnn(images)
    src=t.pos_enc(src*math.sqrt(t.d_model))
    tgt=t.pos_enc(t.embed_tgt(tokens)*math.sqrt(t.d_model))
    mask=torch.triu(torch.ones(tokens.shape[0],tokens.shape[0],device=images.device,dtype=torch.bool),diagonal=1)
    source_mask=None
    if source_widths is not None:
        valid=torch.as_tensor(source_widths,device=images.device)//4
        source_mask=torch.arange(src.shape[0],device=images.device)[None,:] >= valid[:,None]
    out=t.transformer(src,tgt,tgt_mask=mask,tgt_key_padding_mask=padding_mask,
                      src_key_padding_mask=source_mask,memory_key_padding_mask=source_mask)
    return t.fc(out.transpose(0,1))


def decode(model,image_tensor,vocab,max_tokens=256):
    import torch
    model.eval()
    with torch.inference_mode():
        memory=model.transformer.forward_encoder(model.cnn(image_tensor))
        tokens=torch.ones((1,1),device=image_tensor.device,dtype=torch.long)
        probabilities=[]; truncated=True
        for _ in range(max_tokens):
            logits,_=model.transformer.forward_decoder(tokens,memory)
            probs=logits[0,-1].softmax(-1)
            # Never generate padding/start/mask inside a transcript.
            probs[0]=0; probs[1]=0; probs[3]=0
            token=probs.argmax().view(1,1)
            if token.item()==vocab.eos:
                truncated=False;break
            probabilities.append(float(probs[token.item()]))
            tokens=torch.cat([tokens,token],0)
        ids=tokens[:,0].tolist()
    text="".join(vocab.i2c[i] for i in ids[1:] if i>3)
    score=float(np.mean(probabilities)) if probabilities else 0.0
    return text,score,truncated


class LineRecognizer:
    def __init__(self, model_dir, device="cpu"):
        self.directory = Path(model_dir)

        weights = self.directory / "model.pth"
        cfg = self.directory / "config.yml"

        if not weights.exists() or not cfg.exists():
            raise RuntimeError(
                "Chưa có model viết tay. "
                "Chạy scripts/download_models.py hoặc chép bundle Kaggle "
                "vào models/handwriting."
            )

        self.config = load_config(cfg)
        self.device = device

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA chưa sẵn sàng. "
                "Chọn CPU hoặc cài đúng PyTorch CUDA."
            )

        # Không ép toàn bộ PyTorch process về 4 threads nữa.
        # Nếu muốn giới hạn thủ công:
        # $env:VIETDOC_TORCH_THREADS="4"
        threads = os.getenv("VIETDOC_TORCH_THREADS")

        if threads:
            try:
                torch.set_num_threads(
                    max(1, int(threads))
                )
            except ValueError:
                print(
                    "[VietDoc] VIETDOC_TORCH_THREADS không hợp lệ:",
                    threads
                )

        self.preprocess_mode = (
            os.getenv(
                "VIETDOC_PREPROCESS",
                "off"
            )
            .strip()
            .lower()
        )

        allowed_modes = {
            "off",
            "light",
            "degrid",
        }

        if self.preprocess_mode not in allowed_modes:
            print(
                f"[VietDoc] Preprocess mode "
                f"'{self.preprocess_mode}' không hợp lệ. "
                "Fallback -> off"
            )
            self.preprocess_mode = "off"

        print(
            "[VietDoc] Handwriting model:",
            weights.resolve()
        )

        print(
            "[VietDoc] Handwriting preprocess:",
            self.preprocess_mode
        )

        self.model, self.vocab = create_model(
            self.config,
            device
        )

        load_weights(
            self.model,
            weights
        )

        self.model.eval()

    def predict(self, image):
        """
        Nhận dạng một crop / một dòng chữ viết tay.

        Pipeline:
        PIL image
            -> optional preprocessing
            -> resize/normalize theo config VietOCR
            -> model
            -> greedy decode
        """

        processed_image = preprocess_handwriting_line(
            image,
            self.preprocess_mode
        )

        array = image_array(
            processed_image,
            self.config
        )

        data = torch.from_numpy(
            array
        )[None].to(
            self.device
        )

        return decode(
            self.model,
            data,
            self.vocab,
            self.config.get(
                "decode_max_tokens",
                256
            )
        )