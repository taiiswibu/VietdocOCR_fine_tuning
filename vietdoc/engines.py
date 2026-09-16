from pathlib import Path
import os

import numpy as np

from .core import Block, normalize, reading_order


def merge_line_boxes(boxes, columns=1, page_width=None):
    """
    Gom các word/fragment boxes của EasyOCR thành từng dòng.

    Mục tiêu:
    - Fragment nằm cùng hàng Y -> cùng một dòng.
    - Không phụ thuộc quá mạnh vào khoảng cách ngang X.
    - Tách riêng cột nếu columns == 2.
    - Phù hợp hơn với handwriting, nơi detector thường cắt một dòng
      thành nhiều box nhỏ.
    """

    if not boxes:
        return []

    # Chuẩn hóa dữ liệu.
    clean_boxes = []

    for box in boxes:
        if len(box) != 4:
            continue

        x0, y0, x1, y1 = map(float, box)

        if x1 <= x0 or y1 <= y0:
            continue

        clean_boxes.append(
            [x0, y0, x1, y1]
        )

    if not clean_boxes:
        return []

    heights = [
        b[3] - b[1]
        for b in clean_boxes
    ]

    median_height = float(
        np.median(heights)
    )

    # Sai số tâm Y cho hai fragment cùng dòng.
    #
    # Nếu để quá lớn -> dễ gộp 2 dòng thành 1.
    # Nếu quá nhỏ -> một dòng bị vỡ thành nhiều đoạn.
    y_tolerance = max(
        5.0,
        median_height * 0.55
    )

    # Trên xuống trước, sau đó trái sang phải.
    clean_boxes.sort(
        key=lambda b: (
            (b[1] + b[3]) / 2.0,
            b[0]
        )
    )

    rows = []

    for box in clean_boxes:
        x0, y0, x1, y1 = box

        cy = (
            y0 + y1
        ) / 2.0

        height = max(
            1.0,
            y1 - y0
        )

        # Xác định cột.
        if (
            columns == 2
            and page_width
        ):
            center_x = (
                x0 + x1
            ) / 2.0

            column = int(
                center_x >= page_width / 2.0
            )

        else:
            column = 0

        best_row = None
        best_score = float("inf")

        for row in rows:

            if row["column"] != column:
                continue

            rx0, ry0, rx1, ry1 = row["box"]

            row_height = max(
                1.0,
                ry1 - ry0
            )

            # Khoảng cách tâm Y.
            center_distance = abs(
                cy - row["cy"]
            )

            # Mức overlap theo chiều dọc.
            overlap = max(
                0.0,
                min(y1, ry1)
                - max(y0, ry0)
            )

            overlap_ratio = (
                overlap
                / max(
                    1.0,
                    min(
                        height,
                        row_height
                    )
                )
            )

            # Hai box được xem là cùng dòng nếu:
            # 1. tâm Y đủ gần
            # HOẶC
            # 2. overlap chiều dọc đủ lớn.
            same_line = (
                center_distance <= y_tolerance
                and overlap_ratio >= 0.15
            )

            if not same_line:
                continue

            if center_distance < best_score:
                best_score = center_distance
                best_row = row

        if best_row is None:

            rows.append({
                "box": [
                    x0,
                    y0,
                    x1,
                    y1
                ],
                "cy": cy,
                "count": 1,
                "column": column,
            })

        else:

            rx0, ry0, rx1, ry1 = (
                best_row["box"]
            )

            count = (
                best_row["count"]
            )

            best_row["box"] = [
                min(rx0, x0),
                min(ry0, y0),
                max(rx1, x1),
                max(ry1, y1),
            ]

            best_row["cy"] = (
                (
                    best_row["cy"]
                    * count
                )
                + cy
            ) / (
                count + 1
            )

            best_row["count"] += 1

    # Thứ tự đọc cuối cùng.
    if columns == 2:

        rows.sort(
            key=lambda row: (
                row["column"],
                row["cy"],
                row["box"][0],
            )
        )

    else:

        rows.sort(
            key=lambda row: (
                row["cy"],
                row["box"][0],
            )
        )

    return [
        row["box"]
        for row in rows
    ]


class OCREngine:

    def __init__(self, root: Path):
        self.root = root
        self.readers = {}
        self.line_model = None

    def reader(self, recognition=True):
        """
        EasyOCR reader.

        recognition=True:
            detector + recognizer cho chữ in.

        recognition=False:
            chỉ detector, dùng để tìm vùng handwriting.
        """

        if recognition not in self.readers:

            try:
                import easyocr

            except ImportError as e:
                raise RuntimeError(
                    'Chưa cài OCR. '
                    'Chạy pip install -e ".[ocr]" '
                    'theo README.'
                ) from e

            self.readers[
                recognition
            ] = easyocr.Reader(
                ["vi", "en"],
                gpu=False,
                recognizer=recognition,
                model_storage_directory=str(
                    self.root
                    / "models"
                    / "easyocr"
                ),
                download_enabled=False,
                verbose=False,
            )

        return self.readers[
            recognition
        ]

    def _load_handwriting_model(self):
        """
        Lazy-load VietOCR handwriting model.
        """

        if self.line_model is not None:
            return

        from .recognizer import LineRecognizer

        device = os.getenv(
            "VIETDOC_DEVICE",
            "cpu"
        )

        self.line_model = LineRecognizer(
            self.root
            / "models"
            / "handwriting",
            device,
        )

    def _detect_handwriting_boxes(
        self,
        image,
        columns,
    ):
        """
        EasyOCR / CRAFT chỉ detect box.
        Sau đó gom các fragment thành từng dòng.
        """

        horizontal, free = (
            self.reader(False)
            .detect(
                np.asarray(image),

                text_threshold=0.6,
                low_text=0.3,
                link_threshold=0.3,

                canvas_size=2560,
                mag_ratio=1.0,

                min_size=8,

                slope_ths=0.2,
                ycenter_ths=0.5,
                height_ths=0.5,

                # Detector có thể cắt một dòng
                # thành nhiều fragment.
                # merge_line_boxes sẽ gom lại.
                width_ths=0.5,

                add_margin=0.08,
            )
        )

        boxes = []

        # Horizontal boxes từ EasyOCR có dạng:
        #
        # [x_min, x_max, y_min, y_max]
        #
        for (
            x0,
            x1,
            y0,
            y1
        ) in horizontal[0]:

            boxes.append([
                float(x0),
                float(y0),
                float(x1),
                float(y1),
            ])

        # Free-form polygon.
        for polygon in free[0]:

            p = np.asarray(
                polygon
            )

            boxes.append([
                float(
                    p[:, 0].min()
                ),
                float(
                    p[:, 1].min()
                ),
                float(
                    p[:, 0].max()
                ),
                float(
                    p[:, 1].max()
                ),
            ])

        return merge_line_boxes(
            boxes,
            columns,
            image.width,
        )

    def _pad_line_box(
        self,
        box,
        image_width,
        image_height,
    ):
        """
        Nới crop ra một chút.

        Mục đích:
        - không mất dấu tiếng Việt
        - không mất nét đầu/cuối chữ
        - VietOCR có thêm context xung quanh dòng
        """

        x0, y0, x1, y1 = [
            int(v)
            for v in box
        ]

        height = max(
            1,
            y1 - y0
        )

        # Horizontal padding lớn hơn vertical.
        pad_x = max(
            5,
            int(
                height * 0.35
            )
        )

        pad_y = max(
            3,
            int(
                height * 0.20
            )
        )

        padded = [
            max(
                0,
                x0 - pad_x
            ),
            max(
                0,
                y0 - pad_y
            ),
            min(
                image_width,
                x1 + pad_x
            ),
            min(
                image_height,
                y1 + pad_y
            ),
        ]

        return padded

    def _sort_handwriting_blocks(
        self,
        blocks,
        columns,
        page_width,
    ):
        """
        Đảm bảo mỗi block handwriting được xuất đúng thứ tự dòng.
        """

        if not blocks:
            return blocks

        if columns == 1:

            blocks.sort(
                key=lambda block: (
                    (
                        block.bbox[1]
                        + block.bbox[3]
                    ) / 2.0,
                    block.bbox[0],
                )
            )

            return blocks

        return reading_order(
            blocks,
            columns,
            page_width,
        )

    def run(
        self,
        image,
        mode="printed",
        columns=1,
        single_line=False,
    ):

        # =========================================================
        # PRINTED OCR
        # =========================================================

        if mode == "printed":

            result = (
                self.reader(True)
                .readtext(
                    np.asarray(image),
                    detail=1,
                    paragraph=False,
                    batch_size=1,
                )
            )

            blocks = []

            for (
                polygon,
                text,
                score
            ) in result:

                p = np.asarray(
                    polygon
                )

                box = [
                    float(
                        p[:, 0].min()
                    ),
                    float(
                        p[:, 1].min()
                    ),
                    float(
                        p[:, 0].max()
                    ),
                    float(
                        p[:, 1].max()
                    ),
                ]

                blocks.append(
                    Block(
                        str(
                            len(blocks)
                        ),
                        box,
                        normalize(
                            text
                        ),
                        "easyocr",
                        float(
                            score
                        ),
                    )
                )

            blocks = reading_order(
                blocks,
                columns,
                image.width,
            )

        # =========================================================
        # HANDWRITING OCR
        # =========================================================

        else:

            self._load_handwriting_model()

            # ---------------------------------------------
            # User đã crop đúng 1 dòng.
            #
            # Không chạy detector.
            # Toàn bộ ảnh crop được đưa vào VietOCR.
            # ---------------------------------------------

            if single_line:

                boxes = [[
                    0,
                    0,
                    image.width,
                    image.height,
                ]]

            # ---------------------------------------------
            # Full-page handwriting.
            #
            # EasyOCR/CRAFT detect
            # -> merge fragments
            # -> VietOCR từng dòng.
            # ---------------------------------------------

            else:

                boxes = (
                    self._detect_handwriting_boxes(
                        image,
                        columns,
                    )
                )

            blocks = []

            for box in boxes:

                padded_box = (
                    self._pad_line_box(
                        box,
                        image.width,
                        image.height,
                    )
                )

                x0, y0, x1, y1 = (
                    padded_box
                )

                if (
                    x1 <= x0
                    or y1 <= y0
                ):
                    continue

                crop = image.crop(
                    padded_box
                )

                # QUAN TRỌNG:
                #
                # Chỉ gọi VietOCR MỘT LẦN.
                #
                # Code trước của bạn đang predict()
                # và append block hai lần.
                text, score, truncated = (
                    self.line_model.predict(
                        crop
                    )
                )

                if not text:
                    continue

                source = (
                    "vietocr-truncated"
                    if truncated
                    else "vietocr"
                )

                blocks.append(
                    Block(
                        str(
                            len(blocks)
                        ),
                        padded_box,
                        normalize(
                            text
                        ),
                        source,
                        float(
                            score
                        ),
                    )
                )

            blocks = (
                self._sort_handwriting_blocks(
                    blocks,
                    columns,
                    image.width,
                )
            )

        # =========================================================
        # ID cuối cùng
        # =========================================================

        for i, block in enumerate(
            blocks
        ):
            block.id = f"b{i}"

        return blocks