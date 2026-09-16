from __future__ import annotations

import hashlib
import io
import json
import re
import unicodedata
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any

import fitz
from PIL import Image, ImageOps


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text).replace("\x00", "").strip()


def content_id(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class Block:
    id: str
    bbox: list[float]
    text: str
    source: str
    score: float | None = None
    reviewed: bool = False
    original_text: str | None = None


@dataclass
class Page:
    number: int
    width: int
    height: int
    blocks: list[Block] = field(default_factory=list)
    method: str = "pending"
    edited_text: str | None = None
    reviewed: bool = False
    error: str | None = None
    seconds: float = 0.0

    @property
    def text(self):
        return self.edited_text if self.edited_text is not None else "\n".join(b.text for b in self.blocks)

    def to_dict(self):
        return {**asdict(self), "text": self.text}


def parse_pages(expression: str, count: int) -> list[int]:
    if not expression.strip():
        return list(range(count))
    selected = set()
    for part in expression.split(","):
        match = re.fullmatch(r"\s*(\d+)(?:\s*-\s*(\d+))?\s*", part)
        if not match:
            raise ValueError("Trang phải có dạng 1-5,8,10. Để trống để chọn tất cả.")
        a = int(match[1]); b = int(match[2] or a)
        if not 1 <= a <= b <= count:
            raise ValueError(f"Khoảng trang phải nằm trong 1–{count}.")
        selected.update(range(a - 1, b))
    return sorted(selected)


def usable_text(text: str) -> bool:
    # A heuristic, not a quality score. Mixed pages must be reviewed / force-OCRed.
    s = text.strip()
    return len(s) >= 40 and sum(c.isalnum() for c in s) >= 20 and s.count("\ufffd") / max(1,len(s)) < .02


def open_document(path: Path, page_expression="", dpi=160):
    """Yield one page image and native blocks at a time; never rasterize whole PDF."""
    if path.suffix.lower() == ".pdf":
        with fitz.open(path) as doc:
            if doc.is_encrypted:
                raise ValueError("PDF được mã hóa. Hãy mở khóa bản sao trước khi nhập.")
            indices = parse_pages(page_expression, len(doc))
            if len(indices) > 300:
                raise ValueError("Tối đa 300 trang mỗi lượt. Hãy chọn khoảng trang.")
            for i in indices:
                page = doc[i]
                # Cap rendered dimensions and memory while retaining useful preview resolution.
                scale = min(dpi / 72, 2500 / max(page.rect.width, page.rect.height))
                pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                blocks = []
                rotation = page.rotation_matrix
                for item in page.get_text("dict", sort=True)["blocks"]:
                    for line in item.get("lines", []):
                        text = normalize("".join(s["text"] for s in line["spans"]))
                        if text:
                            r = fitz.Rect(line["bbox"]) * rotation
                            blocks.append(Block(f"p{i+1}-b{len(blocks)}", [r.x0*scale,r.y0*scale,r.x1*scale,r.y1*scale], text, "pdf-text"))
                yield i+1, image, blocks
    else:
        with Image.open(path) as src:
            if getattr(src, "n_frames", 1) > 1:
                raise ValueError("Ảnh nhiều frame chưa được hỗ trợ. Hãy chuyển thành PDF.")
            image = ImageOps.exif_transpose(src).convert("RGB")
            image.thumbnail((2500,2500), Image.Resampling.LANCZOS)
        parse_pages(page_expression, 1)
        yield 1, image, []


def reading_order(blocks: list[Block], columns=1, width=None):
    """Explicit 1/2-column mode; avoids promising arbitrary layout reconstruction."""
    if columns == 2 and width:
        # Full-width titles are ordered with the column band before the next spanning block.
        spanning = sorted([b for b in blocks if b.bbox[0] < width*.30 and b.bbox[2] > width*.70], key=lambda b:b.bbox[1])
        remaining = [b for b in blocks if b not in spanning]
        result=[]
        for header in spanning + [None]:
            cutoff=header.bbox[1] if header else float("inf")
            band=[b for b in remaining if (b.bbox[1]+b.bbox[3])/2 < cutoff]
            remaining=[b for b in remaining if b not in band]
            result.extend(sorted(band,key=lambda b: (int((b.bbox[0]+b.bbox[2])/2 >= width/2), b.bbox[1], b.bbox[0])))
            if header: result.append(header)
        return result
    return sorted(blocks, key=lambda b: (round(b.bbox[1]/8), b.bbox[0]))


def chunk_pages(pages: list[dict], document_id: str, filename: str, size=900, overlap=120):
    if size < 100 or not 0 <= overlap < size:
        raise ValueError("Chunk size >=100 và 0 <= overlap < chunk size.")
    for page in pages:
        text=normalize(page["text"])
        start=0; part=0
        while start < len(text):
            end=min(start+size,len(text))
            if end < len(text):
                boundary=text.rfind(" ",start+size//2,end)
                if boundary > start+overlap: end=boundary
            segment=text[start:end]
            yield {"id":f"{document_id[:16]}-p{page['number']}-c{part}","text":segment,
                   "metadata":{"document_id":document_id,"filename":filename,"page":page["number"],
                               "char_start":start,"char_end":end,"method":page["method"],
                               "reviewed":page.get("reviewed",False),"content_sha256":content_id(segment.encode())}}
            if end == len(text): break
            start=end-overlap; part+=1


def export_bytes(document: dict, format: str) -> tuple[bytes,str,str]:
    pages=document["pages"]
    if format == "txt":
        text="\n\n".join(f"--- Trang {p['number']} ---\n{p['text']}" for p in pages)
        return text.encode("utf-8"),"text/plain; charset=utf-8","txt"
    if format == "md":
        text="\n\n".join(f"## Trang {p['number']}\n\n{p['text']}" for p in pages)
        return text.encode("utf-8"),"text/markdown; charset=utf-8","md"
    if format == "json":
        return json.dumps(document,ensure_ascii=False,indent=2).encode(),"application/json","json"
    if format == "jsonl":
        text="\n".join(json.dumps(c,ensure_ascii=False) for c in chunk_pages(pages,document["document_id"],document["filename"]))
        return text.encode(),"application/x-ndjson","jsonl"
    raise ValueError("Định dạng không hỗ trợ.")
