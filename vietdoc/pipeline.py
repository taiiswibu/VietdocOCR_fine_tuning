from pathlib import Path
import time
from .core import Page, open_document, usable_text, content_id, reading_order


def process(path, output_dir, engine, mode="auto", pages="", columns=1, progress=None):
    path=Path(path); output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    document={"processing_complete":False,"schema_version":"1.0","document_id":content_id(path.read_bytes()),"filename":path.name,"pages":[],
              "notes":["Scores OCR không phải xác suất chính xác đã hiệu chuẩn.","JSONL chứa text và metadata; chưa có embeddings."]}
    for number,image,native in open_document(path,pages):
        start=time.perf_counter()
        page=Page(number,image.width,image.height)
        image.save(output_dir/f"page-{number}.jpg",quality=94)
        try:
            if mode=="auto" and not native and image.convert("L").getextrema()==(255,255):
                page.method="blank"
            elif mode=="native" or (mode=="auto" and usable_text("\n".join(b.text for b in native))):
                page.blocks=reading_order(native,columns,image.width);page.method="pdf-text"
                if not native:page.error="Không tìm thấy lớp văn bản. Chọn Chữ in hoặc Chữ viết tay."
            else:
                selected="handwriting" if mode in ("handwriting","line") else "printed"
                page.blocks=engine.run(image,selected,columns,single_line=mode=="line")
                page.method=selected
                if not page.blocks:page.error="Không tìm thấy vùng chữ; thử khoanh vùng một dòng để nhận dạng."
        except Exception as exc:
            page.error=f"{type(exc).__name__}: {exc}";page.method="error"
        page.seconds=round(time.perf_counter()-start,3)
        document["pages"].append(page.to_dict())
        if progress:progress(document)
    document["processing_complete"]=True
    return document
