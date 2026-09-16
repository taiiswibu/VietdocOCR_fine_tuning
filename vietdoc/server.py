from __future__ import annotations
import io
import json
import os
import re
import time
import uuid
import zipfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image

from .core import export_bytes, normalize, reading_order, Block
from .engines import OCREngine
from .pipeline import process

ROOT=Path(os.getenv("VIETDOC_ROOT",Path.cwd())).resolve()
WORK=Path(os.getenv("VIETDOC_WORKSPACE",ROOT/"workspace")).resolve();WORK.mkdir(parents=True,exist_ok=True)
engine=OCREngine(ROOT)
pool=ThreadPoolExecutor(max_workers=1)
guard=threading.RLock()
model_guard=threading.Lock()
jobs={}
app=FastAPI(title="VietDoc OCR",version="1.0.0")
STATIC=Path(__file__).parent/"static"
app.mount("/static",StaticFiles(directory=STATIC),name="static")


@app.middleware("http")
async def local_only(request: Request,call_next):
    # Loopback binding + same-origin writes. No public hosting/auth claims.
    host=request.headers.get("host","").split(":")[0]
    if host not in ("127.0.0.1","localhost","testserver"):
        return Response("Local access only",status_code=403)
    origin=request.headers.get("origin")
    if request.method not in ("GET","HEAD") and origin:
        from urllib.parse import urlparse
        if urlparse(origin).netloc != request.headers.get("host"):
            return Response("Cross-origin write denied",status_code=403)
    response=await call_next(request)
    response.headers["X-Content-Type-Options"]="nosniff"
    return response


def directory(job_id):
    if not re.fullmatch(r"[a-f0-9]{32}",job_id):raise HTTPException(404,"Không tìm thấy tài liệu")
    p=WORK/job_id
    if not p.exists():raise HTTPException(404,"Không tìm thấy tài liệu")
    return p


def write_json(path,value):
    temp=path.with_suffix(".tmp")
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")
    temp.replace(path)


def read_document(job_id):
    p=directory(job_id)/"result.json"
    if not p.exists():raise HTTPException(409,"Tài liệu đang chờ xử lý")
    return json.loads(p.read_text(encoding="utf-8"))


def assert_idle(job_id):
    if jobs.get(job_id,{}).get("status") in ("queued","running"):
        raise HTTPException(409,"Chờ xử lý xong trước khi chỉnh sửa/xuất dữ liệu.")


def run_job(job_id,path,mode,pages,columns,filename):
    def update(doc):
        doc["filename"]=filename
        with guard:
            write_json(directory(job_id)/"result.json",doc)
            jobs[job_id]["processed"]=len(doc["pages"])
    with guard:jobs[job_id]["status"]="running"
    try:
        with model_guard:
            doc=process(path,directory(job_id),engine,mode,pages,columns,update)
        doc["filename"]=filename
        with guard:
            write_json(directory(job_id)/"result.json",doc)
            jobs[job_id]["status"]="done"
            jobs[job_id]["page_errors"]=sum(bool(p["error"]) for p in doc["pages"])
    except Exception as exc:
        with guard:jobs[job_id].update(status="error",error=f"{type(exc).__name__}: {exc}")


@app.get("/")
def index():return FileResponse(STATIC/"index.html")


@app.get("/api/health")
def health():
    return {"status":"ok","handwriting_ready":all((ROOT/"models/handwriting"/f).exists() for f in ("model.pth","config.yml")),
            "printed_ready":all((ROOT/"models/easyocr"/f).exists() for f in ("craft_mlt_25k.pth","latin_g2.pth"))}


@app.post("/api/jobs")
async def upload(file: UploadFile=File(...), mode: str=Form("auto"), pages: str=Form(""), columns: int=Form(1)):
    if mode not in ("auto","native","printed","handwriting","line") or columns not in (1,2):
        raise HTTPException(400,"Chế độ không hợp lệ")
    filename=Path((file.filename or "document").replace("\\","/")).name
    suffix=Path(filename).suffix.lower()
    if suffix not in (".pdf",".png",".jpg",".jpeg",".webp"):
        raise HTTPException(400,"Chỉ nhận PDF, PNG, JPG, WEBP")
    with guard:
        if sum(j["status"] in ("queued","running") for j in jobs.values())>=8:
            raise HTTPException(429,"Hàng đợi đầy. Chờ tài liệu trước xử lý xong.")
        job_id=uuid.uuid4().hex
        jobs[job_id]={"id":job_id,"filename":filename,"status":"queued","processed":0,"created":time.time()}
    folder=WORK/job_id;folder.mkdir()
    path=folder/("source"+suffix); size=0
    try:
        with path.open("wb") as out:
            while chunk:=await file.read(1024*1024):
                size+=len(chunk)
                if size>25*1024*1024:raise HTTPException(413,"Tối đa 25 MB mỗi file")
                out.write(chunk)
        if not size:raise HTTPException(400,"File trống")
        pool.submit(run_job,job_id,path,mode,pages,columns,filename)
    except Exception:
        import shutil
        with guard:jobs.pop(job_id,None)
        shutil.rmtree(folder,ignore_errors=True)
        raise
    return {"id":job_id}


@app.get("/api/jobs")
def history():
    with guard:
        entries=dict(jobs)
        for f in WORK.glob("*/result.json"):
            if f.parent.name not in entries:
                try:
                    d=json.loads(f.read_text(encoding="utf-8"))
                    entries[f.parent.name]={"id":f.parent.name,"filename":d["filename"],"status":"saved" if d.get("processing_complete",True) else "interrupted","processed":len(d["pages"]),"created":f.stat().st_mtime}
                except (ValueError,KeyError):continue
        return sorted(entries.values(),key=lambda j:j["created"],reverse=True)[:100]


@app.get("/api/jobs/{job_id}")
def status(job_id):
    with guard:
        directory(job_id)
        value=dict(jobs.get(job_id,{"id":job_id,"status":"saved"}))
        if (directory(job_id)/"result.json").exists():
            value["document"]=read_document(job_id)
            if job_id not in jobs:
                value["status"]="saved" if value["document"].get("processing_complete",True) else "interrupted"
                value["page_errors"]=sum(bool(p.get("error")) for p in value["document"]["pages"])
                if value["status"]=="interrupted":value["error"]="Phiên trước bị gián đoạn; đây là kết quả một phần. Nạp lại file để xử lý đủ trang."
        return value


@app.get("/api/jobs/{job_id}/pages/{number}/image")
def page_image(job_id,number:int):
    path=directory(job_id)/f"page-{number}.jpg"
    if not path.exists():raise HTTPException(404,"Không tìm thấy ảnh trang")
    return FileResponse(path)


class Edit(BaseModel):
    text: str=Field(max_length=200000)
    reviewed: bool=True
    block_id: str | None=None


@app.patch("/api/jobs/{job_id}/pages/{number}")
def edit_page(job_id,number:int,body:Edit):
    with guard:
        assert_idle(job_id);doc=read_document(job_id)
        page=next((p for p in doc["pages"] if p["number"]==number),None)
        if page is None:raise HTTPException(404,"Trang không tồn tại")
        if body.block_id:
            block=next((b for b in page["blocks"] if b["id"]==body.block_id),None)
            if block is None:raise HTTPException(404,"Vùng chữ không tồn tại")
            if page["edited_text"] is not None:
                raise HTTPException(409,"Trang đã sửa toàn văn. Dùng trang mới để gán nhãn từng dòng tránh mất chỉnh sửa.")
            if block["original_text"] is None:block["original_text"]=block["text"]
            block.update(text=normalize(body.text),reviewed=body.reviewed)
            page["text"]="\n".join(b["text"] for b in page["blocks"])
            page["reviewed"]=all(b["reviewed"] for b in page["blocks"])
        else:
            page.update(edited_text=normalize(body.text),text=normalize(body.text),reviewed=body.reviewed)
        write_json(directory(job_id)/"result.json",doc)
        return page


class Region(BaseModel):
    bbox: list[float]=Field(min_length=4,max_length=4)
    mode: str="handwriting"
    single_line: bool=True


@app.post("/api/jobs/{job_id}/pages/{number}/region")
def region(job_id,number:int,body:Region):
    if body.mode not in ("handwriting","printed"):raise HTTPException(400,"Chế độ không hợp lệ")
    with model_guard, guard:
        assert_idle(job_id);doc=read_document(job_id)
        page=next((p for p in doc["pages"] if p["number"]==number),None)
        if page is None:raise HTTPException(404,"Trang không tồn tại")
        if page["edited_text"] is not None:raise HTTPException(409,"Trang đã sửa toàn văn. Mở lại file thành lượt mới để OCR lại vùng.")
        image=Image.open(directory(job_id)/f"page-{number}.jpg").convert("RGB")
        x0,y0,x1,y1=body.bbox
        if not (0<=x0<x1<=image.width and 0<=y0<y1<=image.height):raise HTTPException(400,"Vùng chọn ngoài ảnh")
        try:
            blocks=engine.run(image.crop((int(x0),int(y0),int(x1),int(y1))),body.mode,single_line=body.single_line)
        except Exception as e:raise HTTPException(422,str(e)) from e
        if not blocks:raise HTTPException(422,"Không nhận dạng được chữ trong vùng")
        # Replace boxes whose centers lie in selected region, preserving other boxes.
        retained=[Block(**b) for b in page["blocks"] if not(x0<=(b["bbox"][0]+b["bbox"][2])/2<=x1 and y0<=(b["bbox"][1]+b["bbox"][3])/2<=y1)]
        for b in blocks:
            b.bbox=[b.bbox[0]+x0,b.bbox[1]+y0,b.bbox[2]+x0,b.bbox[3]+y0];b.id=uuid.uuid4().hex[:12]
        from dataclasses import asdict
        page["blocks"]=[asdict(b) for b in reading_order(retained+blocks)]
        page["text"]="\n".join(b["text"] for b in page["blocks"]);page["reviewed"]=False
        page["error"]=None;page["method"]="mixed-region"
        write_json(directory(job_id)/"result.json",doc)
        return page


@app.get("/api/jobs/{job_id}/export/{format}")
def export(job_id,format):
    with guard:
        assert_idle(job_id);doc=read_document(job_id)
        if format=="training":
            memory=io.BytesIO();rows=[]
            with zipfile.ZipFile(memory,"w",zipfile.ZIP_DEFLATED) as archive:
                for page in doc["pages"]:
                    if page["edited_text"] is not None:continue
                    image=Image.open(directory(job_id)/f"page-{page['number']}.jpg").convert("RGB")
                    for block in page["blocks"]:
                        if not block["reviewed"] or not block["text"]:continue
                        name=f"images/p{page['number']}-{block['id']}.png"
                        cropped=io.BytesIO();image.crop(tuple(int(v) for v in block["bbox"])).save(cropped,format="PNG")
                        archive.writestr(name,cropped.getvalue())
                        rows.append({"image":name,"text":block["text"],"group":doc["document_id"],"page":page["number"],"source":block["source"],"human_reviewed":True})
                if not rows:raise HTTPException(400,"Chọn từng dòng và lưu nhãn đã kiểm tra trước khi xuất training. Sửa toàn văn không tạo nhãn dòng.")
                archive.writestr("labels.jsonl","\n".join(json.dumps(r,ensure_ascii=False) for r in rows))
                archive.writestr("README.txt","Nhãn do người dùng duyệt. Kiểm tra crop chứa đúng một dòng trước khi huấn luyện. Không trộn với test. Giữ group khi chia tập.")
            return Response(memory.getvalue(),media_type="application/zip",headers={"Content-Disposition":'attachment; filename="reviewed-lines.zip"'})
        try:data,mime,ext=export_bytes(doc,format)
        except ValueError as e:raise HTTPException(400,str(e)) from e
        return Response(data,media_type=mime,headers={"Content-Disposition":f'attachment; filename="vietdoc-{job_id[:8]}.{ext}"'})
