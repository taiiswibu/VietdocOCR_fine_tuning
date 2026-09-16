"""Stream gated HF dataset to PNG + JSONL. Credentials only from HF_TOKEN env/Kaggle Secrets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from collections import Counter
import unicodedata

from PIL import Image, ImageOps
import yaml


def split_for(text,seed,val_fraction):
    # Group identical normalized transcriptions. Writer IDs are not assumed available.
    group=" ".join(unicodedata.normalize("NFC",text).split())
    h=hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()
    return ("val" if int(h[:8],16)/2**32 < val_fraction else "train"),h


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=Path("data/handwriting"))
    p.add_argument("--dataset",default="5CD-AI/Viet-Handwriting-OCR-v2");p.add_argument("--revision",default="main")
    p.add_argument("--image-column");p.add_argument("--text-column");p.add_argument("--val-fraction",type=float,default=.05)
    p.add_argument("--seed",type=int,default=42);p.add_argument("--limit-train",type=int,default=0)
    p.add_argument("--config",type=Path,default=Path("configs/vietocr.yml"));p.add_argument("--accept-noncommercial",action="store_true")
    a=p.parse_args()
    if not a.accept_noncommercial:p.error("Dataset công bố CC BY-NC 4.0. Đọc điều kiện và thêm --accept-noncommercial cho dự án phi thương mại.")
    if not 0<a.val_fraction<.5:p.error("val-fraction phải nằm trong (0,0.5)")
    if any((a.output/f"{s}.jsonl").exists() for s in ("train","val","test")):
        p.error("Thư mục đã có manifest; dùng thư mục mới để không ghi đè lần chuẩn bị dữ liệu trước.")
    from datasets import load_dataset
    from huggingface_hub import HfApi
    revision=HfApi().dataset_info(a.dataset,revision=a.revision,token=os.getenv("HF_TOKEN")).sha
    a.output.mkdir(parents=True,exist_ok=True);(a.output/"images").mkdir(exist_ok=True)
    files={s:(a.output/f"{s}.jsonl").open("w",encoding="utf-8") for s in ("train","val","test")}
    counts=Counter();rejected=[];seen=set();chars=set();columns=None
    try:
        for source_split in ("test","train"):
            ds=load_dataset(a.dataset,split=source_split,revision=revision,streaming=True,token=os.getenv("HF_TOKEN"))
            for i,row in enumerate(ds):
                if source_split=="train" and a.limit_train and i>=a.limit_train:break
                if columns is None:
                    image_col=a.image_column or next((k for k,v in row.items() if isinstance(v,Image.Image)),None)
                    text_col=a.text_column or next((k for k in ("text","label","transcription","sentence","ground_truth") if isinstance(row.get(k),str)),None)
                    if not image_col or not text_col:raise ValueError(f"Không xác định được cột. Các cột: {list(row)}. Đặt --image-column / --text-column.")
                    columns=(image_col,text_col)
                try:
                    image=ImageOps.exif_transpose(row[columns[0]]).convert("RGB")
                    text=unicodedata.normalize("NFC",str(row[columns[1]])).strip()
                    if not text or image.width<3 or image.height<3:raise ValueError("empty_text_or_tiny_image")
                    digest=hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
                    if digest in seen:
                        counts["duplicates_removed"]+=1;continue
                    seen.add(digest)
                    destination,group=("test",digest) if source_split=="test" else split_for(text,a.seed,a.val_fraction)
                    name=f"images/{source_split}-{i:07d}.png";image.save(a.output/name)
                    record={"id":f"{source_split}-{i}","image":name,"text":text,"sha256":digest,"group":group,"source_split":source_split}
                    files[destination].write(json.dumps(record,ensure_ascii=False)+"\n");counts[destination]+=1
                    if destination=="train":chars.update(text)
                    if i%2000==0:print(source_split,i,dict(counts),flush=True)
                except Exception as exc:
                    rejected.append({"split":source_split,"index":i,"reason":str(exc)})
    finally:
        for f in files.values():f.close()
    base=yaml.safe_load(a.config.read_text(encoding="utf-8"))
    additions="".join(sorted(chars-set(base["vocab"])))
    base["vocab"]+=additions
    (a.output/"config.yml").write_text(yaml.safe_dump(base,allow_unicode=True,sort_keys=False),encoding="utf-8")
    report={"dataset":a.dataset,"revision":revision,"seed":a.seed,"val_fraction":a.val_fraction,"columns":columns,
            "counts":dict(counts),"rejected":rejected,"appended_vocab":additions,"limited_train":a.limit_train,
            "split_policy":"Official test reserved first; exact RGB duplicates removed across splits; train/val grouped by normalized transcription. Writer-disjointness unverified."}
    (a.output/"preparation.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    if not all(counts[s] for s in ("train","val","test")):raise RuntimeError("Có tập rỗng. Kiểm tra preparation.json; dùng nhiều mẫu hơn hoặc kiểm tra schema.")
    print(json.dumps({"counts":dict(counts),"revision":revision,"vocab_added":len(additions)},ensure_ascii=False))


if __name__=="__main__":main()
