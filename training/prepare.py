"""Stream a public HF handwriting dataset to filtered PNG + JSONL manifests."""
import argparse
import hashlib
import json
from pathlib import Path
from collections import Counter
import unicodedata

from PIL import Image, ImageOps
import yaml


def clean_label(value):
    """NFC-normalize labels, collapse whitespace, and remove control chars."""
    text=unicodedata.normalize("NFC",str(value))
    text="".join(c for c in text if not unicodedata.category(c).startswith("C"))
    return " ".join(text.split())


def split_for(value,seed,val_fraction):
    """Assign a stable group (preferably writer ID) wholly to train or validation."""
    group=clean_label(value)
    h=hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()
    return ("val" if int(h[:8],16)/2**32 < val_fraction else "train"),h


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=Path("data/handwriting"))
    p.add_argument("--dataset",default="blue7012/UIT_HWDB");p.add_argument("--revision",default="main")
    p.add_argument("--image-column");p.add_argument("--text-column");p.add_argument("--group-column",default="writer_id")
    p.add_argument("--val-fraction",type=float,default=.10)
    p.add_argument("--seed",type=int,default=42);p.add_argument("--limit-train",type=int,default=0)
    p.add_argument("--test-only",action="store_true",help="Download only the official test split for evaluation")
    p.add_argument("--max-label-length",type=int,default=160,help="Keep line-level labels no longer than this many characters")
    p.add_argument("--min-aspect-ratio",type=float,default=1.2,help="Keep images whose width/height indicates a text line")
    p.add_argument("--min-new-char-frequency",type=int,default=5)
    p.add_argument("--max-new-chars",type=int,default=64,help="Fail before training when vocab expansion is unexpectedly large")
    p.add_argument("--config",type=Path,default=Path("configs/vietocr.yml"));p.add_argument("--accept-license",action="store_true")
    p.add_argument("--declared-license",default="cc-by-4.0")
    a=p.parse_args()
    if not a.accept_license:p.error("Đọc dataset card/license rồi thêm --accept-license để ghi nhận việc chấp nhận điều khoản.")
    if not 0<a.val_fraction<.5:p.error("val-fraction phải nằm trong (0,0.5)")
    required=("test",) if a.test_only else ("train","val","test")
    if any((a.output/f"{s}.jsonl").exists() for s in required):
        p.error("Thư mục đã có manifest; dùng thư mục mới để không ghi đè lần chuẩn bị dữ liệu trước.")
    if a.min_new_char_frequency<1 or a.max_new_chars<0:p.error("Ngưỡng vocabulary không hợp lệ")
    if a.max_label_length<1 or a.min_aspect_ratio<=0:p.error("Bộ lọc line-level không hợp lệ")
    from datasets import load_dataset
    from huggingface_hub import HfApi
    revision=HfApi().dataset_info(a.dataset,revision=a.revision).sha
    a.output.mkdir(parents=True,exist_ok=True);(a.output/"images").mkdir(exist_ok=True)
    files={s:(a.output/f"{s}.jsonl").open("w",encoding="utf-8") for s in required}
    counts=Counter();rejected=[];seen=set();chars=Counter();columns=None;group_column=None
    try:
        for source_split in (("test",) if a.test_only else ("test","train")):
            ds=load_dataset(a.dataset,split=source_split,revision=revision,streaming=True)
            for i,row in enumerate(ds):
                if source_split=="train" and a.limit_train and i>=a.limit_train:break
                if columns is None:
                    image_col=a.image_column or next((k for k,v in row.items() if isinstance(v,Image.Image)),None)
                    text_col=a.text_column or next((k for k in ("text","label","transcription","sentence","ground_truth") if isinstance(row.get(k),str)),None)
                    if not image_col or not text_col:raise ValueError(f"Không xác định được cột. Các cột: {list(row)}. Đặt --image-column / --text-column.")
                    group_column=a.group_column if a.group_column and a.group_column in row else None
                    if not a.test_only and not group_column:
                        raise ValueError(f"Không tìm thấy group column '{a.group_column}'. Các cột: {list(row)}. Cần writer_id để split chống leakage.")
                    columns=(image_col,text_col)
                try:
                    image=ImageOps.exif_transpose(row[columns[0]]).convert("RGB")
                    text=clean_label(row[columns[1]])
                    if not text or image.width<3 or image.height<3:raise ValueError("empty_text_or_tiny_image")
                    if len(text)>a.max_label_length:raise ValueError(f"label_too_long:{len(text)}")
                    aspect=image.width/image.height
                    if aspect<a.min_aspect_ratio:raise ValueError(f"non_line_aspect:{aspect:.3f}")
                    digest=hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
                    if digest in seen:
                        counts["duplicates_removed"]+=1;continue
                    seen.add(digest)
                    raw_group=row.get(group_column) if group_column else digest
                    normalized_group=clean_label(raw_group)
                    destination,group=("test",normalized_group) if source_split=="test" else split_for(normalized_group,a.seed,a.val_fraction)
                    name=f"images/{source_split}-{i:07d}.png";image.save(a.output/name)
                    record={"id":f"{source_split}-{i}","image":name,"text":text,"sha256":digest,
                            "group":group,"writer_id":normalized_group,"source_split":source_split,
                            "width":image.width,"height":image.height,"aspect_ratio":aspect}
                    files[destination].write(json.dumps(record,ensure_ascii=False)+"\n");counts[destination]+=1
                    if destination=="train":chars.update(text)
                    if i%2000==0:print(source_split,i,dict(counts),flush=True)
                except Exception as exc:
                    rejected.append({"split":source_split,"index":i,"reason":str(exc)})
    finally:
        for f in files.values():f.close()
    base=yaml.safe_load(a.config.read_text(encoding="utf-8"))
    additions="" if a.test_only else "".join(sorted(c for c,n in chars.items() if c not in base["vocab"] and n>=a.min_new_char_frequency))
    vocab_audit=[{"char":c,"codepoint":f"U+{ord(c):04X}","name":unicodedata.name(c,"UNKNOWN"),
                  "category":unicodedata.category(c),"train_frequency":chars[c]} for c in additions]
    if not a.test_only:
        base["vocab"]+=additions
        (a.output/"config.yml").write_text(yaml.safe_dump(base,allow_unicode=True,sort_keys=False),encoding="utf-8")
    report={"dataset":a.dataset,"revision":revision,"declared_license":a.declared_license,
            "seed":a.seed,"val_fraction":a.val_fraction,"columns":columns,"group_column":group_column,
            "counts":dict(counts),"rejected":rejected,"appended_vocab":additions,"appended_vocab_audit":vocab_audit,
            "min_new_char_frequency":a.min_new_char_frequency,"max_new_chars":a.max_new_chars,
            "line_filter":{"max_label_length":a.max_label_length,"min_aspect_ratio":a.min_aspect_ratio},
            "limited_train":a.limit_train,"test_only":a.test_only,
            "split_policy":"Official test reserved first and filtered with the declared line rule; exact RGB duplicates removed across splits; train/validation assigned by writer_id hash (writer-disjoint)."}
    (a.output/"preparation.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    if not all(counts[s] for s in required):raise RuntimeError("Có tập rỗng. Kiểm tra preparation.json; dùng nhiều mẫu hơn hoặc kiểm tra schema.")
    if not a.test_only and len(additions)>a.max_new_chars:
        raise RuntimeError(f"Vocab cần thêm {len(additions)} ký tự, vượt ngưỡng {a.max_new_chars}. Xem appended_vocab_audit; chỉ tăng --max-new-chars sau khi kiểm tra nhãn.")
    print(json.dumps({"counts":dict(counts),"revision":revision,"vocab_added":len(additions)},ensure_ascii=False))


if __name__=="__main__":main()
