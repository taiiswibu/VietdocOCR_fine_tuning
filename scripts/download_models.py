"""Explicit one-time downloads. Application itself never downloads weights."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
MODELS={
    "base":("doanhm/vgg_transformer","4de5b0429954fdac3dee3173978c062b3bc4d534","vgg_transformer.pth"),
    "handwriting":("vudang449/vietocr-vietnamese-handwriting","f1973b930f907c378f6e3152ce9d8658724b228a","best_ep11.pth")
}

EXPECTED_SHA256 = {
    "base": "380512193a8b6cbf6fad80deacdc9b6939d10d473d199892fc6408d13775ea59",
    "handwriting": "f73bc43c152f882a7de3adc38be93901070ecc4b1bbf6ffa02440fd0906fa3ff",
}


def download(name):
    repo,revision,filename=MODELS[name];dest=ROOT/"models"/name;dest.mkdir(parents=True,exist_ok=True)
    target=dest/"model.pth"
    if target.exists() and (dest/"config.yml").exists():
        meta=dest/"provenance.json"
        if not meta.exists() or json.loads(meta.read_text(encoding="utf-8")).get("sha256") != hashlib.sha256(target.read_bytes()).hexdigest():
            print(f"Preserving existing/custom model in {dest}; no overwrite.");return
    if not target.exists():
        print(f"Downloading {name}: {repo}@{revision[:8]}",flush=True)
        temp=target.with_suffix(".download")
        with urllib.request.urlopen(f"https://huggingface.co/{repo}/resolve/{revision}/{filename}",timeout=120) as response,temp.open("wb") as out:
            shutil.copyfileobj(response,out,length=1024*1024)
        if hashlib.sha256(temp.read_bytes()).hexdigest() != EXPECTED_SHA256[name]:
            raise RuntimeError(f"Downloaded {name} checksum mismatch. Retry download; existing model preserved.")
        temp.replace(target)
    shutil.copy2(ROOT/"configs/vietocr.yml",dest/"config.yml")
    digest=hashlib.sha256(target.read_bytes()).hexdigest()
    (dest/"provenance.json").write_text(json.dumps({"repository":repo,"revision":revision,"file":filename,"sha256":digest,
       "kind":"external_pretrained" if name=="base" else "external_finetuned_baseline",
       "note":"Không phải model do người dùng fine-tune. Dùng cho nghiên cứu/portfolio phi thương mại; xem điều khoản dữ liệu gốc."},ensure_ascii=False,indent=2),encoding="utf-8")
    print(name,"saved",target,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument("--base",action="store_true");p.add_argument("--handwriting",action="store_true");p.add_argument("--easyocr",action="store_true");p.add_argument("--all",action="store_true")
    a=p.parse_args()
    if not any(vars(a).values()):p.error("Chọn --all hoặc --base / --handwriting / --easyocr")
    for name in MODELS:
        if a.all or getattr(a,name):download(name)
    if a.all or a.easyocr:
        import easyocr
        easyocr.Reader(["vi","en"],gpu=False,model_storage_directory=str(ROOT/"models/easyocr"),download_enabled=True,verbose=False)
        print("EasyOCR detection + Latin recognition ready")


if __name__=="__main__":main()
