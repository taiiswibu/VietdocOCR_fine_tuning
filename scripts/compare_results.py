import argparse,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument("baseline",type=Path);p.add_argument("finetuned",type=Path);a=p.parse_args()
    b=json.loads(a.baseline.read_text());f=json.loads(a.finetuned.read_text())
    if b['manifest_sha256']!=f['manifest_sha256'] or b['samples']!=f['samples']:raise ValueError("Hai kết quả không đánh giá trên cùng mẫu")
    for key in ("samples","cer","wer_whitespace","exact_match","latency_p50","latency_p95"):
        print(f"{key:20s} baseline={b.get(key)}  finetuned={f.get(key)}")
    print("CER reduction (relative):",(b['cer']-f['cer'])/b['cer'] if b['cer'] else None)
    if b['is_subset'] or f['is_subset']:print("NOTE: subset evaluation; do not claim full test results")


if __name__=="__main__":main()
