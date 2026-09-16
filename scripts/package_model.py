import argparse
import json
import zipfile
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument("run",type=Path);p.add_argument("--output",type=Path,default=Path("handwriting-model.zip"));a=p.parse_args()
    for name in ("model.pth","config.yml","run.json"):
        if not (a.run/name).exists():p.error(f"Thiếu {name}; huấn luyện chưa tạo bundle hoàn chỉnh")
    status=json.loads((a.run/"run.json").read_text())
    if not status.get("training_complete"):p.error("Lần chạy diagnostic/chưa hoàn thành. Resume train trước khi export bundle final.")
    with zipfile.ZipFile(a.output,"w",zipfile.ZIP_DEFLATED) as z:
        for name in ("model.pth","config.yml","run.json","metrics.jsonl"):
            if (a.run/name).exists():z.write(a.run/name,f"handwriting/{name}")
        z.writestr("handwriting/USAGE.txt","Giải nén thư mục handwriting vào models/. Khởi động lại app. Giữ bản baseline ở vị trí khác để so sánh. Dữ liệu huấn luyện có điều khoản phi thương mại.")
    print(a.output.resolve())


if __name__=="__main__":main()
