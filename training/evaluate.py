import argparse
import json
import time
import hashlib
from pathlib import Path
from PIL import Image
from vietdoc.recognizer import LineRecognizer
from training.metrics import summarize,distance,clean


def main():
    p=argparse.ArgumentParser();p.add_argument("--model",type=Path,required=True);p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--device",default="cpu");p.add_argument("--limit",type=int,default=0)
    a=p.parse_args();model=LineRecognizer(a.model,a.device)
    rows=[json.loads(l) for l in a.manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    total=len(rows)
    if a.limit:rows=rows[:a.limit]
    predictions=[];times=[]
    for i,row in enumerate(rows):
        with Image.open(a.manifest.parent/row["image"]) as im:image=im.convert("RGB")
        t=time.perf_counter();pred,score,truncated=model.predict(image);elapsed=time.perf_counter()-t;times.append(elapsed)
        predictions.append({"id":row.get("id",i),"truth":row["text"],"prediction":pred,"score":score,"truncated":truncated,"seconds":elapsed,
                            "cer":distance(clean(row["text"]),clean(pred))/max(1,len(clean(row["text"])))})
        if i%100==0:print(f"Evaluated {i+1}/{len(rows)}",flush=True)
    import numpy as np
    result={**summarize((r["truth"],r["prediction"]) for r in predictions),"full_manifest_samples":total,"is_subset":len(rows)!=total,
            "latency_p50":float(np.median(times)) if times else None,"latency_p95":float(np.percentile(times,95)) if times else None,
            "truncated_samples":sum(r["truncated"] for r in predictions),"model":str(a.model),"device":a.device,
            "manifest_sha256":hashlib.sha256(a.manifest.read_bytes()).hexdigest(),"model_sha256":hashlib.sha256((a.model/"model.pth").read_bytes()).hexdigest()}
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/"metrics.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    (a.output/"predictions.jsonl").write_text("\n".join(json.dumps(r,ensure_ascii=False) for r in predictions),encoding="utf-8")
    (a.output/"worst-50.json").write_text(json.dumps(sorted(predictions,key=lambda r:r["cer"],reverse=True)[:50],ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
