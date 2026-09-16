import argparse
import json
import os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description="VietDoc OCR")
    commands=parser.add_subparsers(dest="command",required=True)
    serve=commands.add_parser("serve");serve.add_argument("--port",type=int,default=8000)
    run=commands.add_parser("extract");run.add_argument("input",type=Path);run.add_argument("--output",type=Path,default=Path("workspace/cli"))
    run.add_argument("--mode",choices=["auto","native","printed","handwriting","line"],default="auto")
    run.add_argument("--pages",default="");run.add_argument("--columns",type=int,choices=[1,2],default=1)
    args=parser.parse_args()
    if args.command=="serve":
        import uvicorn
        uvicorn.run("vietdoc.server:app",host="127.0.0.1",port=args.port,workers=1)
    else:
        from .engines import OCREngine
        from .pipeline import process
        from .core import export_bytes
        result=process(args.input,args.output,OCREngine(Path.cwd()),args.mode,args.pages,args.columns)
        for fmt in ("txt","md","json","jsonl"):
            data,_,ext=export_bytes(result,fmt);(args.output/f"result.{ext}").write_bytes(data)
        errors=sum(bool(p["error"]) for p in result["pages"])
        print(json.dumps({"pages":len(result["pages"]),"page_errors":errors,"output":str(args.output.resolve())},ensure_ascii=False))
        if errors:raise SystemExit(2)


if __name__=="__main__":main()
