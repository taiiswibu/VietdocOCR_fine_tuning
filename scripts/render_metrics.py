"""Render the committed experiment chart from docs/experiment-results.json."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",type=Path,default=Path("docs/experiment-results.json"))
    parser.add_argument("--output",type=Path,default=Path("docs/images/training-metrics.png"))
    args=parser.parse_args()
    result=json.loads(args.input.read_text(encoding="utf-8"))
    if result.get("status") == "pending_new_kaggle_run":
        raise SystemExit("No UIT-HWDB metrics yet. Run the Kaggle notebook and use comparison.json.")
    pretrained=result.get("pretrained") or result["before"]["metrics"]
    finetuned=result.get("fine_tuned") or result["after"]["metrics"]
    labels=["CER ↓","WER ↓","Exact match ↑"]
    left=[pretrained["cer"]*100,pretrained["wer_whitespace"]*100,pretrained["exact_match"]*100]
    right=[finetuned["cer"]*100,finetuned["wer_whitespace"]*100,finetuned["exact_match"]*100]

    plt.rcParams.update({"font.size":11,"axes.titleweight":"bold","axes.edgecolor":"#cbd5d1"})
    fig,ax=plt.subplots(figsize=(10.5,5.8),facecolor="#f7faf8")
    ax.set_facecolor("#f7faf8")
    x=np.arange(len(labels));width=.35
    prebars=ax.bar(x-width/2,left,width,label="Pretrained",color="#087966")
    finbars=ax.bar(x+width/2,right,width,label="Fine-tuned",color="#df8a2b")
    ax.bar_label(prebars,fmt="%.2f%%",padding=4,color="#17332e")
    ax.bar_label(finbars,fmt="%.2f%%",padding=4,color="#17332e")
    ax.set_title("Pretrained vs Fine-tuned VietOCR",loc="left",fontsize=18,pad=20)
    sample_count=pretrained.get("samples", result.get("samples", "?"))
    ax.text(0,1.02,f"UIT-HWDB official-test line subset · n={sample_count}",transform=ax.transAxes,color="#60736e")
    ax.set_ylabel("Percent")
    ax.set_xticks(x,labels)
    ax.set_ylim(0,120)
    ax.grid(axis="y",alpha=.18)
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(frameon=False,loc="upper left")
    ax.text(1,-.18,"CER/WER: lower is better  ·  Exact match: higher is better",
            transform=ax.transAxes,ha="right",color="#60736e",fontsize=9)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=180,bbox_inches="tight",facecolor=fig.get_facecolor())
    print(args.output)


if __name__=="__main__":main()
