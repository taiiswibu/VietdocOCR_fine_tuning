"""Single-GPU fine-tuning with optimizer/AMP/RNG resume and held-out validation."""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from vietdoc.recognizer import load_config,create_model,forward_train,decode
from training.data import Lines,collate
from training.metrics import summarize


def atomic_save(value,path):
    temp=path.with_suffix(".tmp");torch.save(value,temp);temp.replace(path)


def initial_weights(model,path):
    state=torch.load(path,map_location="cpu",weights_only=True)
    state=state.get("model",state.get("state_dict",state))
    target=model.state_dict()
    # New train-only characters are appended; existing character indices must never move.
    added_classes=0
    for name in ("transformer.embed_tgt.weight","transformer.fc.weight","transformer.fc.bias"):
        if name in state and state[name].shape!=target[name].shape:
            old=state[name]
            if old.shape[0]>target[name].shape[0] or old.shape[1:]!=target[name].shape[1:]:raise ValueError("Vocab/checkpoint không tương thích")
            new=target[name].clone();new[:old.shape[0]]=old;state[name]=new
            if name=="transformer.fc.bias":added_classes=new.shape[0]-old.shape[0]
    model.load_state_dict(state,strict=True)
    return added_classes


def validate(model,ds,vocab,device,max_samples):
    pairs=[];start=time.time()
    for i in range(min(len(ds),max_samples or len(ds))):
        image,_,truth,_=ds[i]
        pred,_,_=decode(model,torch.from_numpy(image)[None].to(device),vocab,ds.config.get("decode_max_tokens",256))
        pairs.append((truth,pred))
    return {**summarize(pairs),"seconds":time.time()-start}


def main():
    p=argparse.ArgumentParser();p.add_argument("--data",type=Path,default=Path("data/handwriting"));p.add_argument("--output",type=Path,default=Path("runs/handwriting"))
    p.add_argument("--initial",type=Path,default=Path("models/base/model.pth"));p.add_argument("--resume",type=Path)
    p.add_argument("--epochs",type=int,default=8);p.add_argument("--batch-size",type=int,default=8);p.add_argument("--accum",type=int,default=4)
    p.add_argument("--lr",type=float,default=1e-4);p.add_argument("--seed",type=int,default=42);p.add_argument("--device",default="cuda:0")
    p.add_argument("--val-samples",type=int,default=300);p.add_argument("--save-every",type=int,default=100);p.add_argument("--patience",type=int,default=3)
    p.add_argument("--max-updates",type=int,default=0,help="Diagnostic run; not a completed training run")
    p.add_argument("--allow-large-vocab-expansion",action="store_true",help="Acknowledge >64 new output classes after auditing labels")
    a=p.parse_args()
    if min(a.epochs,a.batch_size,a.accum,a.save_every)<1:p.error("epochs, batch-size, accum, save-every phải >=1")
    if a.device.startswith("cuda") and not torch.cuda.is_available():p.error("Không có CUDA; bật GPU Kaggle hoặc --device cpu để smoke test")
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(a.seed)
    torch.set_num_threads(4)
    config=load_config(a.data/"config.yml")
    model,vocab=create_model(config,a.device)
    train=Lines(a.data/"train.jsonl",config,vocab,augment=True);val=Lines(a.data/"val.jsonl",config,vocab)
    if not len(train) or not len(val):raise ValueError("Train/validation rỗng")
    if max(len(row['text']) for row in train.rows)>=config['transformer']['max_seq_length']-2:
        raise ValueError("Nhãn dài vượt positional encoding. Tách/loại mẫu có ghi nhận trước khi train.")
    signature=hashlib.sha256((a.data/"train.jsonl").read_bytes()+(a.data/"val.jsonl").read_bytes()+json.dumps(config,sort_keys=True).encode()).hexdigest()
    a.output.mkdir(parents=True,exist_ok=True)
    batches=math.ceil(len(train)/a.batch_size);updates_per_epoch=math.ceil(batches/a.accum)
    total_updates=updates_per_epoch*a.epochs
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01)
    warmup=max(1,int(total_updates*.05))
    def schedule(step):
        if step<warmup:return max(.01,(step+1)/warmup)
        return max(.05,.5*(1+math.cos(math.pi*min(1,(step-warmup)/max(1,total_updates-warmup)))))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
    amp=a.device.startswith("cuda");scaler=torch.amp.GradScaler("cuda",enabled=amp)
    start_epoch=next_batch=step=stale=0;best=float("inf");best_source="unset";added_classes=0
    # Keep two selections separate:
    # - best/model.pth: best deployment candidate, including the pretrained initialization.
    # - best_finetuned/best-finetuned.pth: best checkpoint that actually received updates.
    # This preserves a truthful before-vs-after research comparison even when fine-tuning regresses.
    best_finetuned=float("inf");best_finetuned_source="unset"
    if a.resume:
        ck=torch.load(a.resume,map_location="cpu",weights_only=True)
        if ck["data_signature"]!=signature:raise ValueError("Manifest/config thay đổi so với checkpoint")
        if ck["run_shape"]!=[a.batch_size,a.accum,a.epochs,a.seed]:raise ValueError("Resume yêu cầu giữ batch-size, accum, epochs và seed")
        model.load_state_dict(ck["model"]);optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"]);scaler.load_state_dict(ck["scaler"])
        start_epoch=ck["epoch"];next_batch=ck["next_batch"];step=ck["step"];best=ck["best_cer"];stale=ck["stale"]
        best_source=ck.get("best_source","unknown-resumed-checkpoint")
        best_finetuned=ck.get("best_finetuned_cer",float("inf"))
        best_finetuned_source=ck.get("best_finetuned_source","unset")
        random.setstate(ck["python_rng"]);torch.set_rng_state(ck["torch_rng"])
        if amp and ck.get("cuda_rng"):torch.cuda.set_rng_state_all(ck["cuda_rng"])
        ns=ck["numpy_rng"];np.random.set_state((ns[0],np.array(ns[1],dtype=np.uint32),ns[2],ns[3],ns[4]))
    else:
        if not a.initial.exists():p.error("Thiếu initial pretrained checkpoint. Chạy python scripts/download_models.py --base trước.")
        added_classes=initial_weights(model,a.initial)
        if added_classes>64 and not a.allow_large_vocab_expansion:
            raise ValueError(f"Output layer mở rộng {added_classes} lớp (>64). Audit preparation.json rồi chạy lại với --allow-large-vocab-expansion nếu hợp lệ.")
    metadata={**vars(a),"data_signature":signature,"torch":torch.__version__,"gpu":torch.cuda.get_device_name(0) if amp else "CPU","training_complete":False}
    (a.output/"run.json").write_text(json.dumps(metadata,default=str,indent=2),encoding="utf-8")
    (a.output/"config.yml").write_text(yaml.safe_dump(config,allow_unicode=True),encoding="utf-8")
    if not a.resume:
        # The pretrained initialization is a real candidate. Without this
        # comparison, epoch 1 wins against infinity even after catastrophic
        # degradation, which was the selection bug in the first experiment.
        initial_scores=validate(model,val,vocab,a.device,a.val_samples)
        initial_scores.update(source="pretrained-initialization",added_output_classes=added_classes)
        (a.output/"initial-validation.json").write_text(json.dumps(initial_scores,indent=2),encoding="utf-8")
        best=initial_scores["cer"];best_source="pretrained-initialization"
        atomic_save(model.state_dict(),a.output/"model.pth")
        print(json.dumps({"initial_validation":initial_scores}),flush=True)
    def checkpoint(epoch,batch):
        ns=np.random.get_state()
        atomic_save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"scaler":scaler.state_dict(),
                     "epoch":epoch,"next_batch":batch,"step":step,"best_cer":best,"best_source":best_source,
                     "best_finetuned_cer":best_finetuned,"best_finetuned_source":best_finetuned_source,
                     "stale":stale,"data_signature":signature,
                     "run_shape":[a.batch_size,a.accum,a.epochs,a.seed],"python_rng":random.getstate(),"torch_rng":torch.get_rng_state(),
                     "cuda_rng":torch.cuda.get_rng_state_all() if amp else [],"numpy_rng":[ns[0],ns[1].tolist(),ns[2],ns[3],ns[4]]},a.output/"last.pt")
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch,a.epochs):
        model.train();loss_sum=0.;seen_batches=0
        generator=torch.Generator().manual_seed(a.seed+epoch)
        order=torch.randperm(len(train),generator=generator).tolist()
        offset=next_batch if epoch==start_epoch else 0
        # Slice the permutation, rather than load/augment skipped samples, to preserve RNG on resume.
        loader=DataLoader(train,batch_size=a.batch_size,sampler=order[offset*a.batch_size:],num_workers=0,collate_fn=collate,generator=generator)
        for local_i,(images,inputs,targets,widths) in enumerate(loader):
            batch_i=offset+local_i
            images=images.to(a.device);inputs=inputs.to(a.device);targets=targets.to(a.device)
            group_start=(batch_i//a.accum)*a.accum
            group_size=min(a.accum,batches-group_start)
            with torch.autocast(device_type="cuda" if amp else "cpu",dtype=torch.float16,enabled=amp):
                logits=forward_train(model,images,inputs,inputs.T.eq(0),widths)
                loss=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),targets.reshape(-1),ignore_index=0,label_smoothing=.1)
            if not torch.isfinite(loss):raise RuntimeError("Loss không hữu hạn; kiểm tra dữ liệu/AMP. Checkpoint trước vẫn được giữ.")
            scaler.scale(loss/group_size).backward();loss_sum+=float(loss.detach());seen_batches+=1
            if (batch_i+1)%a.accum==0 or batch_i+1==batches:
                scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                previous_scale=scaler.get_scale();scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
                if scaler.get_scale()>=previous_scale:scheduler.step()
                step+=1
                if step%10==0:print(json.dumps({"epoch":epoch+1,"batch":batch_i+1,"updates":step,"loss":loss_sum/seen_batches,"lr":scheduler.get_last_lr()[0]}),flush=True)
                if step%a.save_every==0:checkpoint(epoch,batch_i+1)
                if a.max_updates and step>=a.max_updates:
                    checkpoint(epoch,batch_i+1);print("Diagnostic limit reached. Resume last.pt; no final model claimed.");return
        scores=validate(model,val,vocab,a.device,a.val_samples)
        row={"epoch":epoch+1,"updates":step,"train_loss":loss_sum/max(1,seen_batches),
             "learning_rate":scheduler.get_last_lr()[0],**scores}
        with (a.output/"metrics.jsonl").open("a",encoding="utf-8") as out:out.write(json.dumps(row)+"\n")
        print(json.dumps(row),flush=True)
        if scores["cer"]<best_finetuned:
            best_finetuned=scores["cer"];best_finetuned_source=f"epoch-{epoch+1}"
            atomic_save(model.state_dict(),a.output/"best-finetuned.pth")
            stale=0
        else:
            stale+=1
        if scores["cer"]<best:
            best=scores["cer"];best_source=f"epoch-{epoch+1}";atomic_save(model.state_dict(),a.output/"model.pth")
        checkpoint(epoch+1,0);next_batch=0
        if stale>=a.patience:break
    metadata.update(training_complete=True,best_validation_cer=best,best_source=best_source,
                    best_finetuned_validation_cer=best_finetuned,best_finetuned_source=best_finetuned_source,
                    fine_tuned_improved=best_source.startswith("epoch-"),updates=step)
    (a.output/"run.json").write_text(json.dumps(metadata,default=str,indent=2),encoding="utf-8")
    if best_source=="pretrained-initialization":
        print("Warning: no epoch beat the pretrained initialization on validation; packaged best remains the initialization.")
    print("Done. Run training.evaluate on the held-out test for pretrained AND selected checkpoint.")


if __name__=="__main__":main()
