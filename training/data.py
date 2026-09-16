import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance
from vietdoc.recognizer import image_array


class Lines(Dataset):
    def __init__(self,manifest,config,vocab,augment=False):
        self.path=Path(manifest);self.rows=[json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines() if l.strip()]
        self.config=config;self.vocab=vocab;self.augment=augment
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i]
        with Image.open(self.path.parent/row["image"]) as im:image=im.convert("RGB")
        if self.augment:
            image=ImageEnhance.Contrast(image).enhance(random.uniform(.85,1.15))
            image=ImageEnhance.Brightness(image).enhance(random.uniform(.9,1.1))
        text=row["text"]
        ids=[1]+[self.vocab.c2i.get(c,3) for c in text]+[2]
        return image_array(image,self.config),ids,text,i


def collate(samples):
    width=max(s[0].shape[-1] for s in samples);height=samples[0][0].shape[1]
    images=np.ones((len(samples),3,height,width),dtype=np.float32)
    maxlen=max(len(s[1])-1 for s in samples)
    inputs=np.zeros((maxlen,len(samples)),dtype=np.int64);targets=np.zeros((len(samples),maxlen),dtype=np.int64)
    widths=[]
    for i,(image,ids,_,_) in enumerate(samples):
        images[i,:,:,:image.shape[-1]]=image;inputs[:len(ids)-1,i]=ids[:-1];targets[i,:len(ids)-1]=ids[1:];widths.append(image.shape[-1])
    return torch.from_numpy(images),torch.from_numpy(inputs),torch.from_numpy(targets),widths
