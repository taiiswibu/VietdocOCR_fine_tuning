import unicodedata


def clean(text):
    return " ".join(unicodedata.normalize("NFC",text).split())


def distance(a,b):
    if len(a)<len(b):a,b=b,a
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1):current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
        previous=current
    return previous[-1]


def summarize(pairs):
    ce=we=nc=nw=exact=count=0
    for truth,pred in pairs:
        truth=clean(truth);pred=clean(pred)
        ce+=distance(truth,pred);nc+=len(truth)
        we+=distance(truth.split(),pred.split());nw+=len(truth.split())
        exact+=truth==pred;count+=1
    return {"samples":count,"cer":ce/max(1,nc),"wer_whitespace":we/max(1,nw),"exact_match":exact/max(1,count),
            "character_edits":ce,"reference_characters":nc,"normalization":"NFC + whitespace collapse; case and diacritics preserved"}
