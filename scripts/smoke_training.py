"""Real forward/backward + interrupted resume check on synthetic lines, NOT an accuracy benchmark."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

from PIL import Image, ImageDraw
import yaml


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--initial',type=Path,default=Path('models/base/model.pth'))
    p.add_argument('--report',type=Path,default=Path('workspace/training-smoke.json'))
    a=p.parse_args()
    with tempfile.TemporaryDirectory(prefix='vietdoc-training-') as temp:
        root=Path(temp);data=root/'data';data.mkdir();run=root/'run'
        cfg=yaml.safe_load(Path('configs/vietocr.yml').read_text(encoding='utf-8'))
        cfg['decode_max_tokens']=8
        (data/'config.yml').write_text(yaml.safe_dump(cfg,allow_unicode=True),encoding='utf-8')
        rows=[]
        for i,label in enumerate(['test','text','read']):
            image=Image.new('RGB',(96,32),'white');ImageDraw.Draw(image).text((5,8),label,fill='black')
            image.save(data/f'{i}.png');rows.append({'id':i,'image':f'{i}.png','text':label})
        for split,items in [('train',rows[:2]),('val',rows[2:]),('test',rows[2:])]:
            (data/f'{split}.jsonl').write_text('\n'.join(json.dumps(r) for r in items),encoding='utf-8')
        command=[sys.executable,'-m','training.train','--data',str(data),'--output',str(run),'--initial',str(a.initial.resolve()),'--device','cpu','--epochs','1','--batch-size','1','--accum','1']
        subprocess.run(command+['--max-updates','1'],check=True)
        assert not json.loads((run/'run.json').read_text())['training_complete']
        import torch
        before=torch.load(run/'last.pt',map_location='cpu',weights_only=True)
        assert before['step']==1 and before['next_batch']==1
        del before
        subprocess.run(command+['--resume',str(run/'last.pt')],check=True)
        metadata=json.loads((run/'run.json').read_text())
        assert metadata['training_complete'] and metadata['updates']==2
        from vietdoc.recognizer import LineRecognizer
        prediction,score,truncated=LineRecognizer(run).predict(Image.open(data/'2.png'))
        bundle=root/'model.zip'
        subprocess.run([sys.executable,'scripts/package_model.py',str(run),'--output',str(bundle)],check=True)
        with zipfile.ZipFile(bundle) as z:assert 'handwriting/model.pth' in z.namelist()
        report={'purpose':'functional smoke only; synthetic data; val/test shared intentionally; no accuracy claim',
                'interrupted_at_update':1,'resumed_final_updates':2,'model_reload':True,'bundle_verified':True,
                'prediction':prediction,'score':score,'truncated':truncated,'torch':str(torch.__version__)}
        a.report.parent.mkdir(parents=True,exist_ok=True)
        a.report.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
