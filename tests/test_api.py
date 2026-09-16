import io
import time
import zipfile
import json
import fitz
from PIL import Image
from fastapi.testclient import TestClient
from vietdoc import server


def pdf_bytes():
    with fitz.open() as d:
        p=d.new_page();p.insert_text((50,70),'This is a native PDF document with enough text to avoid unnecessary OCR processing.')
        return d.tobytes()


def test_upload_edit_export_training_and_security(tmp_path,monkeypatch):
    monkeypatch.setattr(server,'WORK',tmp_path)
    with TestClient(server.app) as client:
        r=client.post('/api/jobs',files={'file':('test.pdf',pdf_bytes(),'application/pdf')})
        assert r.status_code==200
        id=r.json()['id']
        for _ in range(100):
            value=client.get(f'/api/jobs/{id}').json()
            if value['status'] not in ('queued','running'):break
            time.sleep(.02)
        assert value['status']=='done'
        page=value['document']['pages'][0]
        response=client.patch(f'/api/jobs/{id}/pages/1',json={'block_id':page['blocks'][0]['id'],'text':'Nhãn đã kiểm tra','reviewed':True})
        assert response.status_code==200
        export=client.get(f'/api/jobs/{id}/export/training')
        assert export.status_code==200
        with zipfile.ZipFile(io.BytesIO(export.content)) as z:
            labels=[json.loads(line) for line in z.read('labels.jsonl').decode().splitlines()]
            assert labels[0]['human_reviewed'] is True
            assert Image.open(io.BytesIO(z.read(labels[0]['image']))).width>0
        assert client.patch(f'/api/jobs/{id}/pages/1',json={'text':'Sửa toàn văn'}).status_code==200
        assert 'Sửa toàn văn' in client.get(f'/api/jobs/{id}/export/jsonl').text
        assert client.get(f'/api/jobs/{id}/export/training').status_code==400
        assert client.post('/api/jobs',files={'file':('bad.exe',b'x')}).status_code==400
        assert client.post('/api/jobs',headers={'Origin':'https://evil.invalid'},files={'file':('test.pdf',pdf_bytes())}).status_code==403
        assert client.get('/api/jobs/not-a-valid-id').status_code==404
