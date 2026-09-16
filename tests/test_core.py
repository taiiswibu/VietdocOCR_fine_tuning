import io
import json
import fitz
import pytest
from PIL import Image
from vietdoc.core import parse_pages,open_document,chunk_pages,export_bytes,Block,reading_order
from vietdoc.pipeline import process
from training.metrics import summarize
from training.prepare import split_for


def test_page_selection():
    assert parse_pages("3,1-2,2",4)==[0,1,2]
    for value in ("0","3-1","1-5","../x","x"):
        with pytest.raises(ValueError):parse_pages(value,4)


def test_native_pdf_bypasses_model_and_preserves_pages(tmp_path):
    source=tmp_path/"example.pdf"
    with fitz.open() as d:
        for i in range(3):
            p=d.new_page();p.insert_text((50,70),f"Page {i+1} has native text that should not require optical character recognition.")
        d.save(source)
    class Forbidden:
        def run(self,*a,**k):raise AssertionError("OCR should not run")
    result=process(source,tmp_path/"out",Forbidden(),pages="2-3")
    assert [p['number'] for p in result['pages']]==[2,3]
    assert all(p['method']=='pdf-text' and not p['error'] for p in result['pages'])
    assert 'Page 2' in result['pages'][0]['text']


def test_page_failure_is_visible(tmp_path):
    p=tmp_path/'image.png';Image.new('RGB',(100,60),'gray').save(p)
    class Broken:
        def run(self,*a,**k):raise RuntimeError('model unavailable')
    result=process(p,tmp_path/'out',Broken())
    assert result['pages'][0]['error']=='RuntimeError: model unavailable'


def test_chunks_reconstruct_and_have_provenance():
    text='Đây là chữ tiếng Việt. '*160
    pages=[{'number':3,'text':text.strip(),'method':'handwriting','reviewed':True}]
    chunks=list(chunk_pages(pages,'a'*64,'doc.pdf',size=200,overlap=20))
    assert len(chunks)>3 and len({c['id'] for c in chunks})==len(chunks)
    for c in chunks:
        m=c['metadata'];assert c['text']==text.strip()[m['char_start']:m['char_end']]
        assert m['page']==3 and m['reviewed']
    assert chunks[-1]['metadata']['char_end']==len(text.strip())


def test_exports_use_corrections():
    d={'document_id':'f'*64,'filename':'x','pages':[{'number':1,'text':'Đã sửa','method':'pdf-text'}]}
    for fmt in ['txt','md','json','jsonl']:
        assert 'Đã sửa' in export_bytes(d,fmt)[0].decode()


def test_metric_diacritics_and_unicode():
    assert summarize([('á','a\u0301')])['cer']==0
    assert summarize([('á','a')])['cer']==1
    assert summarize([('abc','abcabc')])['cer']==1


def test_split_identical_transcription_grouped():
    assert split_for('  Học  tập ',42,.1)==split_for('Học tập',42,.1)


def test_reading_order_columns():
    blocks=[Block('r',[60,10,90,20],'right','test'),Block('l2',[0,30,40,40],'left2','test'),Block('l',[0,10,40,20],'left','test')]
    assert [b.id for b in reading_order(blocks,2,100)]==['l','l2','r']


def test_blank_page_does_not_require_model(tmp_path):
    p=tmp_path/'blank.png';Image.new('RGB',(100,60),'white').save(p)
    class Forbidden:
        def run(self,*a,**k):raise AssertionError('Blank page must not invoke model')
    result=process(p,tmp_path/'out',Forbidden())
    assert result['pages'][0]['method']=='blank'
    assert not result['pages'][0]['error']


def test_high_chunk_overlap_still_advances():
    text='word '*100
    chunks=list(chunk_pages([{'number':1,'text':text,'method':'test'}],'x','x',size=200,overlap=199))
    starts=[c['metadata']['char_start'] for c in chunks]
    assert all(b>a for a,b in zip(starts,starts[1:]))
    assert chunks[-1]['metadata']['char_end']==len(text.strip())


def test_word_regions_merge_into_lines_without_crossing_columns():
    from vietdoc.engines import merge_line_boxes
    boxes=[[5,10,40,30],[45,12,90,32],[5,50,45,70]]
    assert merge_line_boxes(boxes)==[[5,10,90,32],[5,50,45,70]]
    assert len(merge_line_boxes([[30,10,48,30],[52,10,70,30]],columns=2,page_width=100))==2
