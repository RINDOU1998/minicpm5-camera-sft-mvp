"""Build/verify a source + model offline bundle. Standard library only; no network."""
import argparse
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from camera_sft.common import ROOT,read_json,sha,write_json
from camera_sft.cloud import CFG,check_model
from camera_sft.data import verify_release,review_report


def safe_relative(name):
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name or not name or str(p)!=name:
        raise ValueError('Unsafe bundle path: '+name)
    return p


def verify_tree(root):
    root=Path(root).resolve();m=read_json(root/'OFFLINE_MANIFEST.json')
    if m['model_id']!=CFG['model_id'] or m['revision']!=CFG['revision']:raise ValueError('Unexpected model identity')
    if not m['smoke_only']:raise ValueError('This release must remain smoke-only')
    for name,entry in m['files'].items():
        safe_relative(name);p=root/name
        if not p.resolve().is_relative_to(root) or not p.is_file():raise ValueError('Missing/unsafe file: '+name)
        if p.stat().st_size!=entry['bytes'] or sha(p)!=entry['sha256']:raise ValueError('Bundle hash mismatch: '+name)
    return {'ok':True,'files':len(m['files']),'source_commit':m['source_commit'],'smoke_only':True}


def verify_archive(path, expected):
    seen=set()
    with tarfile.open(path,'r:gz') as t:
        for member in t:
            if not member.isfile():raise ValueError('Unexpected archive entry')
            p=safe_relative(member.name)
            if p.parts[0]!='minicpm5-sft':raise ValueError('Unexpected archive root')
            name=str(PurePosixPath(*p.parts[1:]))
            if name not in expected or name in seen:raise ValueError('Unexpected/duplicate member: '+name)
            h=hashlib.sha256();size=0
            with t.extractfile(member) as f:
                for block in iter(lambda:f.read(1024*1024),b''):h.update(block);size+=len(block)
            if size!=expected[name]['bytes'] or h.hexdigest()!=expected[name]['sha256']:raise ValueError('Archive hash mismatch: '+name)
            seen.add(name)
    if seen!=set(expected):raise ValueError('Incomplete archive')


def build(model_dir,metadata,out):
    out=Path(out)
    if out.exists() or Path(str(out)+'.partial').exists():raise ValueError('Output exists; use a new path')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Commit reviewed source changes before packaging')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    model_dir=Path(model_dir);snapshot=check_model(model_dir,weights=True)
    if snapshot.get('derived'):raise ValueError('Package the stock BF16 baseline, not a merged model')
    upstream=read_json(metadata)
    if upstream['sha']!=CFG['revision']:raise ValueError('Upstream revision mismatch')
    catalog={f['rfilename']:f for f in upstream['siblings']}
    expected_assets={'config.json','generation_config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','chat_template.jinja','model.safetensors.index.json'}
    if not expected_assets.issubset(snapshot['files']):raise ValueError('Incomplete model configuration/tokenizer')
    index=read_json(model_dir/'model.safetensors.index.json')
    weights=set(index['weight_map'].values())
    if not weights or not weights.issubset(snapshot['files']):raise ValueError('Missing weight shards from index')
    for name,value in snapshot['files'].items():
        if Path(name).name!=name:raise ValueError('Invalid model filename')
        f=catalog[name];p=model_dir/name
        if p.stat().st_size!=f['size']:raise ValueError('Upstream size mismatch: '+name)
        if 'lfs' in f:
            if value!=f['lfs']['sha256']:raise ValueError('Upstream LFS hash mismatch: '+name)
        else:
            body=p.read_bytes()
            if hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()!=f['blobId']:raise ValueError('Upstream Git blob mismatch: '+name)
    verify_release(ROOT/'data/mvp-reviewed-v1')
    if not review_report(ROOT/'data/mvp-reviewed-v1',ROOT/'data/reviews/author-v1/results/reviews.jsonl',require_independent=False)['ok']:
        raise ValueError('Author review incomplete')
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    files={}
    for name in filter(None,tracked):
        safe_relative(name)
        if name.startswith(('models/','runs/','.venv/','data/reviews/human-')) or Path(name).name.startswith('.env'):
            raise ValueError('Excluded file is tracked: '+name)
        p=ROOT/name
        if p.is_symlink():raise ValueError('Symlink not allowed')
        files[name]=p
    for name in [*snapshot['files'],'snapshot_manifest.json']:files['models/base/'+name]=model_dir/name
    provenance={'model_id':CFG['model_id'],'revision':CFG['revision'],'weights':{n:catalog[n]['lfs'] for n in sorted(weights)},
                'source':'Hugging Face revision API metadata; all model assets verified against upstream Git blob or LFS SHA256',
                'scope':'stock BF16 weights, not an adapter or quantized model'}
    inline={'SOURCE_COMMIT':(commit+'\n').encode(),
            'MODEL_PROVENANCE.json':(json.dumps(provenance,indent=2)+'\n').encode()}
    entries={n:{'bytes':p.stat().st_size,'sha256':sha(p)} for n,p in files.items()}
    entries.update({n:{'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()} for n,b in inline.items()})
    manifest={'format':1,'model_id':CFG['model_id'],'revision':CFG['revision'],'source_commit':commit,
              'smoke_only':True,'human_review':'not performed','files':entries}
    inline['OFFLINE_MANIFEST.json']=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode()
    out.parent.mkdir(parents=True,exist_ok=True);partial=Path(str(out)+'.partial')
    with tarfile.open(partial,'w:gz',compresslevel=1) as t:
        for name,p in files.items():t.add(p,arcname='minicpm5-sft/'+name,recursive=False)
        for name,b in inline.items():
            ti=tarfile.TarInfo('minicpm5-sft/'+name);ti.size=len(b);ti.mode=0o644;t.addfile(ti,io.BytesIO(b))
    expected={**entries,'OFFLINE_MANIFEST.json':{'bytes':len(inline['OFFLINE_MANIFEST.json']),'sha256':hashlib.sha256(inline['OFFLINE_MANIFEST.json']).hexdigest()}}
    verify_archive(partial,expected)
    partial.rename(out)
    checksum=sha(out)
    Path(str(out)+'.sha256').write_text(checksum+'  '+out.name+'\n',encoding='utf-8',newline='\n')
    write_json(Path(str(out)+'.json'),{'archive':out.name,'bytes':out.stat().st_size,'sha256':checksum,
        'files':len(expected),'source_commit':commit,'model_revision':CFG['revision'],'archive_verified':True,'smoke_only':True})
    print(json.dumps({'archive':str(out),'bytes':out.stat().st_size,'sha256':checksum,'files':len(expected)},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='action',required=True)
    b=s.add_parser('build');b.add_argument('--model-dir',required=True);b.add_argument('--metadata',required=True);b.add_argument('--out',required=True)
    v=s.add_parser('verify');v.add_argument('--root',default=str(ROOT))
    a=p.parse_args()
    if a.action=='build':build(a.model_dir,a.metadata,a.out)
    else:print(json.dumps(verify_tree(a.root),indent=2))
