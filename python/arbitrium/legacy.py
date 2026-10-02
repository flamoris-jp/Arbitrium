"""Integrity-checked legacy research reader. Never rewrites historical evidence."""
from pathlib import Path
from collections import Counter
from maidionis_education.contracts import canonical,digest,loads
from maidionis_education.storage import read
SPLITS=('train','dev','calibration_fit','calibration_select','test')
ROOT=Path(__file__).resolve().parents[2]
ARCHIVE=ROOT/'research/legacy'
def jsonl(raw):
    if raw and not raw.endswith(b'\n'): raise ValueError('legacy JSONL framing')
    return [loads(line,128*1024) for line in raw.splitlines()]
def verify_archive():
    raw=read(ARCHIVE/'inventory.json',4*2**20)
    lock=loads((ROOT/'dependency.lock.json').read_bytes())
    if digest(raw)!=lock['legacy_inventory_digest']:raise ValueError('trusted archive inventory digest')
    index=loads(raw,4*2**20)
    paths=set()
    for e in index['files']:
        p=e['path']
        if p in paths or p.startswith('/') or '..' in p.split('/'):raise ValueError('legacy path')
        paths.add(p);b=read(ARCHIVE/p,4*2**20)
        if len(b)!=e['bytes'] or digest(b)!=e['sha256']:raise ValueError('legacy archive integrity: '+p)
    actual={p.relative_to(ARCHIVE).as_posix() for p in ARCHIVE.rglob('*') if p.is_file()}
    if actual!=paths|{'inventory.json','README.md'}:raise ValueError('archive extra/missing member')
    return index

def split_for(family):
    bucket=int(digest(('arbitrium-split-v1\n42\n'+family).encode())[:16],16)%10000
    return SPLITS[sum(bucket>=b for b in (6000,7500,8500,9000))]

def dataset(name):
    if type(name) is not str or name not in loads(read(ROOT/'dependency.lock.json',65536))['datasets']:raise ValueError('unregistered legacy dataset')
    verify_archive()
    base=ARCHIVE/'examples/education'/name
    raw=read(base/'manifest.json',4*2**20);m=loads(raw,4*2**20)
    if m['dataset_id']!=name or m['split_seed']!=42 or m['split_algorithm']!='arbitrium-split-v1':raise ValueError('legacy identity/profile')
    files={}
    for p,e in m['files'].items():
        if '/' in p or p in ('.','..'):raise ValueError('legacy inventory path')
        b=read(base/p,4*2**20)
        if len(b)!=e['bytes'] or digest(b)!=e['sha256']:raise ValueError('legacy file digest')
        files[p]=b
    rows=[];proofs=jsonl(files['provenance-index.jsonl']);byproof={p['provenance_id']:p for p in proofs}
    if len(byproof)!=len(proofs):raise ValueError('duplicate legacy proof')
    ids=set();families={}
    for s in SPLITS:
        selected=jsonl(files[s+'.jsonl']);counts=m['counts'][s]
        if len(selected)!=counts['records'] or dict(Counter(str(r['target']['label']) for r in selected))!=counts['labels']:raise ValueError('legacy split counts')
        for r in selected:
            if r['sample_id'] in ids or r['split']!=s or split_for(r['family_id'])!=s:raise ValueError('legacy family split')
            ids.add(r['sample_id']);families[r['family_id']]=s
            p=byproof.get(r['provenance_id'])
            if not p or p['sample_id']!=r['sample_id'] or p['expected_outcome']!=r['target']:raise ValueError('legacy source/target evidence')
        rows.extend(selected)
    if len(rows)!=len(proofs) or jsonl(files['families.jsonl'])!=[dict(family_id=k,split=v) for k,v in sorted(families.items())]:raise ValueError('legacy family/provenance coverage')
    return m,digest(raw),rows,byproof
