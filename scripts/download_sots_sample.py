import urllib.request, io, zipfile, struct, zlib, pathlib, json, hashlib, csv
import argparse, random, time
from concurrent.futures import ThreadPoolExecutor
parser=argparse.ArgumentParser(description='Download a reproducible scene-disjoint SOTS sample.')
parser.add_argument('--test-count', type=int, default=8)
args=parser.parse_args()
if args.test_count < 8: parser.error('--test-count must be at least 8')
root = pathlib.Path('data/evaluation/sots_sample')
root.mkdir(parents=True, exist_ok=True)
url='https://huggingface.co/datasets/desonglll/reside-SOTS/resolve/04dd3b3eb281637d2e3a0232368c43fce846e505/SOTS.zip'
size=434690507
def fetch(start,end):
    for attempt in range(4):
        try:
            req=urllib.request.Request(url, headers={'Range':f'bytes={start}-{end}'})
            with urllib.request.urlopen(req, timeout=90) as r:
                if r.status != 206: raise RuntimeError('Server does not support partial download')
                data=r.read()
                if len(data)!=end-start+1: raise RuntimeError('Incomplete range response')
                return data
        except Exception:
            if attempt==3: raise
            time.sleep(2**attempt)
tail=fetch(size-100000,size-1)
eocd=tail.rfind(b'PK\x05\x06')
fields=struct.unpack_from('<4s4H2LH',tail,eocd)
central_size,central_offset=fields[5:7]
central=fetch(central_offset,central_offset+central_size-1)
# Use zipfile to parse central-directory metadata without downloading image payloads.
fake=io.BytesIO(central+tail[eocd:])
archive=zipfile.ZipFile(fake)
entries=archive.infolist()
print([(x.filename,x.file_size) for x in entries[:12]],flush=True)
(root/'entries.json').write_text(json.dumps([x.filename for x in entries],indent=2))
def download(entry):
    dest=root/entry.filename
    if dest.is_file():
        raw=dest.read_bytes()
        if len(raw)==entry.file_size and zlib.crc32(raw)==entry.CRC:
            return {'path':entry.filename,'sha256':hashlib.sha256(raw).hexdigest()}
    # zipfile adjusts offsets to our in-memory central directory; restore original offsets.
    offset=entry.header_offset+central_offset
    header=fetch(offset,offset+29)
    name_len,extra_len=struct.unpack_from('<HH',header,26)
    start=offset+30+name_len+extra_len
    data=fetch(start,start+entry.compress_size-1)
    raw=zlib.decompress(data,-15) if entry.compress_type==8 else data
    if zlib.crc32(raw)!=entry.CRC: raise RuntimeError('CRC mismatch')
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_bytes(raw)
    return {'path':entry.filename,'sha256':hashlib.sha256(raw).hexdigest()}
by_name={e.filename:e for e in entries}
hazy=[e for e in entries if '/outdoor/hazy/' in '/'+e.filename and e.filename.endswith(('.png','.jpg'))]
selected=[]; seen=set()
for e in sorted(hazy,key=lambda e:e.filename):
    scene=pathlib.Path(e.filename).stem.split('_')[0]
    if scene in seen: continue
    gt=[g for g in entries if '/outdoor/gt/' in '/'+g.filename and pathlib.Path(g.filename).stem==scene]
    if gt:
        selected.append((scene,e,gt[0])); seen.add(scene)
if not selected: raise RuntimeError('No outdoor pairs found; inspect entries.json')
if args.test_count+4>len(selected): raise ValueError('Requested more scenes than available')
selected=selected[:12]+random.Random(42).sample(selected[12:],args.test_count-8)
suffix='' if args.test_count==8 else f'_{args.test_count}'
provenance=[]
def download_pair(pair):
    scene,h,g=pair
    hashes=[download(h),download(g)]
    print('Downloaded/verified',scene,flush=True)
    return hashes
with ThreadPoolExecutor(max_workers=4) as pool:
    for hashes in pool.map(download_pair,selected): provenance.extend(hashes)
with (root/f'manifest{suffix}.csv').open('w',newline='') as f:
    writer=csv.writer(f); writer.writerow(['image_id','image_path','reference_path'])
    for scene,h,g in selected:
        writer.writerow([scene,h.filename,g.filename])
(root/f'provenance{suffix}.json').write_text(json.dumps({'source':url,'selection':f'Original first 12 distinct scenes plus {args.test_count-8} scenes sampled without replacement from the remaining sorted scenes using Python random seed 42; first haze variant per scene; first four original scenes excluded from testing','files':provenance},indent=2))
