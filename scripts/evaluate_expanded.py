"""Evaluate a larger test sample with frozen settings and checkpoint."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import argparse, csv, hashlib, json
import numpy as np
import torch
from dehazing.classical.dcp import dehaze_dcp
from dehazing.inference import dehaze_with_aodnet
from dehazing.metrics import evaluate_pair
from dehazing.utils.image_io import load_image

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--test-count',type=int,default=100)
    args=parser.parse_args()
    data=ROOT/'data/evaluation/sots_sample'
    out=ROOT/f'outputs/sots_{args.test_count}'
    out.mkdir(parents=True,exist_ok=True)
    prior=json.loads((ROOT/'outputs/sots_sample/results.json').read_text())
    config=prior['selected']['config']
    checkpoint=ROOT/'outputs/sots_sample/aodnet_compatible.pth'
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest!=prior['converted_sha256']: raise ValueError('Checkpoint changed')
    with (data/f'manifest_{args.test_count}.csv').open() as f: manifest=list(csv.DictReader(f))
    validation=set(prior['validation_scenes'])
    test=[r for r in manifest if r['image_id'] not in validation]
    ids=[r['image_id'] for r in test]
    assert len(ids)==len(set(ids))==args.test_count
    assert not set(ids)&validation
    assert set(prior['test_scenes']).issubset(ids)
    torch.set_num_threads(4)
    methods=['hazy_input','dcp_default','dcp_tuned','aodnet']
    records=[]
    with (out/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['scene','group','method','psnr','ssim','ssim_percent'])
        writer.writeheader()
        for index,row in enumerate(test,1):
            hazy=load_image(data/row['image_path']); clear=load_image(data/row['reference_path'])
            if hazy.shape!=clear.shape: raise ValueError(f'Dimensions differ: {row}')
            for method in methods:
                if method=='hazy_input': pred=hazy
                elif method=='dcp_default': pred=dehaze_dcp(hazy)
                elif method=='dcp_tuned': pred=dehaze_dcp(hazy,config)
                else: pred=dehaze_with_aodnet(hazy,checkpoint,'cpu').image
                scores=evaluate_pair(clear,pred)
                if not all(np.isfinite(v) for v in scores.values()): raise ValueError('Nonfinite score')
                result={'scene':row['image_id'],'group':'original' if row['image_id'] in prior['test_scenes'] else 'new','method':method,**scores,'ssim_percent':100*scores['ssim']}
                records.append(result);writer.writerow(result)
            f.flush()
            print(f'Evaluated {index}/{len(test)}: {row["image_id"]}',flush=True)
    summary=[]
    with (ROOT/'outputs/sots_sample/comparison.csv').open() as f:
        old={(r['scene'],r['method']):r for r in csv.DictReader(f) if r['split']=='test'}
    for r in records:
        if r['group']=='original':
            for metric in ['psnr','ssim']:
                if not np.isclose(r[metric],float(old[r['scene'],r['method']][metric]),rtol=1e-6,atol=1e-7):
                    raise ValueError(f'Original-scene result changed: {r["scene"]} {r["method"]} {metric}')
    for group in ['all','new','original']:
        for method in methods:
            rows=[r for r in records if r['method']==method and (group=='all' or r['group']==group)]
            ssim=np.array([r['ssim'] for r in rows]); psnr=np.array([r['psnr'] for r in rows])
            rng=np.random.default_rng(42)
            boot=ssim[rng.integers(0,len(ssim),size=(10000,len(ssim)))].mean(axis=1)*100
            summary.append({'group':group,'method':method,'n':len(rows),'psnr':float(psnr.mean()),'ssim':float(ssim.mean()),'ssim_percent':float(ssim.mean()*100),'ssim_percent_bootstrap95':np.percentile(boot,[2.5,97.5]).tolist()})
    report={'test_count':len(test),'test_scenes':ids,'excluded_validation_scenes':sorted(validation),'frozen_dcp_config':config,'checkpoint_sha256':digest,'metric_note':'SSIM x 100 is structural similarity expressed as a percentage, not classification accuracy. Bootstrap intervals describe sample variability, not domain generalization.','summary':summary}
    (out/'results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
