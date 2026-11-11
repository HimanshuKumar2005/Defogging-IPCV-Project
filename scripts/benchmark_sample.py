"""Reproduce the small SOTS evaluation; see EVALUATION_REPORT.md for scope."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import csv
import hashlib
import itertools
import json
import time
import numpy as np
import torch
from dehazing.classical.dcp import dehaze_dcp, DEFAULT_CONFIG
from dehazing.metrics import evaluate_pair
from dehazing.models.pretrained_loader import load_pretrained_model
from dehazing.models.aodnet import AODNet
from dehazing.inference import dehaze_with_aodnet
from dehazing.utils.image_io import load_image, save_image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/evaluation/sots_sample'
OUT = ROOT / 'outputs/sots_sample'
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)

def main():
    source = ROOT / 'outputs/evaluation_setup/dehazer.pth'
    weights = torch.load(source, map_location='cpu', weights_only=True)
    expected = {f'e_conv{i}.{p}' for i in range(1,6) for p in ('weight','bias')}
    if set(weights) != expected:
        raise ValueError(f'Unexpected checkpoint keys: {list(weights)}')
    mapped = {k.replace('e_conv', 'conv'): v for k,v in weights.items()}
    model = AODNet()
    model.load_state_dict(mapped, strict=True)
    model.eval()
    # Independent source-equation check, without executing downloaded Python.
    torch.manual_seed(42)
    x = torch.rand(1,3,32,32)
    import torch.nn.functional as F
    def conv(x,i):
        return F.relu(F.conv2d(x,weights[f'e_conv{i}.weight'],weights[f'e_conv{i}.bias'],padding=(0,1,2,3,1)[i-1]))
    with torch.inference_mode():
        a=conv(x,1); b=conv(a,2); c=conv(torch.cat([a,b],1),3)
        d=conv(torch.cat([b,c],1),4); k=conv(torch.cat([a,b,c,d],1),5)
        torch.testing.assert_close(model(x), F.relu(k*x-k+1),rtol=0,atol=0)
    checkpoint=OUT/'aodnet_compatible.pth'
    torch.save(mapped,checkpoint)
    model=load_pretrained_model(checkpoint,'cpu')
    pairs=[]
    with (DATA/'manifest.csv').open() as f:
        for row in csv.DictReader(f):
            hazy=load_image(DATA/row['image_path']); clear=load_image(DATA/row['reference_path'])
            if hazy.shape!=clear.shape: raise ValueError('Pair dimensions differ')
            pairs.append((row['image_id'],hazy,clear))
    # Fixed scene-disjoint split before any score is computed.
    validation=pairs[:4]; test=pairs[4:]
    if len(pairs)!=12: raise ValueError('Expected 12 distinct scenes')
    trials=[]
    for patch,omega,tmin in itertools.product([7,15,31],[0.75,0.90,0.98],[0.08,0.16]):
        config={**DEFAULT_CONFIG,'patch_size':patch,'omega':omega,'t_min':tmin}
        scores=[evaluate_pair(g,dehaze_dcp(h,config)) for _,h,g in validation]
        trials.append({'config':config,'psnr':float(np.mean([s['psnr'] for s in scores])),'ssim':float(np.mean([s['ssim'] for s in scores]))})
    best=max(trials,key=lambda t:t['psnr'])
    print('Validation-selected config:',best,flush=True)
    results=[]
    for split,subset in [('validation',validation),('test',test)]:
        for scene,hazy,clear in subset:
            for method in ['hazy_input','dcp_default','dcp_tuned','aodnet']:
                start=time.perf_counter()
                if method=='hazy_input': pred=hazy
                elif method.startswith('dcp'): pred=dehaze_dcp(hazy,best['config'] if method=='dcp_tuned' else None)
                else:
                    pred=dehaze_with_aodnet(hazy,checkpoint,'cpu').image
                elapsed=time.perf_counter()-start
                result={'scene':scene,'split':split,'method':method,**evaluate_pair(clear,pred),'seconds':elapsed}
                results.append(result)
                save_image(pred,OUT/'images'/f'{scene}_{method}.png')
            print('Evaluated',scene,split,flush=True)
    with (OUT/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    summaries=[]
    for split in ['validation','test']:
        for method in ['hazy_input','dcp_default','dcp_tuned','aodnet']:
            rows=[r for r in results if r['split']==split and r['method']==method]
            summaries.append({'split':split,'method':method,'n':len(rows),**{metric:float(np.mean([r[metric] for r in rows])) for metric in ['psnr','ssim','seconds']}})
    report={'checkpoint_source':'https://github.com/MayankSingal/PyTorch-Image-Dehazing','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'converted_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'mapping':'e_conv1..5 to conv1..5; strict load and bit-exact forward-equation check','validation_scenes':[p[0] for p in validation],'test_scenes':[p[0] for p in test],'selected':best,'trials':trials,'summary':summaries}
    (OUT/'results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(summaries,indent=2))

if __name__=='__main__': main()
