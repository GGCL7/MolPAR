#!/usr/bin/env python3
"""Recompute frozen seed0 benchmark using original batch protocol (256/128)."""
import argparse, json
from pathlib import Path
from molpar_runtime import load, probabilities

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=4)
    a=p.parse_args()
    if a.output_dir.exists():p.error('Choose a new output directory')
    import numpy as np
    import torch
    from torch.utils.data import DataLoader,Subset
    model,df,ds,split,config,meta=load(a.device,a.threads)
    from dataset import collate_fn_both
    from utils import multilabel_metrics
    def encode(indices,size):
        embs=[];labels=[]
        loader=DataLoader(Subset(ds,indices),batch_size=size,shuffle=False,collate_fn=collate_fn_both)
        with torch.no_grad():
            for graph,fp,l1,l2 in loader:
                embs.append(model.encoder(graph.to(a.device),fp.to(a.device)).cpu());labels.append(l2)
        return torch.cat(embs),torch.cat(labels)
    print('Encoding training memory (batch 256)...',flush=True)
    memory,train_y=encode(split[0],256)
    a.output_dir.mkdir(parents=True)
    report={'protocol':'canonical split order, retrieval batch=256, evaluation batch=128, threshold=0.5','model':'single random_seed0 Stage2','metadata':meta}
    for name,idx in [('validation',split[1]),('test',split[2])]:
        emb,y=encode(idx,128);prob=probabilities(model,emb,memory,train_y,int(config.get('retrieval_k',32)))
        report[name]=multilabel_metrics(y,prob,.5)
        np.savez_compressed(a.output_dir/f'{name}_predictions.npz',indices=np.asarray(idx),labels=y.numpy(),probabilities=prob.numpy())
    (a.output_dir/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['validation','test']},indent=2))
if __name__=='__main__':main()
