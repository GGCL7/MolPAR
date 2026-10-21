#!/usr/bin/env python3
"""Predict L2 pathway rankings with the released seed0 model and fixed background."""
import argparse, json
from pathlib import Path
from molpar_runtime import ROOT, DATA, load, probabilities, digest, load_label_spaces

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True,help='CSV with smiles; optional unique molecule_id')
    p.add_argument('--output-dir',type=Path,required=True,help='New directory, never overwrite existing output')
    p.add_argument('--cache-dir',type=Path,default=ROOT/'outputs/cache')
    p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=4)
    p.add_argument('--chunk-size',type=int,default=64);p.add_argument('--top-k',type=int,default=10)
    p.add_argument('--threshold',type=float,default=None,help='Optional user-defined threshold; not ensemble calibration')
    p.add_argument('--smoke-test',action='store_true',help='Strict load + encoder equivalence, no retrieval predictions')
    a=p.parse_args()
    if not 1<=a.top_k<=146 or a.chunk_size<1 or a.threads<1:p.error('Invalid positive count')
    if a.threshold is not None and not 0<=a.threshold<=1:p.error('Threshold must be in [0,1]')
    if a.output_dir.exists():p.error('Output directory already exists; use a new directory')
    import numpy as np
    import pandas as pd
    import torch
    from rdkit import Chem
    model,df,ds,split,config,meta=load(a.device,a.threads)
    from dataset import HierarchicalMoleculeDataset
    from fixed_background import FixedBackgroundEncoder
    q=pd.read_csv(a.input,dtype=str,keep_default_na=False)
    if q.empty or 'smiles' not in q:raise ValueError('Input must have nonempty smiles rows')
    if 'molecule_id' not in q:q['molecule_id']=[f'query_{i+1}' for i in range(len(q))]
    if q.molecule_id.duplicated().any() or q.molecule_id.eq('').any():raise ValueError('molecule_id must be nonempty and unique')
    keys=[]
    for i,s in enumerate(q.smiles):
        mol=Chem.MolFromSmiles(s)
        if mol is None or mol.GetNumAtoms()<2 or len(Chem.GetMolFrags(mol))!=1:raise ValueError(f'Row {i+1}: invalid, single-atom or disconnected SMILES; supply a curated connected structure')
        keys.append(Chem.MolToInchiKey(mol))
    q['labels_L1']='';q['labels_L2']=''
    qds=HierarchicalMoleculeDataset(q,mode='both',cache=True)
    train_idx=split[0]
    bg=sorted(train_idx,key=lambda i:(str(df.iloc[i].kegg_id),i))[:255]
    encoder=FixedBackgroundEncoder(model,ds,bg,torch.device(a.device))
    embeddings=encoder.encode_indices(qds,list(range(len(q))),a.chunk_size)
    checks=sorted(set([0,len(q)-1]))
    errs=[float((embeddings[i]-encoder.literal_embedding(qds,i)).abs().max()) for i in checks]
    if max(errs)>=1e-4:raise RuntimeError(f'Fixed-background equivalence failed: {errs}')
    a.output_dir.mkdir(parents=True)
    meta.update(protocol='255 sorted training background molecules + one query; single seed0 L2',background_rows=bg,equivalence_max_abs_error=max(errs),input_sha256=digest(a.input),query_count=len(q),three_seed_consensus=False,threshold=a.threshold)
    if a.smoke_test:
        meta['status']='encoder smoke test only; no retrieval or pathway prediction'
        (a.output_dir/'smoke_test.json').write_text(json.dumps(meta,indent=2)+'\n')
        print(json.dumps({'status':meta['status'],'queries':len(q),'equivalence_max_abs_error':max(errs)},indent=2));return
    import hashlib
    memory_meta={k:v for k,v in meta.items() if k not in ['input_sha256','query_count','equivalence_max_abs_error','threshold','three_seed_consensus']}
    tag=hashlib.sha256(json.dumps(memory_meta,sort_keys=True).encode()).hexdigest()
    a.cache_dir.mkdir(parents=True,exist_ok=True)
    cache=a.cache_dir/f'seed0_{tag}.npz'
    if cache.exists():
        with np.load(cache,allow_pickle=False) as saved:
            if str(saved['fingerprint'].item())!=tag or not np.array_equal(saved['rows'],train_idx):raise ValueError('Bad memory cache')
            memory=torch.from_numpy(saved['embeddings'].copy())
    else:
        print('Building retrieval memory from 4,381 training structures...',flush=True)
        memory=encoder.encode_indices(ds,train_idx,a.chunk_size)
        np.savez_compressed(cache,embeddings=memory.numpy(),rows=np.asarray(train_idx),fingerprint=tag)
        cache.with_suffix('.json').write_text(json.dumps(memory_meta,indent=2)+'\n')
    if memory.shape!=(4381,256) or not torch.isfinite(memory).all():raise ValueError('Invalid memory')
    labels=torch.stack([ds[i][3] for i in train_idx])
    scores=probabilities(model,embeddings,memory,labels,int(config.get('retrieval_k',32))).numpy()
    if scores.shape!=(len(q),146) or not np.isfinite(scores).all():raise ValueError('Invalid predictions')
    info=load_label_spaces()[1]; maps=info['index_to_map']; names=info.get('map_to_name',{})
    train_keys=set(df.iloc[train_idx].inchikey.astype(str));all_keys=set(df.inchikey.astype(str))
    rows=[]
    for i,row in q.iterrows():
        for rank,j in enumerate(np.argsort(-scores[i],kind='stable'),1):
            pid=maps[str(j)]
            record={'molecule_id':row.molecule_id,'smiles':row.smiles,'inchikey_from_smiles':keys[i],'exact_training_match':keys[i] in train_keys,'exact_reference_match':keys[i] in all_keys,'pathway_index':int(j),'pathway_id':pid,'pathway_name':names.get(pid,''),'rank':rank,'score':float(scores[i,j]),'model':'MolPAR_random_seed0_stage2'}
            if a.threshold is not None:record['above_user_threshold']=bool(scores[i,j]>=a.threshold)
            rows.append(record)
    table=pd.DataFrame(rows);table.to_csv(a.output_dir/'all_pathway_scores.csv',index=False)
    table[table['rank']<=a.top_k].to_csv(a.output_dir/'top_pathways.csv',index=False)
    np.savez_compressed(a.output_dir/'probabilities.npz',scores=scores,molecule_ids=q.molecule_id.to_numpy(dtype=str))
    meta.update(status='completed',cache_fingerprint=tag,retrieval_training_rows=len(train_idx),prediction_sha256=digest(a.output_dir/'probabilities.npz'))
    (a.output_dir/'prediction_manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps({'status':'completed','queries':len(q),'labels':146,'output_dir':str(a.output_dir)},indent=2))
if __name__=='__main__':main()
