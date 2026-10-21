"""Portable seed0 runtime. Preserves the archived encoder and frozen parameters."""
from pathlib import Path
import ast, hashlib, json, platform, sys
ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'code/model'
DATA = ROOT / 'data/final/training'
CKPT = ROOT / 'weights/molpar_seed0.pth'
EXPECTED = '16812e3ba9fbd2c2e3981c68441f76318f3cf6f7723d1bebb33d64f7a87a8f4d'
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(ROOT / 'code'))

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def load(device='cpu',threads=4):
    import numpy as np
    import pandas as pd
    import torch
    import torch_geometric
    from rdkit import RDLogger, rdBase
    from dataset import HierarchicalMoleculeDataset
    from ra_glgn_model import RAGLGN
    RDLogger.DisableLog('rdApp.warning')
    torch.set_num_threads(threads)
    torch.manual_seed(42)
    if digest(CKPT)!=EXPECTED: raise ValueError('Canonical random_seed0 checkpoint SHA-256 mismatch')
    ckpt=torch.load(CKPT,map_location='cpu',weights_only=True)
    args=ckpt.get('args',{})
    if not args.get('no_label_gcn') or not args.get('use_retrieval'): raise ValueError('Wrong model configuration')
    df=pd.read_csv(DATA/'molecules.csv')
    ds=HierarchicalMoleculeDataset(df,mode='both',cache=True)
    dims=(ds[0][0].x.size(1),ds[0][0].edge_attr.size(1),ds[0][1].numel())
    if dims!=(51,14,2513): raise ValueError(f'Wrong feature dimensions: {dims}')
    kwargs={k:args.get(k,v) for k,v in dict(graph_hidden_dim=128,graph_out_dim=256,fp_hidden_dim=256,fusion_hidden_dim=256,final_dim=256,pool_ratio=.5,label_emb_dim=256,retrieval_beta=.3).items()}
    model=RAGLGN(node_in_dim=51,edge_in_dim=14,fp_dim=2513,num_l1=11,num_l2=146,use_label_gcn=False,use_retrieval=True,**kwargs).to(device)
    model.load_state_dict(ckpt['model'],strict=True); model.eval()
    split=[df.loc[df.split_seed0.eq(name)].sort_values('split_order_seed0').index.tolist() for name in ['train','validation','test']]
    if [len(x) for x in split]!=[4381,547,549] or set().union(*map(set,split))!=set(range(len(df))) or sum(map(len,split))!=len(df): raise ValueError('Invalid split')
    metadata={'checkpoint_sha256':EXPECTED,'data_sha256':digest(DATA/'molecules.csv'),'split_sha256':hashlib.sha256(json.dumps(split).encode()).hexdigest(),'code_sha256':{p.name:digest(p) for p in sorted(CODE.glob('*.py'))},'fixed_background_sha256':digest(ROOT/'code/fixed_background.py'),'runtime_sha256':digest(Path(__file__)),'versions':{'python':platform.python_version(),'torch':torch.__version__,'torch_geometric':torch_geometric.__version__,'rdkit':rdBase.rdkitVersion,'numpy':np.__version__,'pandas':pd.__version__},'device':str(device),'threads':threads}
    return model,df,ds,split,args,metadata

def probabilities(model,embedding,train_embedding,train_y,k=32,chunk=256):
    import torch
    import torch.nn.functional as F
    dev=next(model.parameters()).device
    outputs=[]
    with torch.no_grad():
        train=F.normalize(train_embedding.to(dev),dim=-1); labels=train_y.to(dev)
        _,anchors=model.get_label_embeddings()
        for block in embedding.split(chunk):
            q=block.to(dev); values,indices=(F.normalize(q,dim=-1)@train.T).topk(k,dim=-1)
            votes=(torch.softmax(values,dim=-1).unsqueeze(-1)*labels[indices]).sum(1)
            outputs.append(torch.sigmoid(model.knn_voter(q@anchors.T+model.bias_l2,votes)).cpu())
    return torch.cat(outputs)


def load_label_spaces():
    import csv
    with (DATA/'pathways.csv').open(newline='') as f: rows=list(csv.DictReader(f))
    first=[r for r in rows if r['level']=='L1']; second=[r for r in rows if r['level']=='L2']
    l1={'n_classes':len(first),'index_to_name':{r['label_index']:r['name'] for r in first}}
    l2={'n_classes':len(second),'min_count':int(second[0]['minimum_reference_count']),'index_to_map':{r['label_index']:r['pathway_id'] for r in second},'map_to_name':{r['pathway_id']:r['name'] for r in second},'map_to_l1_index':{r['pathway_id']:int(r['parent_l1_index']) for r in second}}
    return l1,l2
