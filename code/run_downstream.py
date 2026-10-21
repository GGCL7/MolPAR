#!/usr/bin/env python3
"""Run the MolPAR downstream analyses from repository data."""
from pathlib import Path
import argparse
import json
import shutil
import sys
import tempfile
import zipfile
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/final'
sys.path.insert(0,str(ROOT/'code/downstream'))
sys.path.insert(0,str(ROOT/'code'))
TASKS=['candidates','reactions','network','families','edits','drugs','metabolomics','mvpool','chemical-space']

def extract(archive,prefix,destination):
    destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(ROOT/'data/raw'/archive) as z:
        for name in z.namelist():
            if name.startswith(prefix) and not name.endswith('/'):
                relative=Path(name).relative_to(prefix)
                if '..' in relative.parts:raise ValueError('Invalid archive path')
                path=destination/relative;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(z.read(name))

def reference(work):
    from molpar_runtime import load_label_spaces
    import pandas as pd
    d=work/'reference';d.mkdir(parents=True,exist_ok=True)
    pd.read_csv(DATA/'training/molecules.csv').iloc[:,:6].to_csv(d/'kegg2026_hier.csv',index=False)
    for level,obj in zip(['L1','L2'],load_label_spaces()):
        (d/f'label_space_{level}.json').write_text(json.dumps(obj))
    return d

def candidates(out):
    import numpy as np
    import pandas as pd
    from downstream.review_candidate_evidence import load_two_column_links
    with tempfile.TemporaryDirectory() as tmp:
        source=Path(tmp);extract('reaction_sources.zip','reactions/',source)
        links=load_two_column_links(source/'compound_pathways.tsv')
    d=pd.read_csv(DATA/'candidates/candidate_predictions.csv')
    labels=pd.read_csv(DATA/'training/pathways.csv'); space=set(labels.loc[labels.level.eq('L2'),'pathway_id'])
    unassigned=np.asarray([not any(links[k]&space for k in str(ids).split('|')) for ids in d.kegg_id])
    mask=(d.l1_three_seed_consensus & d.l2_top1_seed_votes.ge(2) & d.l2_in_all_seed_top3 & d.l1_l2_consistent &
          d.l1_ensemble_probability.ge(.06774) & d.l2_ensemble_probability.ge(.51135) & ~d.exact_training_duplicate & unassigned)
    selected=d.loc[mask].copy();selected.to_csv(out/'final_nominations.csv',index=False)
    selected.groupby('predicted_l1_name').size().rename('nominations').to_csv(out/'class_counts.csv')
    (out/'summary.json').write_text(json.dumps(dict(eligible_structures=len(d),nominations=len(selected)),indent=2)+'\n')

def edits(out):
    import pandas as pd
    from structural_edits import edits as generate
    inputs=pd.read_csv(DATA/'reactions/structural_edits.csv').fillna('')
    originals=inputs[inputs.edit_type.eq('original')]
    generated=[]
    for r in originals.itertuples():
        center={int(x) for x in str(r.approximate_center_atoms).split(';') if x}
        for edit in generate(r.original_smiles,center):generated.append(dict(candidate_kegg_id=r.candidate_kegg_id,**edit))
    pd.DataFrame(generated).to_csv(out/'generated_edits.csv',index=False)
    scores=pd.read_csv(DATA/'reactions/structural_edit_scores.csv')
    baseline=scores[scores.edit_type.eq('original')][['candidate_kegg_id','seed','target_l2_probability']].rename(columns={'target_l2_probability':'original_score'})
    s=scores.merge(baseline,on=['candidate_kegg_id','seed'],validate='many_to_one')
    s['score_drop']=s.original_score-s.target_l2_probability
    s=s.merge(inputs.reset_index(names='input_row')[['input_row','targets_approximate_center','edited_atom_indices']],on='input_row',validate='many_to_one')
    s.to_csv(out/'edit_score_changes.csv',index=False)
    s[~s.edit_type.eq('original')].groupby(['candidate_kegg_id','input_row','edit_type','targets_approximate_center']).agg(mean_score_drop=('score_drop','mean'),minimum_score_drop=('score_drop','min'),models=('seed','nunique')).reset_index().to_csv(out/'edit_summary.csv',index=False)
    (out/'summary.json').write_text(json.dumps(dict(compounds=len(originals),evaluated_edits=int((~inputs.edit_type.eq('original')).sum()),model_runs=int(scores.seed.nunique())),indent=2)+'\n')

def mvpool(out,input_path,device):
    import numpy as np
    import pandas as pd
    import torch
    from molpar_runtime import load
    from torch_geometric.data import Batch
    model,df,ds,split,args,meta=load(device)
    from dataset import HierarchicalMoleculeDataset
    if input_path:
        df=pd.read_csv(input_path);df['labels_L1']='';df['labels_L2']=''
        ds=HierarchicalMoleculeDataset(df,mode='both',cache=True)
    rows=[]
    with torch.no_grad():
        for i in range(len(df)):
            graph=Batch.from_data_list([ds[i][0]]).to(device)
            _,_,score,_,_=model.encoder.graph_encoder(graph,return_all=True)
            score=score.cpu().numpy().reshape(-1);cut=np.quantile(score,.7)
            for atom,value in enumerate(score):
                rows.append(dict(row_index=i,smiles=df.iloc[i].smiles,atom_index=atom,mvpool_score=float(value),highlight=bool(value>=cut)))
    pd.DataFrame(rows).to_csv(out/'atom_scores.csv',index=False)

def run(task,out,args):
    out.mkdir(parents=True,exist_ok=False)
    if task=='chemical-space':
        import chemical_space
        return chemical_space.run(DATA,out)
    if task=='candidates':return candidates(out)
    if task=='reactions':
        import reactions
        return reactions.run(DATA,out,args.permutations)
    if task=='families':
        import families
        return families.run(DATA,out)
    if task=='edits':return edits(out)
    if task=='mvpool':return mvpool(out,args.input,args.device)
    with tempfile.TemporaryDirectory(prefix='molpar-analysis-') as temp:
        work=Path(temp);ref=reference(work)
        if task=='network':
            import network
            source=work/'sources';extract('reaction_sources.zip','reactions/',source)
            nom=work/'nomination';(nom/'cache').mkdir(parents=True)
            import pandas as pd
            d=pd.read_csv(DATA/'candidates/final_nominations.csv');d['high_confidence_nomination']=True
            d.to_csv(nom/'candidate_catalogue_with_support.csv',index=False)
            shutil.copy2(source/'compound_pathways.tsv',nom/'cache/link_pathway_compound')
            model=work/'model';model.mkdir();shutil.copytree(ref,model/'data')
            network.NOMINATION=nom;network.MODEL_ROOT=model;network.SOURCE=source;network.OUT=out;network.CONTROL_PATH=DATA/'reactions/network_support.csv'
            return network.main()
        if task=='drugs':
            import drugs
            model=work/'MVP_model_0706';model.mkdir();shutil.copytree(ref,model/'data')
            # Original input names are created only in the temporary working directory.
            inputs=work/'drugs';(inputs/'predictions').mkdir(parents=True)
            for source,name in [('drugs.csv','drug_catalogue.csv'),('target_pathway_reference.csv','drug_target_pathway_evidence.csv')]:shutil.copy2(DATA/'drugs'/source,inputs/name)
            shutil.copy2(DATA/'drugs/pathway_scores.npz',inputs/'predictions/ensemble_drug_scores.npz')
            (inputs/'dataset_summary.json').write_text(json.dumps({'standardized_drugs':1696,'source':'ChEMBL 37'}))
            drugs.HERE=out;drugs.INPUT=inputs;drugs.ROOT=work
            return drugs.main()
        if task=='metabolomics':
            import metabolomics
            inputs=work/'inputs';inputs.mkdir()
            extract('metabolomics_sources.zip','metabolomics/',inputs/'raw')
            for source,name in [('sample_metadata.csv','sample_metadata.csv'),('structure_mapping.csv','metabolite_structure_map.csv'),('measurement_profiles.csv','ST002787_measurement_profiles.csv'),('oxylipin_scope.csv','oxylipin_scope.csv'),('arachidonic_scope.csv','arachidonic_bridge_scope.csv')]:shutil.copy2(DATA/'metabolomics'/source,inputs/name)
            import pandas as pd
            for study in ['ST000284','ST002787']:
                pd.read_csv(DATA/f'metabolomics/{study}_pathway_consensus.csv.gz').to_csv(out/f'{study}_ra_glgn_consensus.csv',index=False)
            shutil.copy2(DATA/'metabolomics/serum_case_scores.csv',out/'ST000284_ra_glgn_l2_scores.csv')
            metabolomics.build_tables(work,out)
            # Scores are inputs, already distributed in data/final; retain computed outputs here.
            for name in ['ST000284_ra_glgn_consensus.csv','ST002787_ra_glgn_consensus.csv','ST000284_ra_glgn_l2_scores.csv']:(out/name).unlink()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis',required=True,choices=TASKS+['all'])
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--permutations',type=int,default=1000)
    p.add_argument('--input',type=Path,help='Optional CSV for MVPool atom scoring')
    p.add_argument('--device',default='cpu')
    a=p.parse_args()
    if a.output_dir.exists():p.error('Choose a new output directory')
    if a.permutations<1000:p.error('Use at least 1000 permutations')
    tasks=[t for t in TASKS if t not in ['mvpool','chemical-space']] if a.analysis=='all' else [a.analysis]
    for task in tasks:
        target=a.output_dir.resolve()/task if a.analysis=='all' else a.output_dir.resolve()
        print('Running',task,flush=True);run(task,target,a);print('Saved',target,flush=True)
if __name__=='__main__':main()
