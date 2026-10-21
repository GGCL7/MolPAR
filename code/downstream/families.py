"""Rebuild evidence-anchored ECFP4/Butina chemical families."""
import json
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina

def run(data,out,threshold=.55):
    RDLogger.DisableLog('rdApp.warning')
    candidates=pd.read_csv(data/'candidates/candidate_predictions.csv')
    # Family construction uses the frozen cross-model nomination population.
    high=candidates[candidates.high_confidence_nomination].copy().reset_index(drop=True)
    review=pd.read_csv(data/'reactions/reviewed_candidates.csv')
    anchors=set(review.loc[review.direct_support,'kegg_id'])
    fp=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    fps=[fp.GetFingerprint(Chem.MolFromSmiles(s)) for s in high.smiles]
    distances=[]
    for i in range(1,len(fps)):
        distances.extend(1-v for v in DataStructs.BulkTanimotoSimilarity(fps[i],fps[:i]))
    clusters=Butina.ClusterData(distances,len(fps),1-threshold,isDistData=True,reordering=True)
    high['chemical_cluster_id']=0
    for i,cluster in enumerate(clusters,1):high.loc[list(cluster),'chemical_cluster_id']=i
    high['direct_anchor']=high.kegg_id.isin(anchors)
    high['reaction_context']=high.has_kegg_reaction | high.rhea_reaction_count.gt(0)
    rows=[]; members=[]
    for cid,cluster in high.groupby('chemical_cluster_id'):
        for pathway,group in cluster.groupby('predicted_l2_map'):
            a=group[group.direct_anchor]; e=group[~group.direct_anchor]
            if a.empty or e.empty:continue
            fid=len(rows)+1
            rows.append(dict(family_id=fid,chemical_cluster_id=int(cid),pathway_id=pathway,pathway_name=group.predicted_l2_name.iloc[0],
                             cluster_size=len(cluster),anchor_pathway_fraction=len(group)/len(cluster),direct_anchors=len(a),
                             expansion_members=len(e),reaction_context_members=int(e.reaction_context.sum()),
                             prospective_members=int((~e.reaction_context).sum())))
            for r in group.to_dict('records'):
                members.append(dict(family_id=fid,kegg_id=r['kegg_id'],name=r['name'],smiles=r['smiles'],pathway_id=pathway,
                                    member_role='direct_anchor' if r['direct_anchor'] else 'model_expansion',
                                    reaction_context=r['reaction_context'],max_reference_tanimoto=r['max_train_ecfp4_tanimoto'],scaffold_status=r['scaffold_status']))
    pd.DataFrame(rows).to_csv(out/'chemical_families.csv',index=False)
    members=pd.DataFrame(members);members.to_csv(out/'family_members.csv',index=False)
    high[['kegg_id','chemical_cluster_id','predicted_l2_map']].to_csv(out/'cluster_assignments.csv',index=False)
    e=members[members.member_role.eq('model_expansion')].drop_duplicates('kegg_id')
    summary=dict(families=len(rows),expansion_compounds=len(e),reaction_context=int(e.reaction_context.sum()),
                 prospective=int((~e.reaction_context).sum()),tanimoto_below_08=int(e.max_reference_tanimoto.lt(.8).sum()),unseen_scaffolds=int(e.scaffold_status.eq('unseen').sum()))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
