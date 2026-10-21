"""Pathway-specific reaction support and enzyme-class permutation tests."""
import json
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

EC_CLASSES = ['oxidoreductase','transferase','hydrolase','lyase','isomerase','ligase','transport/context-only']

def run(data, out, permutations=1000, seed=20260914):
    c = pd.read_csv(data/'candidates/candidate_predictions.csv')
    evidence = pd.read_csv(data/'reactions/pathway_evidence.csv').fillna('').set_index('candidate_kegg_id').loc[c.kegg_id]
    labels = pd.read_csv(data/'training/pathways.csv')
    maps = labels.loc[labels.level.eq('L2')].sort_values('label_index').pathway_id.tolist()
    index = {m:i for i,m in enumerate(maps)}
    pred = c.predicted_l2_map.map(index).to_numpy()
    l1 = c.predicted_l1_index.to_numpy()
    eligible = evidence.eligible.to_numpy(bool)
    contextual = eligible & evidence.reaction_context.to_numpy(bool)
    groups = [np.flatnonzero(eligible & (l1 == v)) for v in sorted(set(l1))]
    matrices = {name:np.asarray([[m in str(v).split(';') for m in maps] for v in evidence[col]])
                for name,col in [('pathway_reaction','direct_maps'),('ec_pathway','ec_maps')]}
    rng = np.random.default_rng(seed)
    rows, null_rows = [], []
    for scope,mask in [('all_robust_eligible',eligible),('reaction_context',contextual)]:
        for mode in ['global','within_L1']:
            null = {k:[] for k in matrices}
            for b in range(permutations):
                shuffled = pred.copy()
                for g in ([np.flatnonzero(eligible)] if mode == 'global' else groups):
                    shuffled[g] = rng.permutation(pred[g])
                for key,matrix in matrices.items():
                    count = int(matrix[np.flatnonzero(mask),shuffled[mask]].sum())
                    null[key].append(count)
                    null_rows.append(dict(scope=scope,shuffle=mode,metric=key,permutation=b,count=count))
            for key,counts in null.items():
                obs = int(matrices[key][np.flatnonzero(mask),pred[mask]].sum())
                mean = float(np.mean(counts))
                rows.append(dict(scope=scope,shuffle=mode,metric=key,denominator=int(mask.sum()),observed_count=obs,
                                 null_mean_count=mean,observed_null_ratio=obs/mean if mean else None,
                                 permutation_p_value=float((1+(np.asarray(counts)>=obs).sum())/(permutations+1))))
    pd.DataFrame(rows).to_csv(out/'permutation_tests.csv',index=False)
    pd.DataFrame(null_rows).to_csv(out/'permutation_null_counts.csv',index=False)
    master = pd.read_csv(data/'reactions/reaction_evidence.csv').fillna('')
    classsets = master.loc[~master.currency_excluded].groupby('candidate_kegg_id').ec_class.agg(set).to_dict()
    cm = np.asarray([[cl in classsets.get(cid,set()) for cl in EC_CLASSES] for cid in c.kegg_id])
    # L2 tests preserve L1; L1 tests shuffle broad-class labels globally.
    for level,codes,within in [('L2',pred,True),('L1',l1,False)]:
        levels = sorted(set(codes)); eye = np.eye(max(levels)+1,dtype=int)
        obs = cm[eligible].T @ eye[codes[eligible]]
        total = np.zeros_like(obs,float); greater = np.zeros_like(obs,int)
        for b in range(permutations):
            shuffled = codes.copy()
            for g in (groups if within else [np.flatnonzero(eligible)]):
                shuffled[g] = rng.permutation(codes[g])
            count = cm[eligible].T @ eye[shuffled[eligible]]
            total += count; greater += count >= obs
        tests = []
        for ci,cl in enumerate(EC_CLASSES):
            for li in levels:
                if within and not np.any(eligible & (codes==li)):continue
                mean = total[ci,li]/permutations
                tests.append(dict(ec_class=cl,pathway=maps[li] if within else int(li),candidate_count=int(obs[ci,li]),
                                  null_mean_count=mean,observed_null_ratio=obs[ci,li]/mean if mean else np.nan,
                                  permutation_p_value=(1+greater[ci,li])/(permutations+1)))
        table = pd.DataFrame(tests)
        table['fdr_bh'] = false_discovery_control(table.permutation_p_value.to_numpy())
        table.to_csv(out/f'enzyme_{level}_enrichment.csv',index=False)
    reviewed = pd.read_csv(data/'reactions/reviewed_candidates.csv')
    reviewed.groupby('evidence_tier').size().rename('candidates').to_csv(out/'reviewed_evidence_counts.csv')
    network = pd.read_csv(data/'reactions/network_support.csv')
    ids = set(pd.read_csv(data/'candidates/final_nominations.csv').kegg_id)
    network = network[network.kegg_id.isin(ids) & network.network_start_available]
    network.groupby('scaffold_status').agg(evaluable=('kegg_id','size'),direct_support=('predicted_l2_within_1_steps','sum')).to_csv(out/'network_scaffold_support.csv')
    # The same size-matched, within-class Monte Carlo control used by the study.
    rng = np.random.default_rng(20260908); tests=[]
    for step in (1,2,3):
        observed = int(network[f'predicted_l2_within_{step}_steps'].sum())
        prob = network[f'matched_within_{step}_fraction'].fillna(0).to_numpy()
        null = np.asarray([(rng.random(len(prob)) < prob).sum() for _ in range(10000)])
        tests.append(dict(steps=step,evaluable=len(prob),supported=observed,matched_support_rate=float(prob.mean()),
                          fold_enrichment=float(observed/len(prob)/prob.mean()),p_value=float((1+(null>=observed).sum())/10001)))
    pd.DataFrame(tests).to_csv(out/'network_matched_controls.csv',index=False)
    summary = next(r for r in rows if r['scope']=='reaction_context' and r['shuffle']=='within_L1' and r['metric']=='pathway_reaction')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
