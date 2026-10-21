"""Produce model-dependent figure-8 tables and rerun measured effects from raw data.

The original chemical-class scope is retained as a pre-existing exploratory
selection, with new pathway memberships shown separately. No old model scores,
thresholds, memberships, or model-dependent effect summaries are imported.
"""
from pathlib import Path
import csv
import json


def bh(values):
    import numpy as np
    p = np.asarray(values, float)
    order = np.argsort(p)
    q = np.empty(len(p))
    last = 1.0
    for rank in range(len(p)-1, -1, -1):
        i = order[rank]
        last = min(last, p[i] * len(p)/(rank+1))
        q[i] = last
    return q


def design(meta, case='CRC'):
    import numpy as np
    import pandas as pd
    age = pd.to_numeric(meta.age).to_numpy(float)
    age = (age-age.mean())/age.std(ddof=1)
    return np.column_stack([np.ones(len(meta)), meta.disease_group.eq(case).to_numpy(float),
                            age, meta.sex.str.lower().str.startswith('m').to_numpy(float)])


def hc3(y, x):
    import numpy as np
    from scipy import stats
    inv = np.linalg.pinv(x.T @ x)
    b = inv @ x.T @ y
    residual = y-x@b
    leverage = np.sum((x@inv)*x, axis=1)
    adjusted = residual/np.maximum(1-leverage, 1e-8)
    cov = inv @ (x.T @ ((adjusted**2)[:, None]*x)) @ inv
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    df = max(1, len(y)-x.shape[1])
    p = 2*stats.t.sf(abs(b/np.where(se>0, se, np.nan)), df)
    critical = stats.t.ppf(.975, df)
    return {'std_effect': b[1], 'std_se_hc3': se[1], 'ci95_low': b[1]-critical*se[1],
            'ci95_high': b[1]+critical*se[1], 'p_value': p[1]}


def records_and_matrix(root, study, meta):
    import numpy as np
    data = list(json.loads((root/f'inputs/raw/{study}_data.json').read_text()).values())
    matrix = np.asarray([[float(row['DATA'][sid]) for row in data] for sid in meta.sample_id], float)
    if not np.isfinite(matrix).all() or np.any(matrix <= 0):
        raise ValueError('Non-finite/non-positive abundance; review preprocessing before rerun')
    log = np.log2(matrix)
    z = (log-log.mean(0))/log.std(0, ddof=1)
    return data, log, z


def clinical_tables(root, out):
    import numpy as np
    import pandas as pd
    meta_all = pd.read_csv(root/'inputs/sample_metadata.csv', dtype=str).fillna('')
    serum = meta_all[meta_all.study_id.eq('ST000284') & meta_all.disease_group.isin(['CRC','healthy'])].reset_index(drop=True)
    data, log, z = records_and_matrix(root, 'ST000284', serum)
    if z.shape != (148, 113):
        raise ValueError(f'Unexpected serum matrix {z.shape}')
    x = design(serum)
    rows = [{'study_id': 'ST000284', 'original_name': rec['metabolite_name'],
             'original_id': rec.get('metabolite_id', ''), 'units': rec.get('units', ''),
             'n': len(serum), 'n_crc': int(serum.disease_group.eq('CRC').sum()),
             'n_healthy': int(serum.disease_group.eq('healthy').sum()), **hc3(z[:, i], x)} for i, rec in enumerate(data)]
    effects = pd.DataFrame(rows)
    effects['fdr_bh'] = bh(effects.p_value)
    effects.to_csv(out/'ST000284_metabolite_effects.csv', index=False)
    stool = meta_all[meta_all.study_id.eq('ST002787')].reset_index(drop=True)
    data, log, z = records_and_matrix(root, 'ST002787', stool)
    if z.shape != (102, 1641):
        raise ValueError(f'Unexpected stool matrix {z.shape}')
    structures = pd.read_csv(root/'inputs/metabolite_structure_map.csv', dtype=str).fillna('')
    lookup = {}
    for r in structures[structures.study_id.eq('ST002787') & structures.smiles.ne('')].itertuples():
        lookup.setdefault((r.original_id, r.original_name), set()).add(r.inchi_key)
    columns = {}
    for i, r in enumerate(data):
        for key in lookup.get((r.get('metabolite_id',''), r['metabolite_name']), ()):
            columns.setdefault(key, []).append(i)
    structure_z = {key: z[:, ix].mean(1) for key, ix in columns.items()}
    profiles = pd.read_csv(root/'inputs/ST002787_measurement_profiles.csv', dtype=str).fillna('')
    profile_z = {}
    for p in profiles.itertuples():
        value = structure_z[p.representative_inchi_key]
        for key in p.candidate_inchi_keys.split('|'):
            if np.max(np.abs(structure_z[key]-value)) > 1e-10:
                raise ValueError(f'Raw-data profile equivalence failed: {p.measurement_profile_id}')
        profile_z[p.measurement_profile_id] = value
    rows = []
    for case in ['CRC', 'adenoma']:
        use = stool.disease_group.isin([case, 'healthy']).to_numpy()
        mm = stool.loc[use].reset_index(drop=True)
        x = design(mm, case)
        part = []
        for p in profiles.to_dict('records'):
            part.append({'study_id': 'ST002787', 'contrast': f'{case}_vs_healthy', **p,
                         'n': len(mm), 'n_case': int(mm.disease_group.eq(case).sum()),
                         'n_healthy': int(mm.disease_group.eq('healthy').sum()),
                         **hc3(profile_z[p['measurement_profile_id']][use], x)})
        pf = pd.DataFrame(part)
        pf['fdr_bh'] = bh(pf.p_value)
        rows.extend(pf.to_dict('records'))
    profile_effects = pd.DataFrame(rows)
    profile_effects.to_csv(out/'ST002787_profile_effects.csv', index=False)
    np.savez_compressed(out/'ST002787_profile_standardized_abundance.npz',
                        sample_ids=stool.sample_id.to_numpy(str),
                        profile_ids=np.asarray(list(profile_z)), values=np.column_stack(list(profile_z.values())))
    return effects, profile_effects


def build_tables(root, out):
    import numpy as np
    import pandas as pd
    root, out = Path(root), Path(out)
    training = pd.read_csv(root/'reference/kegg2026_hier.csv', dtype=str).fillna('')
    known = {}
    for r in training.itertuples():
        known.setdefault(r.inchikey, set()).update(int(v) for v in r.labels_L2.split(',') if v.strip())
    summary = {'model': 'MolPAR canonical final no-label-GCN', 'coverage': {},
               'scope': 'Only figure-8 CRC inputs and results; other figure groups unchanged',
               'identity_matching': 'actual SMILES-derived InChIKey for exact reference tests; original key retained for measurement joins'}
    memberships = {}
    for study in ['ST000284','ST002787']:
        consensus = pd.read_csv(out/f'{study}_ra_glgn_consensus.csv')
        kept = consensus[consensus.retained.eq(True)].copy()
        key_column = 'model_input_inchi_key' if 'model_input_inchi_key' in kept else 'inchi_key'
        kept['reference_overlap'] = [int(r.pathway_index) in known.get(getattr(r, key_column),set()) for r in kept.itertuples()]
        if study == 'ST000284':
            kept['member_set'] = np.where(kept.reference_overlap, 'RA_GLGN_existing', 'RA_GLGN_extension')
        else:
            kept['member_set'] = np.where(kept.reference_overlap, 'existing_KEGG_exact_structure', 'RA_GLGN_extension')
        kept.to_csv(out/f'{study}_annotation_extension_membership.csv', index=False)
        memberships[study] = kept
        summary['coverage'][study] = {'retained_structure_pathway_pairs':len(kept),
                                     'reference_overlap': int(kept.reference_overlap.sum()),
                                     'model_extension': int((~kept.reference_overlap).sum()),
                                     'retained_unique_source_structures': int(kept.inchi_key.nunique()),
                                     'structures_with_any_score': int(consensus.inchi_key.nunique())}
    serum_effects, profile_effects = clinical_tables(root, out)
    serum_scores = pd.read_csv(out/'ST000284_ra_glgn_l2_scores.csv')
    cases = []
    for name, target in [('Histidine','map00340'),('Glyceraldehyde','map00040')]:
        selected = serum_scores[serum_scores.original_name.eq(name) & serum_scores.pathway_id.eq(target)].copy()
        current = memberships['ST000284']
        present = current[current.original_name.eq(name) & current.pathway_id.eq(target)]
        selected['retained_by_latest_consensus'] = len(present)>0
        selected['latest_member_set'] = present.iloc[0].member_set if len(present) else 'not_retained'
        cases.extend(selected.to_dict('records'))
    pd.DataFrame(cases).to_csv(out/'figure8_C1_C2_latest_scores.csv', index=False)
    profiles = pd.read_csv(root/'inputs/ST002787_measurement_profiles.csv', dtype=str).fillna('')
    key_to_profile = {key:r.measurement_profile_id for r in profiles.itertuples() for key in r.candidate_inchi_keys.split('|')}
    scope = pd.read_csv(root/'inputs/oxylipin_scope.csv', dtype=str).fillna('')
    bridge_scope = pd.read_csv(root/'inputs/arachidonic_bridge_scope.csv', dtype=str).fillna('')
    scope_keys = set(bridge_scope.inchi_key)
    aa = memberships['ST002787']
    aa = aa[aa.pathway_id.eq('map00590') & aa.inchi_key.isin(scope_keys)].copy()
    aa['measurement_profile_id'] = aa.inchi_key.map(key_to_profile)
    crc_effects = profile_effects[profile_effects.contrast.eq('CRC_vs_healthy')]
    aa = aa.merge(crc_effects[['measurement_profile_id','std_effect','p_value','fdr_bh']], on='measurement_profile_id', how='left')
    aa.to_csv(out/'ST002787_oxylipin_arachidonic_members_profile_map.csv', index=False)
    branch = scope.merge(crc_effects[['measurement_profile_id','std_effect','std_se_hc3','ci95_low','ci95_high','p_value','fdr_bh']], on='measurement_profile_id', validate='one_to_one')
    branch.rename(columns={'fdr_bh':'profile_fdr_bh'}, inplace=True)
    branch['study_id'] = 'ST002787'
    branch['contrast'] = 'CRC_vs_healthy'
    branch['direction'] = np.where(branch.std_effect>0,'higher_in_CRC','lower_in_CRC')
    all_aa = memberships['ST002787']
    all_aa = all_aa[all_aa.pathway_id.eq('map00590')].copy()
    all_aa['measurement_profile_id'] = all_aa.inchi_key.map(key_to_profile)
    branch['latest_model_arachidonic_candidate_count'] = branch.measurement_profile_id.map(all_aa.groupby('measurement_profile_id').size()).fillna(0).astype(int)
    branch.to_csv(out/'ST002787_oxylipin_branch_context_profile_dedup.csv', index=False)
    target_scores = pd.read_csv(out/'ST002787_ra_glgn_consensus.csv')
    target_scores = target_scores[target_scores.pathway_id.eq('map00590')].copy()
    scope_members = scope[['measurement_profile_id','candidate_inchi_keys','product_group']].copy()
    scope_members['inchi_key'] = scope_members.candidate_inchi_keys.str.split('|')
    scope_members = scope_members.explode('inchi_key').drop(columns='candidate_inchi_keys')
    supplement = scope_members.merge(target_scores, on='inchi_key', how='left', validate='many_to_one')
    supplement['shown_in_structure_bridge'] = supplement.inchi_key.isin(set(aa.inchi_key))
    supplement.to_csv(out/'oxylipin_case_annotation_supplement.csv', index=False)
    summary['serum_metabolites_tested'] = len(serum_effects)
    summary['serum_metabolites_fdr_lt_005'] = int(serum_effects.fdr_bh.lt(.05).sum())
    summary['oxylipin_scope'] = 'Existing exploratory chemical-class scope frozen; latest model annotation recalculated separately'
    summary['arachidonic_bridge_scope'] = 'Original model-independent Butina cluster 1 at Tanimoto similarity 0.55, restricted to measured ST002787 structures'
    summary['arachidonic_bridge'] = {'candidate_structures':int(aa.inchi_key.nunique()),
                                    'measurement_profiles':int(aa.measurement_profile_id.nunique()),
                                    'reference_relations':int(aa.reference_overlap.sum()),
                                    'extension_relations':int((~aa.reference_overlap).sum())}
    summary['arachidonic_full_cohort'] = {'candidate_structures':int(all_aa.inchi_key.nunique()),
                                         'measurement_profiles':int(all_aa.measurement_profile_id.nunique()),
                                         'reference_relations':int(all_aa.reference_overlap.sum()),
                                         'extension_relations':int((~all_aa.reference_overlap).sum())}
    summary['cases'] = {name: {'target_pathway':path, 'retained':bool(any(row['original_name']==name and row['retained_by_latest_consensus'] for row in cases))}
                        for name,path in [('Histidine','map00340'),('Glyceraldehyde','map00040')]}
    (out/'figure8_update_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
