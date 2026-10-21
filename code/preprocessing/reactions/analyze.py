#!/usr/bin/env python3
"""Reaction-aware hypothesis generation with explicit denominators and nulls."""
from __future__ import annotations
import argparse
import json
import sys
from collections import Counter, defaultdict, deque
from itertools import combinations
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator, rdFMCS
from sources import HERE, ROOT, NOM, digest, fetch_base, fetch_uniprot
sys.path.insert(0, str(ROOT / 'downstream/nomination_refinement'))
from review_candidate_evidence import parse_rhea_sides, load_two_column_links
RDLogger.DisableLog('rdApp.*')
REF = ROOT / 'downstream/nomination_refinement'
MODEL = ROOT / 'MVP_model_0706'
EC_CLASSES = ['oxidoreductase','transferase','hydrolase','lyase','isomerase','ligase','transport/context-only']
# Explicit currency exclusion complements the graph-degree rule. Acyl-CoA
# derivatives are not excluded merely for containing CoA as a substructure.
CURRENCY_KEGG = set('C00001 C00002 C00003 C00004 C00005 C00006 C00007 C00008 C00009 C00010 C00011 C00013 C00014 C00020 C00080 C00015 C00044 C00035 C00055 C00063 C00075 C00016 C01352'.split())
CURRENCY_CHEBI = {'CHEBI:'+x for x in '15377 15378 15379 57540 57945 58349 57783 57287 30616 456216 43474 18361 29888 28938'.split()}
FG = {'hydroxyl':'[OX2H][CX4]', 'carbonyl':'[CX3]=[OX1]', 'phosphate':'[P](=[O])([O])[O]',
      'carboxyl':'[CX3](=O)[O;H1,-1]', 'amine':'[NX3;H1,H2;!$(N-C=O)]',
      'glycosyl':'[C;R]1[O;R][C;R][C;R][C;R][C;R]1', 'acyl':'[CX3](=O)[O,N,S][C]'}
FP = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
def vals(x):
    return set(str(x).split(';')) - {'','nan'} if pd.notna(x) else set()
def joined(xs): return ';'.join(sorted(set(xs)))
def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
def ec_class(ec):
    return EC_CLASSES[int(ec[0])-1] if ec and ec[0] in '1234567' else 'unclassified'
def resolve_side(reaction, candidate_chebi):
    if not reaction['aligned']: return 'unresolved', set()
    left,right = set(reaction['left']),set(reaction['right'])
    a,b=bool(left & candidate_chebi),bool(right & candidate_chebi)
    if a == b: return 'ambiguous', set()
    return ('left',right) if a else ('right',left)
def tier(robust, currency, duplicate, unassigned, direct, second, family, context):
    if currency or duplicate or not unassigned or not robust: return 'excluded'
    if direct and second: return 'A'
    if not direct and second and family and context: return 'B'
    if direct or (family and context): return 'C'
    return 'context-only'
@lru_cache(maxsize=20000)
def fingerprint(smiles):
    mol=Chem.MolFromSmiles(smiles)
    return (mol,FP.GetFingerprint(mol)) if mol is not None else (None,None)
@lru_cache(maxsize=20000)
def structure_delta(smiles1, smiles2):
    a,fa=fingerprint(smiles1); b,fb=fingerprint(smiles2)
    if a is None or b is None: return {'status':'missing_structure'}
    mcs=rdFMCS.FindMCS([a,b],timeout=2,ringMatchesRingOnly=True,completeRingsOnly=True,
                      bondCompare=rdFMCS.BondCompare.CompareAny)
    q=Chem.MolFromSmarts(mcs.smartsString)
    aa=a.GetSubstructMatch(q) if q else (); bb=b.GetSubstructMatch(q) if q else ()
    mapping=dict(zip(aa,bb)); reverse={v:k for k,v in mapping.items()}
    changed=[]
    for bond in a.GetBonds():
        x,y=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
        if x in mapping and y in mapping:
            other=b.GetBondBetweenAtoms(mapping[x],mapping[y])
            if other and other.GetBondType()!=bond.GetBondType(): changed.extend([x,y])
    delta={name:len(b.GetSubstructMatches(Chem.MolFromSmarts(s)))-len(a.GetSubstructMatches(Chem.MolFromSmarts(s))) for name,s in FG.items()}
    unmatched_a=set(range(a.GetNumAtoms()))-set(aa)
    unmatched_b=set(range(b.GetNumAtoms()))-set(bb)
    # An MCS is not atom mapping. These are differences under one approximate
    # correspondence, not experimentally established broken/formed bonds.
    motifs=[]
    if delta['hydroxyl']*delta['carbonyl']<0: motifs.append('hydroxyl/carbonyl_change')
    for key in ['phosphate','glycosyl','acyl','carboxyl','amine']:
        if delta[key]: motifs.append(key+'_change')
    if delta['hydroxyl'] and not delta['carbonyl']: motifs.append('hydroxyl_change')
    dc=sum(x.GetSymbol()=='C' for x in b.GetAtoms())-sum(x.GetSymbol()=='C' for x in a.GetAtoms())
    if dc: motifs.append('carbon_count_change')
    return {'status':'approximate_mcs_timeout' if mcs.canceled else 'approximate_mcs',
            'motifs':joined(motifs), 'functional_group_delta':json.dumps(delta,sort_keys=True),
            'anchor_candidate_similarity':DataStructs.TanimotoSimilarity(fa,fb),
            'mcs_atom_fraction':mcs.numAtoms/max(a.GetNumAtoms(),b.GetNumAtoms()),
            'candidate_center_atom_indices':joined(map(str,set(changed)|unmatched_a)),
            'unmatched_candidate_atoms':joined(map(str,unmatched_a)),
            'unmatched_opposite_atoms':joined(map(str,unmatched_b)),
            'approximate_bond_order_changed_atoms':joined(map(str,changed)),
            'broken_bonds':'unresolved_without_atom_mapping', 'formed_bonds':'unresolved_without_atom_mapping',
            'carbon_count_delta':dc,
            'morgan_bits_lost':len(set(fa.GetOnBits())-set(fb.GetOnBits())),
            'morgan_bits_gained':len(set(fb.GetOnBits())-set(fa.GetOnBits()))}

def build(out, online):
    source=out/'sources'
    if online: fetch_base(source)
    required=['rhea_reactions.tsv','kegg_compound_to_chebi.tsv','kegg_enzyme_to_pathway.tsv','compound_pathways.tsv','manifest.json']
    if any(not (source/x).exists() for x in required): raise FileNotFoundError('Run with --fetch to obtain source snapshots')
    for name,meta in json.loads((source/'manifest.json').read_text()).items():
        path=source/name
        if path.is_file() and digest(path)!=meta['sha256']: raise ValueError('Source checksum mismatch: '+name)
    candidates=pd.read_csv(NOM/'candidate_catalogue_with_support.csv').fillna('')
    high_ids=set(pd.read_csv(NOM/'high_confidence_candidates.csv').kegg_id)
    labels=json.loads((MODEL/'data/label_space_L2.json').read_text())
    maps=[labels['index_to_map'][str(i)] for i in range(146)]
    parents={m:int(labels['map_to_l1_index'][m]) for m in maps}
    train=pd.read_csv(MODEL/'data/kegg2026_hier.csv').fillna('')
    cp=load_two_column_links(source/'compound_pathways.tsv')
    ep=load_two_column_links(source/'kegg_enzyme_to_pathway.tsv')
    kc=defaultdict(set); ck=defaultdict(set)
    for line in (source/'kegg_compound_to_chebi.tsv').read_text().splitlines():
        k,c=line.split('\t');k=k.split(':')[-1];c='CHEBI:'+c.split(':')[-1];kc[k].add(c);ck[c].add(k)
    rhea=pd.read_csv(source/'rhea_reactions.tsv',sep='\t',dtype=str).fillna('')
    if len(rhea.columns)!=6 or rhea.columns[2]!='ChEBI identifier': raise ValueError('Unexpected Rhea schema')
    rhea.columns=['id','equation','chebi','ec','enzyme_count','pmid']
    degrees=Counter(); reactions={}; cr=defaultdict(set)
    for r in rhea.itertuples(index=False):
        ids=vals(r.chebi); degrees.update(ids)
        reactions[r.id]={**parse_rhea_sides(r.equation,r.chebi),'ecs':{v.removeprefix('EC:') for v in vals(r.ec)},'pmids':vals(r.pmid),'equation':r.equation}
        for c in ids: cr[c].add(r.id)
    cutoff=max(50,int(np.quantile(list(degrees.values()),.99)))
    currency=CURRENCY_CHEBI|{c for c,d in degrees.items() if d>=cutoff}|set().union(*(kc[k] for k in CURRENCY_KEGG))
    graph=defaultdict(lambda:defaultdict(set))
    for rid,r in reactions.items():
        if not r['aligned']: continue
        # A participant on both sides has no unambiguous conversion role.
        shared=set(r['left'])&set(r['right'])
        for a in sorted(set(r['left'])-currency-shared):
            for b in sorted(set(r['right'])-currency-shared):
                if a!=b: graph[a][b].add(rid);graph[b][a].add(rid)
    chebi_maps={c:set().union(*(cp[k]&set(maps) for k in ks)) for c,ks in ck.items()}
    candidate_chebi={r.kegg_id:set().union(*(kc[k] for k in r.kegg_id.split('|'))) for r in candidates.itertuples()}
    relevant=set().union(*(set().union(*(cr[c] for c in cs)) for cs in candidate_chebi.values()))
    if online: uni=fetch_uniprot(source,relevant)
    elif (source/'uniprot_relations.csv').exists(): uni=pd.read_csv(source/'uniprot_relations.csv',dtype=str).fillna('')
    else: raise FileNotFoundError('UniProt cache missing; run --fetch')
    ui=defaultdict(list)
    for u in uni.to_dict('records'): ui[(u['rhea_reaction_id'],u['ec_number'])].append(u)
    families=pd.read_csv(REF/'chemical_family_assignments.csv').set_index('kegg_id')
    family_counts=families.groupby(['chemical_cluster_id','predicted_l2_map']).size().to_dict()
    calibration=json.loads((NOM/'prediction_summary.json').read_text())
    threshold=calibration['pooled_l2_calibration']['threshold']
    l1_threshold=calibration['pooled_l1_calibration']['threshold']
    names={}
    listing=NOM/'cache/list_compound'
    if listing.exists():
        for line in listing.read_text().splitlines():
            k,n=line.split('\t',1); names[k.split(':')[-1]]=n
    smiles={}
    for frame in [train,candidates]:
        for r in frame.itertuples():
            for k in r.kegg_id.split('|'): smiles[k]=r.smiles
    evidence=[]; master=[]; paths=[]
    for r in candidates.itertuples(index=False):
        cid=r.kegg_id; cs=candidate_chebi[cid]; own=set(r.kegg_id.split('|'))
        curr=bool(cs&currency or own&CURRENCY_KEGG)
        unassigned=not any(cp[k]&set(maps) for k in own)
        robust=(cid in high_ids and r.l1_three_seed_consensus and r.l2_top1_seed_votes>=2 and r.l2_in_all_seed_top3 and r.l1_l2_consistent and r.l2_ensemble_probability>=threshold and r.l1_ensemble_probability>=l1_threshold)
        family_id=int(families.loc[cid,'chemical_cluster_id']) if cid in families.index else -1
        family=family_counts.get((family_id,r.predicted_l2_map),0)>=2
        direct_maps=set(); ec_maps=set(); dual_maps=set(); context_maps=set(); candidate_rows=[]
        rids=set().union(*(cr[c] for c in cs))
        for rid in sorted(rids):
            reaction=reactions[rid]; side,opposite=resolve_side(reaction,cs)
            opposite=opposite-currency-cs-(set(reaction['left'])&set(reaction['right']))
            opposite_kegg=set().union(*(ck[c] for c in opposite))-own
            omaps=set().union(*(cp[k]&set(maps) for k in opposite_kegg))
            if curr: omaps=set()
            direct_maps.update(omaps)
            for ec in sorted(reaction['ecs']) or ['']:
                proteins=ui.get((rid,ec),[])
                # EC mapping is from this reaction's EC, not any other reaction
                # of the molecule. Independent source does not mean independent experiment.
                emaps=ep[ec]&set(maps) if ec and not curr else set()
                ec_maps.update(emaps)
                second=bool(ec or proteins)
                dual_maps.update(omaps if second else set())
                l2=r.predicted_l2_map in omaps
                l1=any(parents[m]==r.predicted_l1_index for m in omaps)
                evidence_tier=tier(robust,curr,r.exact_training_duplicate,unassigned,l2 or l1,second,family, bool(rids))
                row={'candidate_kegg_id':cid,'candidate_name':r.name,'smiles':r.smiles,'predicted_l1':r.predicted_l1_name,
                     'predicted_l1_index':r.predicted_l1_index,'predicted_l2_map':r.predicted_l2_map,'predicted_l2_name':r.predicted_l2_name,
                     'ensemble_probability':r.l2_ensemble_probability,'seed_consensus':f'L1:3/3;L2:{r.l2_top1_seed_votes}/3;all_top3:{r.l2_in_all_seed_top3}',
                     'robust_nomination':bool(robust),'max_train_tanimoto':r.max_train_ecfp4_tanimoto,'scaffold_status':r.scaffold_status,
                     'rhea_reaction_id':rid,'candidate_chebi_ids':joined(cs),'candidate_reaction_side':side,
                     'opposite_side_compound_id':joined(opposite_kegg),'opposite_side_chebi_ids':joined(opposite),
                     'opposite_side_compound_name':joined(names.get(k,k) for k in opposite_kegg),
                     'ec_number':ec,'ec_class':ec_class(ec),'uniprot_accessions':joined(p['uniprot_accession'] for p in proteins),
                     'organisms':joined(p['organism'] for p in proteins),'organism_taxon_ids':joined(p['organism_taxon_id'] for p in proteins),
                     'reaction_center':'approximation_pending','known_pathway':joined(omaps),'ec_known_pathway':joined(emaps),
                     'pathway_consistency':'L2' if l2 else 'L1' if l1 else 'none',
                     'direct_l2_support':l2,'second_evidence':second,'ec_l2_consistency':r.predicted_l2_map in emaps,
                     'evidence_tier':evidence_tier,'evidence_sources':joined([rid]+(['EC:'+ec] if ec else [])+['UniProt:'+p['uniprot_accession'] for p in proteins]),
                     'pubmed_ids':joined(reaction['pmids']),'equation':reaction['equation'], 'chemical_cluster_id':family_id,
                     'currency_excluded':curr,'exact_training_duplicate':r.exact_training_duplicate,'currently_pathway_unassigned':unassigned,
                     'identity_mapping_status':'unique' if len(cs)==1 else 'multiple' if cs else 'unmapped',
                     'enzyme_evidence_scope':'known_reaction; candidate_specific_activity_not_tested'}
                candidate_rows.append(row)
        # Deterministic, side-aware BFS, retaining reaction IDs for every step.
        seen=set(cs); queue=deque((c,[c],[]) for c in sorted(cs-currency) if not curr)
        reached={}
        while queue:
            node,nodes,rr=queue.popleft()
            if len(rr)>=3: continue
            for neighbor in sorted(graph.get(node,{})):
                if neighbor in seen: continue
                seen.add(neighbor); nr=rr+[sorted(graph[node][neighbor])[0]]; nn=nodes+[neighbor]
                for m in sorted(chebi_maps.get(neighbor,set())):
                    if m not in reached: reached[m]=(nn,nr)
                queue.append((neighbor,nn,nr))
        for m,(nn,rr) in reached.items():
            context_maps.add(m)
            paths.append({'candidate_kegg_id':cid,'pathway':m,'steps':len(rr),'chebi_path':joined([]) if not nn else ';'.join(nn),'rhea_path':';'.join(rr)})
        evidence.append({'candidate_kegg_id':cid,'direct_maps':joined(direct_maps),'ec_maps':joined(ec_maps),'dual_maps':joined(dual_maps),
                         'context_maps':joined(context_maps),'reaction_context':bool(rids) and not curr,'eligible':bool(robust and not curr and not r.exact_training_duplicate and unassigned),
                         'family_consistent':family,'currently_pathway_unassigned':unassigned,'currency_excluded':curr})
        master.extend(candidate_rows)
    master=pd.DataFrame(master)
    master.to_csv(out/'reaction_substrate_master.csv',index=False)
    evidence=pd.DataFrame(evidence);evidence.to_csv(out/'method_independent_evidence.csv',index=False)
    pd.DataFrame(paths,columns=['candidate_kegg_id','pathway','steps','chebi_path','rhea_path']).to_csv(out/'reaction_network_paths.csv',index=False)
    write_json(out/'analysis_config.json',{'currency_kegg_ids':sorted(CURRENCY_KEGG),'currency_chebi_ids':sorted(currency),'degree_cutoff':cutoff,
               'l2_threshold':threshold,'l1_threshold':l1_threshold,'ec_class_mapping':dict(zip(range(1,8),EC_CLASSES)),
               'A_rule':'robust + noncurrency + no exact duplicate + unassigned + same-reaction opposite-side L1/L2 consistency + EC or same-reaction UniProt',
               'B_rule':'robust + noncurrency + no exact duplicate + unassigned + same-family/L2 member + own Rhea context + own EC/UniProt; no direct L1/L2 connection',
               'network_semantics':'opposite-side undirected graph, not physiological direction; all shared-side participants removed',
               'atom_mapping':'unavailable in selected Rhea endpoint; MCS/functional-group/Morgan approximations only',
               'input_sha256':{str(p.relative_to(ROOT)):digest(p) for p in [NOM/'high_confidence_candidates.csv',NOM/'candidate_catalogue_with_support.csv',NOM/'prediction_summary.json',REF/'candidate_evidence_review.csv',REF/'multihop_network_support.csv',REF/'evidence_anchored_family_members.csv',REF/'method_candidate_predictions.csv',MODEL/'data/kegg2026_hier.csv']}})
    return candidates,master,evidence,maps,parents,train,graph,smiles

def statistics(out,candidates,master,evidence,maps,parents,train,permutations,seed):
    if permutations<1000: raise ValueError('At least 1000 permutations required')
    e=evidence.set_index('candidate_kegg_id').loc[candidates.kegg_id]
    idx={m:i for i,m in enumerate(maps)}; n=len(candidates); pos=np.arange(n)
    matrices={key:np.array([[m in vals(v) for m in maps] for v in e[col]],dtype=bool) for key,col in [('pathway_reaction','direct_maps'),('ec_pathway','ec_maps'),('dual','dual_maps'),('context','context_maps')]}
    clusters=pd.read_csv(REF/'chemical_family_assignments.csv').set_index('kegg_id').chemical_cluster_id.to_dict()
    # Structural anchor adjacency is fixed before shuffling pathway assignments.
    # Build expansion eligibility before looking at predicted pathway labels:
    # the anchor pathway is its externally annotated opposite-side pathway.
    cluster_members=defaultdict(list)
    for cid,cluster in clusters.items():cluster_members[cluster].append(cid)
    relation_groups={cid:g for cid,g in master.groupby('candidate_kegg_id')}
    structures=candidates.set_index('kegg_id').smiles.to_dict()
    expansion=np.zeros((n,len(maps)),dtype=bool)
    for i,cid in enumerate(candidates.kegg_id):
        own=relation_groups.get(cid)
        if own is None or not e.iloc[i].eligible:continue
        own_ec=set(own.ec_number)-{''};own_reactions=set(own.rhea_reaction_id)
        if not own_ec:continue
        _,cf=fingerprint(structures[cid])
        if cf is None:continue
        for anchor_id in cluster_members.get(clusters.get(cid),[]):
            if anchor_id==cid:continue
            anchors=relation_groups.get(anchor_id)
            if anchors is None:continue
            _,af=fingerprint(structures[anchor_id])
            if af is None or DataStructs.TanimotoSimilarity(cf,af)<.55:continue
            for anchor in anchors.itertuples():
                if not anchor.ec_number or anchor.rhea_reaction_id in own_reactions or anchor.currency_excluded:continue
                if not any(ec.rsplit('.',1)[0]==anchor.ec_number.rsplit('.',1)[0] for ec in own_ec):continue
                for m in vals(anchor.known_pathway):expansion[i,idx[m]]=True
    matrices['substrate_expansion']=expansion
    pred=np.array([idx[m] for m in candidates.predicted_l2_map]); l1=candidates.predicted_l1_index.to_numpy()
    eligible=e.eligible.to_numpy(bool); contextual=eligible & e.reaction_context.to_numpy(bool)
    # Mean frequency across the three actual retrieval/training splits.
    import ast
    frequencies=[]
    for s in range(3):
        ids=ast.literal_eval((MODEL/f'data/splits/random_seed{s}.txt').read_text().splitlines()[0])
        freq=np.zeros(len(maps))
        for lab in train.iloc[ids].labels_L2:
            for j in str(lab).split(','):
                if j: freq[int(j)]+=1
        frequencies.append(freq)
    freq=np.mean(frequencies,axis=0)
    cp=load_two_column_links(out/'sources/compound_pathways.tsv')
    network_sizes=np.zeros(len(maps))
    kc=load_two_column_links(out/'sources/kegg_compound_to_chebi.tsv')
    for m,j in idx.items(): network_sizes[j]=len(set().union(*(kc[k] for k,ps in cp.items() if m in ps)))
    controls={}
    for m,j in idx.items():
        others=[k for k,p in enumerate(maps) if parents[p]==parents[m] and k!=j]
        controls[j]=sorted(others,key=lambda k:(abs(np.log1p(freq[k])-np.log1p(freq[j]))+abs(np.log1p(network_sizes[k])-np.log1p(network_sizes[j])),maps[k]))[:5]
    rows=[]
    for i,r in enumerate(candidates.itertuples()):
        if not contextual[i]: continue
        for k in controls[pred[i]]:
            rows.append({'candidate_kegg_id':r.kegg_id,'predicted_pathway':r.predicted_l2_map,'control_pathway':maps[k],
                         'train_frequency':freq[pred[i]],'control_train_frequency':freq[k],'network_size':network_sizes[pred[i]],'control_network_size':network_sizes[k],
                         'observed_direct':bool(matrices['pathway_reaction'][i,pred[i]]),'control_direct':bool(matrices['pathway_reaction'][i,k]),
                         'observed_ec':bool(matrices['ec_pathway'][i,pred[i]]),'control_ec':bool(matrices['ec_pathway'][i,k])})
    matched=pd.DataFrame(rows);matched.to_csv(out/'same_l1_matched_controls.csv',index=False)
    rng=np.random.default_rng(seed); groups=[np.flatnonzero(eligible & (l1==v)) for v in sorted(set(l1))]
    summary=[]; null_rows=[]
    # Actual label permutation, not independent Bernoulli sampling of support.
    for scope,mask in [('all_robust_eligible',eligible),('reaction_context',contextual)]:
        for mode in ['global','within_L1']:
            null={k:[] for k in ['pathway_reaction','ec_pathway','substrate_expansion']}
            for b in range(permutations):
                shuffled=pred.copy()
                for g in ([np.flatnonzero(eligible)] if mode=='global' else groups): shuffled[g]=rng.permutation(pred[g])
                for k in null:
                    count=int(matrices[k][pos[mask],shuffled[mask]].sum());null[k].append(count)
                    null_rows.append({'scope':scope,'shuffle':mode,'metric':k,'permutation':b,'count':count})
            for k,values in null.items():
                obs=int(matrices[k][pos[mask],pred[mask]].sum()); a=np.array(values); den=int(mask.sum())
                summary.append({'scope':scope,'shuffle':mode,'metric':k,'denominator':den,'observed_count':obs,'observed_rate':obs/den if den else None,
                                'null_mean_count':float(a.mean()),'null_sd_count':float(a.std(ddof=1)), 'observed_null_ratio':obs/a.mean() if a.mean() else None,
                                'permutation_p_value':float((1+(a>=obs).sum())/(permutations+1)),
                                'p_value_mc_se':float(np.sqrt(((1+(a>=obs).sum())/(permutations+1))*(1-(1+(a>=obs).sum())/(permutations+1))/(permutations+1)))})
    pd.DataFrame(null_rows).to_csv(out/'permutation_null_counts.csv',index=False)
    pd.DataFrame(summary).to_csv(out/'permutation_tests.csv',index=False)
    # EC/pathway enrichment permutes the candidate labels within their L1.
    classsets=master.loc[~master.currency_excluded].groupby('candidate_kegg_id').ec_class.agg(set).to_dict()
    class_matrix=np.array([[cl in classsets.get(cid,set()) for cl in EC_CLASSES] for cid in candidates.kegg_id])
    enrichment=[]; obs=class_matrix[eligible].T @ np.eye(len(maps),dtype=int)[pred[eligible]]
    ns=np.zeros_like(obs,float); nge=np.zeros_like(obs,int)
    for b in range(permutations):
        shuffled=pred.copy()
        for g in groups: shuffled[g]=rng.permutation(pred[g])
        count=class_matrix[eligible].T @ np.eye(len(maps),dtype=int)[shuffled[eligible]]
        ns+=count; nge+=count>=obs
    for ci,cl in enumerate(EC_CLASSES):
        for j,m in enumerate(maps):
            mask=eligible & (pred==j); den=int(mask.sum())
            if not den: continue
            relevant_mask=mask & class_matrix[:,ci]; cnt=int(relevant_mask.sum()); null=ns[ci,j]/permutations
            tierids={t:set(master.loc[master.evidence_tier.eq(t)&master.ec_class.eq(cl),'candidate_kegg_id']) for t in ['A','B']}
            enrichment.append({'ec_class':cl,'pathway':m,'pathway_name':candidates.loc[candidates.predicted_l2_map.eq(m),'predicted_l2_name'].iloc[0],
                               'l1_index':parents[m],'denominator':den,'candidate_count':cnt,'null_mean_count':null,'observed_null_ratio':cnt/null if null else np.nan,
                               'permutation_p_value':(1+nge[ci,j])/(permutations+1),
                               'direct_support_rate':float(matrices['pathway_reaction'][pos[relevant_mask],pred[relevant_mask]].mean()) if cnt else np.nan,
                               'A_count':len(set(candidates.loc[mask,'kegg_id'])&tierids['A']),'B_count':len(set(candidates.loc[mask,'kegg_id'])&tierids['B']),
                               'unseen_scaffold_count':int((candidates.scaffold_status.eq('unseen').to_numpy()&relevant_mask).sum()),
                               'median_max_train_tanimoto':float(candidates.loc[relevant_mask,'max_train_ecfp4_tanimoto'].median()) if cnt else np.nan})
    enrich=pd.DataFrame(enrichment)
    from scipy.stats import false_discovery_control
    enrich['fdr_bh']=false_discovery_control(enrich.permutation_p_value.to_numpy())
    enrich.to_csv(out/'reaction_class_enrichment.csv',index=False)
    # L1 enrichment needs a global shuffle: a within-L1 shuffle leaves L1
    # counts invariant and cannot test superclass enrichment.
    l1_levels=sorted(set(l1)); l1_idx={v:i for i,v in enumerate(l1_levels)}
    l1_codes=np.array([l1_idx[v] for v in l1]); l1_eye=np.eye(len(l1_levels),dtype=int)
    l1_obs=class_matrix[eligible].T@l1_eye[l1_codes[eligible]]
    l1_sum=np.zeros_like(l1_obs,float);l1_ge=np.zeros_like(l1_obs,int)
    for _ in range(permutations):
        shuffled=rng.permutation(l1_codes[eligible])
        count=class_matrix[eligible].T@l1_eye[shuffled]
        l1_sum+=count;l1_ge+=count>=l1_obs
    l1_rows=[]
    for ci,cl in enumerate(EC_CLASSES):
        for li,value in enumerate(l1_levels):
            mask=eligible&(l1==value);cmask=mask&class_matrix[:,ci];cnt=int(cmask.sum());null=l1_sum[ci,li]/permutations
            l1_rows.append({'ec_class':cl,'l1_index':int(value),'l1_name':candidates.loc[candidates.predicted_l1_index.eq(value),'predicted_l1_name'].iloc[0],
                           'denominator':int(mask.sum()),'candidate_count':cnt,'null_mean_count':null,'observed_null_ratio':cnt/null if null else np.nan,
                           'permutation_p_value':(1+l1_ge[ci,li])/(permutations+1),
                           'direct_l2_support_rate':float(matrices['pathway_reaction'][pos[cmask],pred[cmask]].mean()) if cnt else np.nan,
                           'unseen_scaffold_fraction':float(candidates.loc[cmask,'scaffold_status'].eq('unseen').mean()) if cnt else np.nan,
                           'median_max_train_tanimoto':float(candidates.loc[cmask,'max_train_ecfp4_tanimoto'].median()) if cnt else np.nan})
    l1_enrich=pd.DataFrame(l1_rows);l1_enrich['fdr_bh']=false_discovery_control(l1_enrich.permutation_p_value.to_numpy())
    l1_enrich.to_csv(out/'reaction_class_l1_enrichment.csv',index=False)

    consistency=master.groupby(['predicted_l1','predicted_l2_map','ec_class']).agg(relation_rows=('candidate_kegg_id','size'),candidate_count=('candidate_kegg_id','nunique'),direct_rows=('direct_l2_support','sum'),ec_consistent_rows=('ec_l2_consistency','sum')).reset_index()
    consistency.to_csv(out/'pathway_ec_consistency.csv',index=False)
    # All four methods evaluated using the same ID order and evidence matrix.
    method=pd.read_csv(REF/'method_candidate_predictions.csv')
    freq_order=np.argsort(-freq,kind='stable')
    frequency=pd.DataFrame({'kegg_id':candidates.kegg_id,'method':'Frequency','top1_map':maps[freq_order[0]],'top3_maps':';'.join(maps[j] for j in freq_order[:3])})
    method=pd.concat([method,frequency],ignore_index=True)
    details=[]
    for name,g in method.groupby('method'):
        if g.kegg_id.duplicated().any() or set(g.kegg_id)!=set(candidates.kegg_id): raise ValueError('Methods must use identical unique candidate pools')
        g=g.set_index('kegg_id').loc[candidates.kegg_id]
        for i,r in enumerate(g.itertuples()):
            ds=vals(e.iloc[i].direct_maps)
            details.append({'candidate_kegg_id':r.Index,'method':name,'top1_map':r.top1_map,'top3_maps':r.top3_maps,
                            'direct_top1':r.top1_map in ds,'direct_top3':bool(vals(r.top3_maps)&ds),'evaluable':bool(ds),
                            'low_tanimoto':candidates.iloc[i].max_train_ecfp4_tanimoto<.8,'unseen_scaffold':candidates.iloc[i].scaffold_status=='unseen'})
    details=pd.DataFrame(details);details.to_csv(out/'method_candidate_predictions.csv',index=False)
    comparison=[];overlap=[]
    for subset in ['all','evaluable','low_tanimoto','unseen_scaffold']:
        d=details if subset=='all' else details.loc[details[subset]]
        for name,g in d.groupby('method'):
            for k in [1,3]: comparison.append({'subset':subset,'method':name,'top_k':k,'denominator':len(g),'supported':int(g[f'direct_top{k}'].sum()),'support_rate':float(g[f'direct_top{k}'].mean())})
        for k in [1,3]:
            wide=d.pivot(index='candidate_kegg_id',columns='method',values=f'direct_top{k}')
            for name in ['RA-GLGN','Weighted-ECFP4-kNN','Fingerprint-LR']:
                others=[x for x in ['RA-GLGN','Weighted-ECFP4-kNN','Fingerprint-LR'] if x!=name]
                overlap.append({'subset':subset,'top_k':k,'category':name+'-only','denominator':len(wide),'count':int((wide[name]&~wide[others].any(axis=1)).sum())})
            overlap.append({'subset':subset,'top_k':k,'category':'all_three','denominator':len(wide),'count':int(wide[['RA-GLGN','Weighted-ECFP4-kNN','Fingerprint-LR']].all(axis=1).sum())})
    pd.DataFrame(comparison).to_csv(out/'method_support_comparison.csv',index=False);pd.DataFrame(overlap).to_csv(out/'method_support_overlap.csv',index=False)
    write_json(out/'reaction_type_summary.json',{'candidate_pool':n,'robust_eligible':int(eligible.sum()),'reaction_context_eligible':int(contextual.sum()),'permutations':permutations,'random_seed':seed,
               'permutation_tests':summary,'matched_control_candidates':int(matched.candidate_kegg_id.nunique()) if len(matched) else 0,
               'matched_direct_observed_rate':float(matched.groupby('candidate_kegg_id').observed_direct.first().mean()) if len(matched) else None,
               'matched_direct_null_rate':float(matched.groupby('candidate_kegg_id').control_direct.mean().mean()) if len(matched) else None,
               'significant_ec_pathway_cells_fdr_0_05':int((enrich.fdr_bh<.05).sum()),
               'significant_ec_l1_cells_fdr_0_05':int((l1_enrich.fdr_bh<.05).sum()),
               'expansion_statistic':'new-to-anchor reaction pairs in a fixed chemical cluster with pair Tanimoto >=0.55 and own exact/adjacent EC; shuffled candidate label compared to external opposite-side anchor pathway; anchor model label not used in null construction',
               'database_absence_interpretation':'unrecorded/unevaluable, not biological negative'})

def structures_and_bridges(out,candidates,master,evidence,maps,parents,graph,smiles):
    byid=candidates.set_index('kegg_id'); ev=evidence.set_index('candidate_kegg_id')
    families=pd.read_csv(REF/'chemical_family_assignments.csv')
    method=pd.read_csv(out/'method_candidate_predictions.csv')
    methods={m:g.set_index('candidate_kegg_id').top1_map.to_dict() for m,g in method.groupby('method')}
    anchors=master.loc[master.evidence_tier.eq('A') & master.direct_l2_support].drop_duplicates(['candidate_kegg_id','rhea_reaction_id','ec_number'])
    pairs=[]
    for anchor in anchors.itertuples():
        members=families.loc[families.chemical_cluster_id.eq(anchor.chemical_cluster_id)&families.predicted_l2_map.eq(anchor.predicted_l2_map)]
        am,af=fingerprint(anchor.smiles)
        if am is None: continue
        for r in members.itertuples():
            if r.kegg_id==anchor.candidate_kegg_id or not ev.loc[r.kegg_id,'eligible']: continue
            cm,cf=fingerprint(r.smiles)
            if cm is None: continue
            similarity=DataStructs.TanimotoSimilarity(af,cf)
            if similarity<.55: continue
            own=master.loc[master.candidate_kegg_id.eq(r.kegg_id)]
            ecs=set(own.ec_number)-{''}
            shared=anchor.ec_number in ecs and bool(anchor.ec_number)
            adjacent=any(ec.rsplit('.',1)[0]==anchor.ec_number.rsplit('.',1)[0] for ec in ecs) if anchor.ec_number else False
            candidates_tiers=set(own.evidence_tier)
            evidence_tier=next((t for t in ['A','B','C'] if t in candidates_tiers),'C')
            # Enzyme compatibility inherited from an anchor is a hypothesis;
            # it cannot alone promote the expansion to A/B.
            candidate_evidence_tier=evidence_tier
            if evidence_tier in ['A','B'] and not (shared or adjacent): evidence_tier='C'
            delta=structure_delta(r.smiles,anchor.smiles)
            pairs.append({'anchor_compound':anchor.candidate_kegg_id,'anchor_name':anchor.candidate_name,'candidate_expansion_compound':r.kegg_id,'candidate_name':r.name,
                          'anchor_rhea_reaction_id':anchor.rhea_reaction_id,'candidate_rhea_reaction_ids':joined(own.rhea_reaction_id),
                          'predicted_enzyme_class':anchor.ec_number,'reaction_class':anchor.ec_class,'predicted_pathway':r.predicted_l2_map,
                          'anchor_candidate_similarity':similarity,'candidate_training_max_similarity':r.max_train_ecfp4_tanimoto,'scaffold_novelty':r.scaffold_status,
                          'same_ec':shared,'adjacent_ec':adjacent,'candidate_ec_numbers':joined(ecs),'evidence_tier':evidence_tier,'chemical_cluster_id':anchor.chemical_cluster_id,
                          'candidate_evidence_tier':candidate_evidence_tier,
                          'candidate_enzyme_evidence':'own reaction EC compatible' if shared or adjacent else 'anchor-derived hypothesis only',
                          'mcs_atom_fraction':delta.get('mcs_atom_fraction',np.nan),'reaction_center_similarity':'unresolved_without_atom_mapping',
                          'priority':int(evidence_tier in ['A','B'])*4+int(r.max_train_ecfp4_tanimoto<.8)*2+int(r.scaffold_status=='unseen')})
    pair_columns=['anchor_compound','anchor_name','candidate_expansion_compound','candidate_name','anchor_rhea_reaction_id','candidate_rhea_reaction_ids','predicted_enzyme_class','reaction_class','predicted_pathway','anchor_candidate_similarity','candidate_training_max_similarity','scaffold_novelty','same_ec','adjacent_ec','candidate_ec_numbers','evidence_tier','chemical_cluster_id','candidate_evidence_tier','candidate_enzyme_evidence','mcs_atom_fraction','reaction_center_similarity','priority']
    pairs=pd.DataFrame(pairs,columns=pair_columns).sort_values(['priority','anchor_candidate_similarity'],ascending=False)
    participation=set(zip(master.candidate_kegg_id,master.rhea_reaction_id))
    pairs['candidate_already_in_anchor_reaction']=[(r.candidate_expansion_compound,r.anchor_rhea_reaction_id) in participation for r in pairs.itertuples()]
    pairs['substrate_scope_status']=np.where(pairs.candidate_already_in_anchor_reaction,'known_Rhea_participant; pathway-context extension','unrecorded_pair; substrate-scope hypothesis')
    pairs.to_csv(out/'anchor_expansion_pairs.csv',index=False)
    pairs.loc[~pairs.candidate_already_in_anchor_reaction].drop_duplicates('candidate_expansion_compound').to_csv(out/'substrate_scope_expansion_candidates.csv',index=False)
    motif_rows=[]
    valid=master.loc[master.evidence_tier.isin(['A','B','C']) & master.candidate_reaction_side.isin(['left','right'])]
    for r in valid.drop_duplicates(['candidate_kegg_id','rhea_reaction_id']).itertuples():
        # Retain every noncurrency opposite-side structure, not just the most
        # similar one: approximation artifacts stay inspectable.
        for opposite in sorted(vals(r.opposite_side_compound_id)):
            if opposite not in smiles:
                motif_rows.append({'candidate_kegg_id':r.candidate_kegg_id,'opposite_compound':opposite,'rhea_reaction_id':r.rhea_reaction_id,'status':'missing_opposite_structure'})
                continue
            delta=structure_delta(r.smiles,smiles[opposite])
            motif_rows.append({'candidate_kegg_id':r.candidate_kegg_id,'opposite_compound':opposite,'rhea_reaction_id':r.rhea_reaction_id,
                               'ec_class':r.ec_class,'predicted_pathway':r.predicted_l2_map,'predicted_l1':r.predicted_l1,'evidence_tier':r.evidence_tier,'scaffold_status':r.scaffold_status,
                               'ra_knn_disagreement':methods['Weighted-ECFP4-kNN'][r.candidate_kegg_id]!=r.predicted_l2_map,
                               'ra_lr_disagreement':methods['Fingerprint-LR'][r.candidate_kegg_id]!=r.predicted_l2_map,**delta})
    motifs=pd.DataFrame(motif_rows,columns=list(dict.fromkeys(['candidate_kegg_id','opposite_compound','rhea_reaction_id','status','motifs','candidate_center_atom_indices','mcs_atom_fraction']+[k for row in motif_rows for k in row])))
    motifs.to_csv(out/'reaction_center_motifs.csv',index=False)
    if len(motifs) and 'ec_class' in motifs:
        motif_long=motifs.assign(motif=motifs.motifs.fillna('').str.split(';')).explode('motif')
        motif_long=motif_long.loc[motif_long.motif.ne('')]
        motif_summary=motif_long.groupby(['predicted_pathway','ec_class','motif'],dropna=False).agg(
            pair_count=('candidate_kegg_id','size'),candidate_count=('candidate_kegg_id','nunique'),
            unseen_scaffold_pair_count=('scaffold_status',lambda v:int(v.eq('unseen').sum())),
            ra_knn_disagreement_pair_count=('ra_knn_disagreement','sum'),ra_lr_disagreement_pair_count=('ra_lr_disagreement','sum')).reset_index()
        motif_summary.to_csv(out/'reaction_motif_summary.csv',index=False)

    transitions=[]
    for r in motif_rows:
        if 'functional_group_delta' not in r: continue
        for fg,change in json.loads(r['functional_group_delta']).items():
            if change: transitions.append({k:r[k] for k in ['candidate_kegg_id','opposite_compound','rhea_reaction_id','ec_class','predicted_pathway','evidence_tier']}|{'functional_group':fg,'delta_opposite_minus_candidate':change,'interpretation':'approximate pair difference'})
    pd.DataFrame(transitions,columns=['candidate_kegg_id','opposite_compound','rhea_reaction_id','ec_class','predicted_pathway','evidence_tier','functional_group','delta_opposite_minus_candidate']).to_csv(out/'functional_group_transitions.csv',index=False)
    center_map=motifs.dropna(subset=['motifs']).groupby(['candidate_kegg_id','rhea_reaction_id']).motifs.agg(lambda x:joined(set().union(*(vals(v) for v in x)))).to_dict() if len(motifs) else {}
    master['reaction_center']=[center_map.get((r.candidate_kegg_id,r.rhea_reaction_id),'unresolved') for r in master.itertuples()]
    master.to_csv(out/'reaction_substrate_master.csv',index=False)
    # Pathway switching is a predicted label difference between family members;
    # it is not a measured reaction-induced pathway change.
    switches=[]
    for cluster,g in families.groupby('chemical_cluster_id'):
        if g.predicted_l2_map.nunique()<2: continue
        rows=list(g.itertuples())
        for a,b in combinations(rows,2):
            if a.predicted_l2_map==b.predicted_l2_map: continue
            if not (ev.loc[a.kegg_id,'reaction_context'] or ev.loc[b.kegg_id,'reaction_context']): continue
            _,fa=fingerprint(a.smiles);_,fb=fingerprint(b.smiles)
            if fa is None or fb is None: continue
            sim=DataStructs.TanimotoSimilarity(fa,fb)
            if sim<.55: continue
            switches.append({'compound_a':a.kegg_id,'compound_b':b.kegg_id,'pathway_a':a.predicted_l2_map,'pathway_b':b.predicted_l2_map,
                             'cross_l1':parents[a.predicted_l2_map]!=parents[b.predicted_l2_map],'similarity':sim,'chemical_cluster_id':cluster,
                             **structure_delta(a.smiles,b.smiles)})
    pd.DataFrame(switches,columns=list(dict.fromkeys(['compound_a','compound_b','pathway_a','pathway_b','cross_l1','similarity','chemical_cluster_id']+[k for r in switches for k in r]))).to_csv(out/'pathway_switching_pairs.csv',index=False)
    path_table=pd.read_csv(out/'reaction_network_paths.csv'); path_lookup={(r.candidate_kegg_id,r.pathway):r for r in path_table.itertuples()}
    threshold=json.loads((out/'analysis_config.json').read_text())['l2_threshold']
    bridges=[]; edges=[]
    for r in candidates.itertuples():
        if not ev.loc[r.kegg_id,'eligible'] or not ev.loc[r.kegg_id,'reaction_context']: continue
        own=master.loc[master.candidate_kegg_id.eq(r.kegg_id)&master.second_evidence]
        if own.empty: continue
        high=[(getattr(r,f'l2_top{k}_map'),getattr(r,f'l2_top{k}_probability')) for k in range(1,6) if getattr(r,f'l2_top{k}_probability')>=threshold]
        for (a,pa),(b,pb) in combinations(high,2):
            p1=path_lookup.get((r.kegg_id,a));p2=path_lookup.get((r.kegg_id,b))
            if p1 is None or p2 is None: continue
            # Short opposite-side paths to both pathway target sets constitute
            # the prespecified alternative to high betweenness.
            bridges.append({'candidate_kegg_id':r.kegg_id,'candidate_name':r.name,'pathway_a':a,'pathway_b':b,'l1_a':parents[a],'l1_b':parents[b],
                            'cross_l1':parents[a]!=parents[b],'probability_a':pa,'probability_b':pb,'steps_a':p1.steps,'steps_b':p2.steps,
                            'via_candidate_path_length':p1.steps+p2.steps,'rhea_path_a':p1.rhea_path,'rhea_path_b':p2.rhea_path,
                            'chebi_path_a':p1.chebi_path,'chebi_path_b':p2.chebi_path,'ec_numbers':joined(own.ec_number),
                            'uniprot_accessions':joined(set().union(*(vals(v) for v in own.uniprot_accessions))),
                            'evidence_tier':next((t for t in ['A','B','C'] if t in set(own.evidence_tier)),'context-only')})
            for p in [p1,p2]:
                nodes=p.chebi_path.split(';');rids=p.rhea_path.split(';')
                for x,y,rid in zip(nodes,nodes[1:],rids): edges.append({'candidate_kegg_id':r.kegg_id,'source':x,'target':y,'rhea_reaction_id':rid,'target_pathway':p.pathway,'cross_l1':parents[a]!=parents[b]})
    pd.DataFrame(bridges,columns=list(dict.fromkeys(['candidate_kegg_id','candidate_name','pathway_a','pathway_b','cross_l1','evidence_tier']+[k for r in bridges for k in r]))).to_csv(out/'pathway_bridge_candidates.csv',index=False)
    pd.DataFrame(edges,columns=['candidate_kegg_id','source','target','rhea_reaction_id','target_pathway','cross_l1']).drop_duplicates().to_csv(out/'cross_l1_bridge_network.csv',index=False)
    write_json(out/'bridge_pathway_summary.json',{'candidate_pool':len(candidates),'eligible_reaction_context_candidates':int((evidence.eligible&evidence.reaction_context).sum()),
               'bridge_candidates':len({r['candidate_kegg_id'] for r in bridges}),'bridge_pairs':len(bridges),'cross_l1_pairs':sum(r['cross_l1'] for r in bridges),
               'high_score_threshold':threshold,'score_policy':'same pooled validation-derived nomination threshold; not separately calibrated multi-pathway confidence',
               'max_steps_to_each_pathway':3,'centrality_policy':'short paths to two target pathways; no betweenness claim',
               'pathway_family_policy':'distinct KEGG L2 maps; cross-L1 flag distinguishes superclass bridges'})
    return pairs,motifs

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=HERE)
    parser.add_argument('--fetch',action='store_true')
    parser.add_argument('--permutations',type=int,default=1000)
    parser.add_argument('--seed',type=int,default=20260914)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if args.permutations<1000: parser.error('--permutations must be >=1000')
    print('Building source-linked evidence',flush=True)
    c,m,e,maps,parents,t,g,s=build(args.out,args.fetch)
    print('Matched controls, label permutations and method comparisons',flush=True)
    statistics(args.out,c,m,e,maps,parents,t,args.permutations,args.seed)
    print('Structural families, reaction motifs and bridges',flush=True)
    structures_and_bridges(args.out,c,m,e,maps,parents,g,s)
    print('Analysis complete',flush=True)

if __name__=='__main__': main()
