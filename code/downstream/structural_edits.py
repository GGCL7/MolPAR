"""Chemically sanitized structural edits used in the study."""
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
def joined(values): return ";".join(sorted(set(values)))

def sanitized(mol):
    try:
        Chem.SanitizeMol(mol)
        if len(Chem.GetMolFrags(mol))!=1 or mol.GetNumHeavyAtoms()<3 or any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()): return None
        return Chem.MolToSmiles(mol,isomericSmiles=True)
    except (ValueError,RuntimeError): return None

def edits(smiles, center):
    mol=Chem.MolFromSmiles(smiles); output=[]; seen={Chem.MolToSmiles(mol)}
    def add(edit, rw, atoms):
        s=sanitized(rw.GetMol())
        if s is None or s in seen: return
        seen.add(s)
        output.append({'edit_type':edit,'edited_smiles':s,'edited_atom_indices':joined(map(str,atoms)),
                       'targets_approximate_center':bool(set(atoms)&center),
                       'scaffold_preserved':MurckoScaffold.MurckoScaffoldSmiles(smiles=s)==MurckoScaffold.MurckoScaffoldSmiles(mol=mol)})
    # Atom-local edits; atom indices always refer to the original canonical SMILES.
    for match in mol.GetSubstructMatches(Chem.MolFromSmarts('[CX4;H1,H2]-[OX2H]')):
        c,o=match
        rw=Chem.RWMol(mol);rw.GetAtomWithIdx(c).SetNumExplicitHs(0);rw.GetAtomWithIdx(c).SetNoImplicit(False);rw.GetAtomWithIdx(c).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED);rw.RemoveAtom(o);add('remove_hydroxyl',rw,match)
        rw=Chem.RWMol(mol);rw.GetBondBetweenAtoms(c,o).SetBondType(Chem.BondType.DOUBLE)
        for i in match: rw.GetAtomWithIdx(i).SetNumExplicitHs(0);rw.GetAtomWithIdx(i).SetNoImplicit(False);rw.GetAtomWithIdx(i).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        add('hydroxyl_to_carbonyl',rw,match)
    for match in mol.GetSubstructMatches(Chem.MolFromSmarts('[CX3;!$(C(=O)[O,N,S])]=[OX1]')):
        rw=Chem.RWMol(mol);rw.GetBondBetweenAtoms(*match).SetBondType(Chem.BondType.SINGLE)
        for i in match: rw.GetAtomWithIdx(i).SetNumExplicitHs(0);rw.GetAtomWithIdx(i).SetNoImplicit(False)
        add('carbonyl_to_hydroxyl',rw,match)
    for a in mol.GetAtoms():
        if a.GetAtomicNum()==6 and not a.GetIsAromatic() and a.GetDegree()==1 and a.GetTotalNumHs()==3:
            neighbor=a.GetNeighbors()[0]
            if neighbor.GetAtomicNum()!=6: continue
            rw=Chem.RWMol(mol);rw.GetAtomWithIdx(neighbor.GetIdx()).SetNumExplicitHs(0);rw.GetAtomWithIdx(neighbor.GetIdx()).SetNoImplicit(False);rw.GetAtomWithIdx(neighbor.GetIdx()).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED);rw.RemoveAtom(a.GetIdx());add('shorten_terminal_carbon',rw,[a.GetIdx()])
            rw=Chem.RWMol(mol);new=rw.AddAtom(Chem.Atom(6));rw.AddBond(a.GetIdx(),new,Chem.BondType.SINGLE)
            rw.GetAtomWithIdx(a.GetIdx()).SetNumExplicitHs(0);rw.GetAtomWithIdx(a.GetIdx()).SetNoImplicit(False)
            add('extend_terminal_carbon',rw,[a.GetIdx()])
    def cleave(edit, a, b, retain_atom, affected):
        rw=Chem.RWMol(mol);rw.RemoveBond(a,b)
        fragments=Chem.GetMolFrags(rw.GetMol(),asMols=False,sanitizeFrags=False)
        keep=next((set(f) for f in fragments if retain_atom in f),set())
        if len(keep)==mol.GetNumAtoms(): return
        removed=set(range(mol.GetNumAtoms()))-keep
        if edit=='remove_glycosyl':
            # Delete a terminal sugar, never discard the aglycone and call
            # the retained sugar a deglycosylated counterfactual.
            if len(keep)<len(removed) or sum(mol.GetAtomWithIdx(i).GetAtomicNum()==6 for i in removed)>6:return
            if any(mol.GetAtomWithIdx(i).GetAtomicNum() not in [6,8] for i in removed):return
        for endpoint in [a,b]:
            if endpoint in keep:
                rw.GetAtomWithIdx(endpoint).SetNumExplicitHs(0);rw.GetAtomWithIdx(endpoint).SetNoImplicit(False)
        for atom in sorted(removed,reverse=True): rw.RemoveAtom(atom)
        add(edit,rw,affected)
    for match in mol.GetSubstructMatches(Chem.MolFromSmarts('[C]-[O]-[P](=O)(O)O')):
        c,o,p=match[:3];cleave('remove_phosphate',o,p,c,match)
    # Require a sugar-like saturated O-containing ring with >=2 exocyclic OH/O.
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) not in [5,6]: continue
        atoms=[mol.GetAtomWithIdx(i) for i in ring]
        if sum(a.GetAtomicNum()==8 for a in atoms)!=1 or any(a.GetIsAromatic() for a in atoms): continue
        oxy=[(a.GetIdx(),b.GetIdx()) for a in atoms for b in a.GetNeighbors() if b.GetIdx() not in ring and b.GetAtomicNum()==8]
        if len(oxy)<2: continue
        for c,o in oxy:
            for other in mol.GetAtomWithIdx(o).GetNeighbors():
                if other.GetIdx()==c or other.GetIdx() in ring: continue
                cleave('remove_glycosyl',c,o,other.GetIdx(),list(ring)+[o])
    return output

from functools import lru_cache
import json
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator, rdFMCS
FG = {'hydroxyl':'[OX2H][CX4]', 'carbonyl':'[CX3]=[OX1]', 'phosphate':'[P](=[O])([O])[O]',
      'carboxyl':'[CX3](=O)[O;H1,-1]', 'amine':'[NX3;H1,H2;!$(N-C=O)]',
      'glycosyl':'[C;R]1[O;R][C;R][C;R][C;R][C;R]1', 'acyl':'[CX3](=O)[O,N,S][C]'}
FP = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
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
