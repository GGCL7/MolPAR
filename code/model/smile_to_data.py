"""
smile_to_data.py — SMILES → PyG Data 转换

与 model_fixed/smile_to_data.py 完全一致。
原子特征 51 维, 键特征 14 维。
"""

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Mol, AllChem
from torch_geometric.data import Data
from typing import List

# ---------------------------------------------------------------------------
# 原子/键特征白名单
# ---------------------------------------------------------------------------
atom_types = ['H', 'C', 'B', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'I']
formal_charges = [-2, -1, 0, 1, 2]
degree = [0, 1, 2, 3, 4, 5, 6]
num_hs = [0, 1, 2, 3, 4]
hybridization = [
    Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
    Chem.rdchem.HybridizationType.UNSPECIFIED,
]
chiral_tags = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]
bond_types = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
bond_stereos = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOTRANS,
]


def one_hot_embedding(value, options: List) -> List[int]:
    """One-hot encoding with an extra 'unknown' slot."""
    embedding = [0] * (len(options) + 1)
    try:
        index = options.index(value)
    except ValueError:
        index = -1
    embedding[index] = 1
    return embedding


# ---------------------------------------------------------------------------
# H-bond donor / acceptor SMARTS
# ---------------------------------------------------------------------------
_HBD_SMARTS = Chem.MolFromSmarts("[$([N;!H0;v3]),$([N;!H0;+1;v4]),$([O,S;H1;+0]),$([n;H1;+0])]")
_HBA_SMARTS = Chem.MolFromSmarts("[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=!@[O,N,P,S])]),$([nH0,o,s;+0])]")


def _atom_hbd_hba_flags(mol: Mol):
    hbd = set()
    hba = set()
    if _HBD_SMARTS is not None:
        for m in mol.GetSubstructMatches(_HBD_SMARTS):
            for a in m:
                hbd.add(a)
    if _HBA_SMARTS is not None:
        for m in mol.GetSubstructMatches(_HBA_SMARTS):
            for a in m:
                hba.add(a)
    return hbd, hba


def get_node_features(mol: Mol) -> np.ndarray:
    """
    Per-atom feature vector (51 dim).
    """
    atoms = mol.GetAtoms()
    hbd, hba = _atom_hbd_hba_flags(mol)

    features_list = []
    for atom in atoms:
        idx = atom.GetIdx()
        f = []
        f += one_hot_embedding(atom.GetSymbol(), atom_types)
        f += one_hot_embedding(atom.GetTotalDegree(), degree)
        f += one_hot_embedding(atom.GetFormalCharge(), formal_charges)
        f += one_hot_embedding(atom.GetTotalNumHs(), num_hs)
        f += one_hot_embedding(atom.GetHybridization(), hybridization)
        f += one_hot_embedding(atom.GetChiralTag(), chiral_tags)
        f += [int(atom.GetIsAromatic())]
        f += [int(atom.IsInRing())]
        f += [int(idx in hbd)]
        f += [int(idx in hba)]
        f += [atom.GetMass() * 0.01]
        features_list.append(f)

    return np.array(features_list, dtype=np.float32)


def get_edge_features_and_index(mol: Mol):
    """
    Build undirected edge_index + edge_attr (14 dim).
    """
    edge_index_src = []
    edge_index_dst = []
    edge_feats = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        f = []
        f += one_hot_embedding(str(bond.GetBondType()), bond_types)
        f += [int(bond.GetIsConjugated())]
        f += [int(bond.IsInRing())]
        f += one_hot_embedding(bond.GetStereo(), bond_stereos)

        edge_index_src += [i, j]
        edge_index_dst += [j, i]
        edge_feats.append(f)
        edge_feats.append(f)

    if len(edge_feats) == 0:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros(
            (0, (len(bond_types) + 1) + 2 + (len(bond_stereos) + 1)),
            dtype=np.float32,
        )
        return edge_index, edge_attr

    edge_index = np.array([edge_index_src, edge_index_dst], dtype=np.int64)
    edge_attr = np.array(edge_feats, dtype=np.float32)
    return edge_index, edge_attr


def mol_to_data(mol: Mol) -> Data:
    node_features = get_node_features(mol)
    edge_index, edge_attr = get_edge_features_and_index(mol)

    data = Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
    )
    return data


# ---------------------------------------------------------------------------
# SMILES → Mol
# ---------------------------------------------------------------------------
def smiles_to_mol(smiles: str) -> Mol:
    """Parse SMILES -> 2D RDKit Mol."""
    return Chem.MolFromSmiles(smiles)


def smiles_to_3d_mol_real(smiles: str, max_iters: int = 200) -> Mol:
    """Embed a 3D conformer via ETKDGv3 + MMFF optimization."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    molH = Chem.AddHs(mol)
    try:
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        cid = AllChem.EmbedMolecule(molH, params)
        if cid == -1:
            return mol
        AllChem.MMFFOptimizeMolecule(molH, maxIters=max_iters)
        return molH
    except Exception:
        return mol


# Backward compat alias
smiles_to_3d_mol = smiles_to_mol
