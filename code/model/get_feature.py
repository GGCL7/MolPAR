"""
get_feature.py — 分子特征提取

与 model_fixed/get_feature.py 完全一致:
- smiles_to_data: SMILES → PyG Data (2D molecular graph)
- extract_fingerprints: SMILES → 拼接指纹 (MACCS + ErG + PubChem + ECFP4)
"""

from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
import numpy as np
import torch

from smile_to_data import smiles_to_mol, mol_to_data
from pubchemfp import GetPubChemFPs


def smiles_to_data(smiles: str):
    """SMILES -> PyG Data(x, edge_index, edge_attr) on 2D molecular graph."""
    mol = smiles_to_mol(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return mol_to_data(mol)


def extract_fingerprints(smiles: str) -> torch.Tensor:
    """Concatenate MACCS + ErG + PubChem + ECFP4 -> single float tensor."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    fp_maccs = np.array(AllChem.GetMACCSKeysFingerprint(mol), dtype=int)
    fp_phaErGfp = np.array(
        AllChem.GetErGFingerprint(mol, fuzzIncrement=0.3, maxPath=21, minPath=1),
        dtype=int,
    )
    fp_pubcfp = np.array(GetPubChemFPs(mol), dtype=int)
    fp_ecfp2 = np.array(
        AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024),
        dtype=int,
    )

    fp_all = np.concatenate([fp_maccs, fp_phaErGfp, fp_pubcfp, fp_ecfp2])
    return torch.tensor(fp_all, dtype=torch.float32)
