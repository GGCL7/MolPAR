"""Joint ECFP4 / SVD / t-SNE projection of reference and reviewed molecules."""
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from threadpoolctl import threadpool_limits

def run(data,out):
    training=pd.read_csv(data/'training/molecules.csv').assign(source='Reference',support_l1=-1)
    reviewed=pd.read_csv(data/'reactions/reviewed_candidates.csv')
    labels=pd.read_csv(data/'training/pathways.csv')
    mapping=labels[labels.level.eq('L2')].set_index('pathway_id').parent_l1_index.to_dict()
    reviewed=reviewed.assign(source='Reviewed',support_l1=reviewed.predicted_l2_map.map(mapping).astype(int))
    combined=pd.concat([training,reviewed],ignore_index=True)
    generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    matrix=np.zeros((len(combined),2048),dtype=np.uint8)
    for i,s in enumerate(combined.smiles):
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(Chem.MolFromSmiles(s)),matrix[i])
    with threadpool_limits(limits=4):
        reduced=TruncatedSVD(n_components=50,random_state=2026).fit_transform(matrix)
        xy=TSNE(n_components=2,perplexity=35,learning_rate='auto',init='pca',max_iter=1200,random_state=2026).fit_transform(reduced)
    result=combined[['kegg_id','source','support_l1']].copy()
    result[['tsne_1','tsne_2']]=xy
    result.to_csv(out/'chemical_space_coordinates.csv',index=False)
