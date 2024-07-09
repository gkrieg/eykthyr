from typing import Sequence, Union, Optional
from pathlib import Path

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch

from popari.model import Popari, load_trained_model
from popari.components import PopariDataset
from popari.io import save_anndata
from popari import tl, pl

from .util import get_metagene_edges_window, run_all_perturbations



class Eykthyr():
    
    def __init__(self,
    RNA: Optional[sc.AnnData] = None,
    ATAC: Optional[sc.AnnData] = None,
    popari: Optional[Popari] = None,
    TF: Optional[sc.AnnData] = None,
    edge_weights: Optional[sc.AnnData] = None,
    perturbed_X: Optional[sc.AnnData] = None,
    name: str = 'eykthyr_dataset',
    cluster_annotation: Sequence[str] = [],
    num_metagenes: int = -1,
    ):

        self.RNA = RNA
        self.ATAC = ATAC
        self.popari = popari
        self.TF = TF
        self.edge_weights = edge_weights
        self.perturbed_X = perturbed_X

        self.datasetname = name
        self.rna_preprocessed = False

        self.cluster_annotation = cluster_annotation
        self.num_metagenes = num_metagenes

    def preprocess_rna(self,
    make_plots: bool = False,
    cluster_annotation: Optional[Sequence[str]] = [],
    ):
        '''
        '''

        if self.rna_preprocessed == True:
            print('RNA appears to already be preprocessed, doing nothing')
            return

        sc.pp.filter_genes(self.RNA,min_cells=5)
        sc.pp.filter_cells(self.RNA, min_counts=10)
        self.RNA.layers['raw'] = self.RNA.X
        sc.pp.normalize_total(self.RNA, inplace=True, target_sum=10000)
        sc.pp.log1p(self.RNA)
        if make_plots == True:
            sc.pp.pca(self.RNA)
            sc.pp.neighbors(self.RNA)
            sc.tl.umap(self.RNA)
            sc.tl.leiden(self.RNA, key_added="clusters")
            sc.pl.umap(self.RNA, color=["clusters"] + cluster_annotation)
        self.rna_preprocessed = True

    def compute_metagenes(self,
    K: int = 16,
    lambda_Sigma_x_inv: float = 1e-4,
    torch_context: dict = dict(device='cuda:0', dtype=torch.float64),
    initial_iterations: int = 10,
    spatial_iterations: int = 200,

    ):
        '''
        '''
        self.num_metagenes = K
        popari_d = PopariDataset(self.RNA, self.datasetname)
        popari_d.compute_spatial_neighbors()
        save_anndata(f"{self.datasetname}_popari_rna.h5ad", [popari_d])
        self.popari = Popari(
            K=K,
            dataset_path=f"{self.datasetname}_popari_rna.h5ad",
            lambda_Sigma_x_inv=lambda_Sigma_x_inv,
            torch_context=torch_context,
            initial_context=torch_context,
            verbose=1,
        )
        for iteration in range(initial_iterations):
            self.popari.estimate_parameters(update_spatial_affinities=False)
            self.popari.estimate_weights(use_neighbors=False)
        for iteration in range(spatial_iterations):
            self.popari.estimate_parameters()
            self.popari.estimate_weights()

    def analyze_metagenes(self,
    num_leiden_clusters: int = 10,
    ):
        '''
        '''
        tl.preprocess_embeddings(self.popari)
        tl.leiden(self.popari, use_rep="normalized_X", target_clusters=11)
        sc.pp.neighbors(self.popari.datasets[0], use_rep='normalized_X', key_added='norm_X_neighbors')
        sc.tl.umap(self.popari.datasets[0], neighbors_key='norm_X_neighbors')
        sc.pl.umap(self.popari.datasets[0], color=['leiden'] + self.cluster_annotation)

    def save_anndata(self,
    dirpath: str,
    ):
        '''
        '''
        #dirpath gives a directory where each anndata will be saved
        dirpath = Path(dirpath)
        path_without_extension = dirpath.parent / dirpath.stem
        path_without_extension.mkdir(exist_ok=True)
        if self.RNA:
            self.RNA.uns['datasetname'] = self.datasetname
            self.RNA.uns['rna_preprocessed'] = self.rna_preprocessed
            self.RNA.uns['cluster_annotation'] = self.cluster_annotation
            self.RNA.uns['num_metagenes'] = self.num_metagenes
            self.RNA.write(f'{path_without_extension}/RNA.h5ad')
        if self.ATAC:
            self.ATAC.write(f'{path_without_extension}/ATAC.h5ad')
        if self.popari:
            self.popari.save_results(f'{path_without_extension}/popari.h5ad')
        if self.TF:
            self.TF.write(f'{path_without_extension}/TF.h5ad')
        if self.edge_weights:
            self.edge_weights.write(f'{path_without_extension}/edge_weights.h5ad')
        if self.perturbed_X:
            del self.perturbed_X.obs['adjacency_list']
            self.perturbed_X.write(f'{path_without_extension}/perturbed_X.h5ad')

    def set_RNA(self,
    RNA: sc.AnnData
    ):

        '''
        '''
        self.RNA = RNA

    def set_ATAC(self,
    ATAC: sc.AnnData
    ):
        '''
        '''
        self.ATAC = ATAC

    def set_popari(self,
    popari: Popari
    ):
        '''
        '''
        self.popari = popari

    def set_TF(self,
    TF: sc.AnnData
    ):
        '''
        '''
        self.TF = TF

    def set_edge_weights(self,
    edge_weights: sc.AnnData
    ):
        '''
        '''
        self.edge_weights = edge_weights

    def set_perturbed_X(self,
    perturbed_X: sc.AnnData
    ):
        '''
        '''
        self.perturbed_X = perturbed_X

    def compute_TF_activity(self,
    peak_tsv: str,
    archr_dataset_name: str,
    motif_tsv: str
    ):
        '''
        '''
        tfpeaks = pd.read_csv(peak_tsv, sep=' ')
        tfpeaks = tfpeaks.rename(columns={c:f'{c[19:-2]}-1' for c in tfpeaks.columns})
        tfpeaks = tfpeaks.T

        tfmotifs = pd.read_csv(motif_tsv, sep=' ')
        tfmotifs.index = range(1,tfmotifs.shape[0]+1)
        tfmotifs = tfmotifs.rename(columns={c:c.split('_')[0] for c in tfmotifs.columns})
        tfmotifs = tfmotifs.astype(int)

        cellmotifs = sc.AnnData(tfpeaks.dot(tfmotifs))
        cellmotifs.layers['raw'] = cellmotifs.X

        sc.pp.normalize_total(cellmotifs, exclude_highly_expressed=True)
        sc.pp.scale(cellmotifs, zero_center=False)

        included = self.RNA.obs.index[self.RNA.obs.index.isin(cellmotifs.obs.index)]
        excluded = self.RNA.obs.index[~self.RNA.obs.index.isin(cellmotifs.obs.index)]
        cellmotifs = cellmotifs[included,:]
        tf_archr = cellmotifs.to_df()
        for exclude in excluded:
            tf_archr.loc[exclude,:] = tf_archr.mean(axis=0)
        tf_archr = tf_archr.rename(columns={c: c.split('_')[0] for c in tf_archr.columns})
        tf_archr = tf_archr.loc[:,~tf_archr.columns.duplicated()]
        self.TF = sc.AnnData(tf_archr)
        self.TF = self.TF[self.RNA.obs_names,:]
        self.TF.obsm['spatial'] = self.RNA.obsm['spatial']
        self.TF.layers['raw'] = self.TF.X
        self.TF.X -= self.TF.X.min(axis=0)
        self.TF.X /= self.TF.X.max(axis=0)

    def compute_TF_metagene_weights(self,
    num_hops: int = 2,
    ):
        '''

        '''
        #check if popari run, ATAC is imported
        M_edges = []
        df_edges = []
        top_tfs = []
        for j in range(self.num_metagenes):
            temp = []
            temp_tfs = []
            edges = get_metagene_edges_window(self.popari.datasets[0],self.TF,j,self.popari.datasets[0],num_hops=num_hops)
            temp_tfs.append((edges.quantile(0.9,axis=0) - edges.quantile(0.1,axis=0)).sort_values().index[-1])
            df_edges.append(edges)
            Madata = sc.AnnData(edges)
            Madata = Madata[self.popari.datasets[0].obs_names,:]
            Madata.obsm['spatial'] = self.popari.datasets[0].obsm['spatial']
            temp.append(Madata)
            M_edges.append(temp)
            top_tfs.append(temp_tfs)
        self.edge_weights = M_edges[0][0]
        for i in range(self.num_metagenes):
            self.edge_weights.layers[f'M_{i}'] = M_edges[i][0].X

    def run_all_perturbations(self,
    ):
        '''
        '''
        for d in self.popari.datasets:
            d.obs['original_leiden'] = d.obs['leiden']
        run_all_perturbations(self.popari, [self.TF], [self.edge_weights], K=self.num_metagenes, useX=True)
        self.perturbed_X = sc.AnnData(X = self.popari.datasets[0].obsm['X'], obs=self.popari.datasets[0].obs, obsm=self.popari.datasets[0].obsm)
 
def load_anndata(dirpath):
    '''
    '''
    #load all objects, then construct Eykthyr
    dirpath = Path(dirpath)
    path_without_extension = dirpath.parent / dirpath.stem
    RNA = ATAC = TF = edge_weights = perturbed_X = popari = None
    if Path(f"{path_without_extension}/RNA.h5ad").is_file():
        RNA = sc.read(f'{path_without_extension}/RNA.h5ad')
    if Path(f"{path_without_extension}/ATAC.h5ad").is_file():
        ATAC = sc.read(f'{path_without_extension}/ATAC.h5ad')
    if Path(f"{path_without_extension}/TF.h5ad").is_file():
        TF = sc.read(f'{path_without_extension}/TF.h5ad')
    if Path(f"{path_without_extension}/edge_weights.h5ad").is_file():
        edge_weights = sc.read(f'{path_without_extension}/edge_weights.h5ad')
    if Path(f"{path_without_extension}/perturbed_X.h5ad").is_file():
        perturbed_X = sc.read(f'{path_without_extension}/perturbed_X.h5ad')
    if Path(f"{path_without_extension}/popari.h5ad").is_file():
        popari = load_trained_model(f'{path_without_extension}/popari.h5ad')
    elif Path(f"{path_without_extension}/popari").is_dir():
        popari = load_trained_model(f'{path_without_extension}/popari.h5ad')
    eykthyr = Eykthyr(RNA = RNA,
                      ATAC = ATAC,
                      TF = TF,
                      edge_weights = edge_weights,
                      perturbed_X = perturbed_X,
                      popari = popari
                      )
    eykthyr.datasetname = eykthyr.RNA.uns['datasetname']
    eykthyr.rna_preprocessed = eykthyr.RNA.uns['rna_preprocessed']
    eykthyr.cluster_annotation = eykthyr.RNA.uns['cluster_annotation']
    eykthyr.num_metagenes = eykthyr.RNA.uns['num_metagenes']
    return eykthyr

