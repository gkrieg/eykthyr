from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from popari import pl, tl
from popari.components import PopariDataset
from popari.io import save_anndata
from popari.model import Popari, load_trained_model
from scipy.sparse import spmatrix

from .embedding import Embedding
from .modified_VelocytoLoom_class import modified_VelocytoLoom
from .util import get_metagene_edges_window, run_all_perturbations


class Eykthyr(modified_VelocytoLoom):

    def __init__(
        self,
        RNA: Optional[List[sc.AnnData]] = None,
        popari: Optional[Popari] = None,
        TF: Optional[List[sc.AnnData]] = None,
        edge_weights: Optional[List[sc.AnnData]] = None,
        perturbed_X: Optional[List[sc.AnnData]] = None,
        names: Optional[List[str]] = ["eykthyr_dataset"],
        cluster_annotation: Sequence[str] = [],
        num_metagenes: int = -1,
        embeddings: Optional[List[Dict[str, Embedding]]] = [],
    ):

        self.RNA = RNA if RNA is not None else []
        self.popari = popari
        self.TF = TF if TF is not None else []
        self.edge_weights = edge_weights if edge_weights is not None else []
        self.perturbed_X = perturbed_X if perturbed_X is not None else []

        self.datasetnames = names
        self.rna_preprocessed = False

        self.cluster_annotation = cluster_annotation
        self.num_metagenes = num_metagenes
        self.embeddings = embeddings

    def preprocess_rna(
        self,
        make_plots: bool = False,
        cluster_annotation: Optional[Sequence[str]] = [],
    ):
        """"""
        for RNA in self.RNA:
            if self.rna_preprocessed:
                print("RNA appears to already be preprocessed, doing nothing")
                return

            sc.pp.filter_genes(RNA, min_cells=5)
            sc.pp.filter_cells(RNA, min_counts=10)
            RNA.layers["raw"] = RNA.X
            sc.pp.normalize_total(RNA, inplace=True, target_sum=10000)
            sc.pp.log1p(RNA)
            if len(cluster_annotation) > 0:
                self.cluster_annotation = cluster_annotation
            if make_plots:
                sc.pp.pca(RNA)
                sc.pp.neighbors(RNA)
                sc.tl.umap(RNA)
                sc.tl.leiden(RNA, key_added="clusters")
                sc.pl.umap(RNA, color=["clusters"] + self.cluster_annotation)
        # subset to the common genes
        common_genes = set(self.RNA[0].var_names)
        for ad in self.RNA:
            common_genes.intersection_update(ad.var_names)

        # Convert common_genes back to a list
        common_genes = list(common_genes)
        self.RNA = [ad[:, common_genes].copy() for ad in self.RNA]
        self.rna_preprocessed = True

    def compute_metagenes(
        self,
        K: int = 16,
        lambda_Sigma_x_inv: float = 1e-4,
        torch_context: dict = dict(device="cuda:0", dtype=torch.float64),
        initial_iterations: int = 10,
        spatial_iterations: int = 200,
    ):
        """"""
        if not self.rna_preprocessed:
            print(
                "RNA appears to not be preprocessed. Please preprocess RNA using Eykthyr.preprocess_rna() or set Eykthyr.rna_preprocessed = True",
            )
            return

        self.num_metagenes = K
        popari_datasets = []
        for RNA, name in zip(self.RNA, self.datasetnames):
            popari_d = PopariDataset(RNA, name)
            popari_d.compute_spatial_neighbors()
            RNA.obs["adjacency_list"] = popari_d.obs["adjacency_list"]
            RNA.obsp["adjacency_matrix"] = popari_d.obsp["adjacency_matrix"]
            if isinstance(RNA.X, spmatrix):
                RNA.X = RNA.X.todense()
            RNA.X = np.asarray(RNA.X)
            altRNA = RNA.copy()
            if "X_diffmap" in altRNA.obsm.keys():
                del altRNA.obsm["X_diffmap"]
            if "X_pca" in altRNA.obsm.keys():
                del altRNA.obsm["X_pca"]
            popari_datasets.append(altRNA)
        print(popari_datasets)
        self.popari = Popari(
            K=K,
            replicate_names=self.datasetnames,
            datasets=popari_datasets,
            lambda_Sigma_x_inv=lambda_Sigma_x_inv,
            torch_context=torch_context,
            initial_context=torch_context,
            verbose=0,
        )

        for iteration in range(initial_iterations):
            self.popari.estimate_parameters(update_spatial_affinities=False)
            self.popari.estimate_weights(use_neighbors=False)

        for iteration in range(spatial_iterations):
            self.popari.estimate_parameters()
            self.popari.estimate_weights()

    def analyze_metagenes(
        self,
        num_leiden_clusters: int = 10,
    ):
        """"""
        if not self.popari:
            print("Popari has not been run. Please run compute_metagenes() first.")
            return

        tl.preprocess_embeddings(self.popari)
        tl.leiden(
            self.popari,
            use_rep="normalized_X",
            target_clusters=num_leiden_clusters,
        )

        for dataset in self.popari.datasets:
            sc.pp.neighbors(
                dataset,
                use_rep="normalized_X",
                key_added="norm_X_neighbors",
            )
            sc.tl.umap(dataset, neighbors_key="norm_X_neighbors")
            sc.pl.umap(dataset, color=["leiden"] + self.cluster_annotation)

    def save_anndata(
        self,
        dirpath: str,
    ):
        """"""
        dirpath = Path(dirpath)
        path_without_extension = dirpath.parent / dirpath.stem
        path_without_extension.mkdir(exist_ok=True)

        for i, RNA in enumerate(self.RNA):
            RNA.uns["datasetname"] = self.datasetnames[i]
            RNA.uns["rna_preprocessed"] = self.rna_preprocessed
            RNA.uns["cluster_annotation"] = self.cluster_annotation
            RNA.uns["num_metagenes"] = self.num_metagenes
            if "adjacency_list" in RNA.obs.columns:
                del RNA.obs["adjacency_list"]
            RNA.write(f"{path_without_extension}/RNA_{i}.h5ad")

        if self.popari and not os.path.isfile(f"{path_without_extension}/popari.h5ad"):
            self.popari.save_results(f"{path_without_extension}/popari.h5ad")

        for i, TF in enumerate(self.TF):
            TF.write(f"{path_without_extension}/TF_{i}.h5ad")

        for i, edge_weight in enumerate(self.edge_weights):
            edge_weight.write(f"{path_without_extension}/edge_weights_{i}.h5ad")

        for i, perturbed_X in enumerate(self.perturbed_X):
            if "adjacency_list" in perturbed_X.obs.columns:
                del perturbed_X.obs["adjacency_list"]
            perturbed_X.write(f"{path_without_extension}/perturbed_X_{i}.h5ad")

    def set_RNA(
        self,
        RNA: List[sc.AnnData],
    ):
        """"""
        self.RNA = RNA

    def set_popari(
        self,
        popari: Popari,
    ):
        """"""
        self.popari = popari

    def set_TF(
        self,
        TF: List[sc.AnnData],
    ):
        """"""
        self.TF = TF

    def set_edge_weights(
        self,
        edge_weights: List[sc.AnnData],
    ):
        """"""
        self.edge_weights = edge_weights

    def set_perturbed_X(
        self,
        perturbed_X: List[sc.AnnData],
    ):
        """"""
        self.perturbed_X = perturbed_X

    # Adjust the rest of the methods similarly

    def compute_TF_activity(
        self,
        peak_tsvs: List[str],
        archr_dataset_names: List[str],
        motif_tsvs: List[str],
    ):
        """Computes TF activity across multiple RNA datasets."""

        if not isinstance(self.RNA, list):
            raise ValueError("self.RNA should be a list of AnnData objects.")

        if not (
            len(self.RNA)
            == len(peak_tsvs)
            == len(archr_dataset_names)
            == len(motif_tsvs)
        ):
            raise ValueError(
                "The lengths of RNA, peak_tsvs, archr_dataset_names, and motif_tsvs must match.",
            )

        self.TF = []

        for i, RNA in enumerate(self.RNA):
            peak_tsv = peak_tsvs[i]
            archr_dataset_name = archr_dataset_names[i]
            motif_tsv = motif_tsvs[i]

            archr_name_len = len(archr_dataset_name) + 1
            tfpeaks = pd.read_csv(peak_tsv, sep=" ")
            # new_col_names = [c[archr_name_len:-2] for c in tfpeaks.columns]
            new_col_names = [f"{c[archr_name_len:-2]}-1" for c in tfpeaks.columns]
            tfpeaks.rename(
                columns={c: new_c for c, new_c in zip(tfpeaks.columns, new_col_names)},
                inplace=True,
            )
            tfpeaks = tfpeaks.T

            tfmotifs = pd.read_csv(motif_tsv, sep=" ")
            tfmotifs.index = range(1, tfmotifs.shape[0] + 1)
            tfmotifs = tfmotifs.rename(
                columns={c: c.split("_")[0] for c in tfmotifs.columns},
            )
            tfmotifs = tfmotifs.astype(int)

            cellmotifs = sc.AnnData(tfpeaks.dot(tfmotifs))
            cellmotifs.layers["raw"] = cellmotifs.X

            sc.pp.normalize_total(cellmotifs, exclude_highly_expressed=True)
            sc.pp.scale(cellmotifs, zero_center=False)

            included = RNA.obs.index[RNA.obs.index.isin(cellmotifs.obs.index)]
            excluded = RNA.obs.index[~RNA.obs.index.isin(cellmotifs.obs.index)]
            cellmotifs = cellmotifs[included, :]
            tf_archr = cellmotifs.to_df()
            for exclude in excluded:
                tf_archr.loc[exclude, :] = tf_archr.mean(axis=0)
            tf_archr = tf_archr.rename(
                columns={c: c.split("_")[0] for c in tf_archr.columns},
            )
            tf_archr = tf_archr.loc[:, ~tf_archr.columns.duplicated()]
            tf_adata = sc.AnnData(tf_archr)
            tf_adata = tf_adata[RNA.obs_names, :]
            tf_adata.obsm["spatial"] = RNA.obsm["spatial"]
            tf_adata.layers["raw"] = tf_adata.X
            tf_adata.X -= tf_adata.X.min(axis=0)
            tf_adata.X /= tf_adata.X.max(axis=0)
            self.TF.append(tf_adata)

    def compute_TF_metagene_weights(
        self,
        num_hops: int = 2,
    ):
        """"""

        if not self.popari or not self.TF:
            print(
                "Popari needs to be run first, please run compute_metagenes().\n"
                "TF activity also needs to be computed by ArchR. Please follow the jupyter notebook for\n"
                "preprocessing ATAC-seq data and then run compute_TF_activity().",
            )
            return

        if len(self.TF) != len(self.popari.datasets):
            print(
                "Number of TF datasets must match the number of datasets in Popari.",
            )
            return

        self.edge_weights = []
        for i, TF, popdata, RNA in zip(
            range(len(self.TF)),
            self.TF,
            self.popari.datasets,
            self.RNA,
        ):
            M_edges = []
            for j in range(self.num_metagenes):
                temp = []
                edges = get_metagene_edges_window(
                    RNA,
                    TF,
                    j,
                    popdata,
                    num_hops=num_hops,
                )
                Madata = sc.AnnData(edges)
                Madata = Madata[popdata.obs_names, :]
                Madata.obsm["spatial"] = popdata.obsm["spatial"]
                temp.append(Madata)
                M_edges.append(temp)
            self.edge_weights.append(M_edges[0][0])
            for k in range(self.num_metagenes):
                self.edge_weights[i].layers[f"M_{k}"] = M_edges[k][0].X

    def run_all_perturbations(
        self,
    ):
        """"""

        if not self.popari:
            print("Popari needs to be run first, please run compute_metagenes().")
            return

        if not self.TF:
            print(
                "TF activity needs to be computed by ArchR. Please follow the jupyter notebook for\n"
                "preprocessing ATAC-seq data and then run compute_TF_activity().",
            )
            return

        if not self.edge_weights:
            print(
                "GRN edge weights need to be computed first, please run compute_TF_metagene_weights().",
            )
            return

        for d in self.popari.datasets:
            d.obs["original_leiden"] = d.obs["leiden"]
        self.perturbed_X = run_all_perturbations(
            self.popari,
            self.TF,
            self.edge_weights,
            K=self.num_metagenes,
            useX=True,
        )
        # self.perturbed_X = [sc.AnnData(
        #     X=popdata.obsm["X"],
        #     obs=popdata.obs,
        #     obsm=popdata.obsm,
        # ) for popdata in self.popari.datasets]


def load_anndata(dirpath: str) -> Eykthyr:
    """Loads Eykthyr datasets from the specified directory."""

    dirpath = Path(dirpath)
    path_without_extension = dirpath.parent / dirpath.stem
    RNA_list = []
    TF_list = []
    edge_weights_list = []
    perturbed_X_list = []
    popari = None

    # Load RNA datasets
    for file in sorted(path_without_extension.glob("RNA_*.h5ad")):
        RNA_list.append(sc.read(file))

    # Load TF datasets
    for file in sorted(path_without_extension.glob("TF_*.h5ad")):
        TF_list.append(sc.read(file))

    # Load edge weights datasets
    for file in sorted(path_without_extension.glob("edge_weights_*.h5ad")):
        edge_weights_list.append(sc.read(file))

    # Load perturbed_X dataset
    for file in sorted(path_without_extension.glob("perturbed_X*.h5ad")):
        perturbed_X_list.append(sc.read(file))

    # Load popari model
    popari_file = path_without_extension / "popari.h5ad"
    if popari_file.is_file():
        popari = load_trained_model(popari_file)
    elif (path_without_extension / "popari").is_dir():
        popari = load_trained_model(popari_file)

    # Create Eykthyr instance
    eykthyr = Eykthyr(
        RNA=RNA_list,
        TF=TF_list,
        edge_weights=edge_weights_list,
        perturbed_X=perturbed_X_list,
        popari=popari,
    )

    # Optionally, restore attributes from the saved RNA dataset
    if RNA_list:
        eykthyr.datasetnames = [R.uns.get("datasetname", "unknown") for R in RNA_list]
        eykthyr.rna_preprocessed = RNA_list[0].uns.get("rna_preprocessed", False)
        eykthyr.cluster_annotation = RNA_list[0].uns.get("cluster_annotation", [])
        eykthyr.num_metagenes = RNA_list[0].uns.get("num_metagenes", -1)

    return eykthyr
