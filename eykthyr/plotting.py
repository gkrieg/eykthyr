from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import scanpy as sc

from .embedding import Embedding
from .eykthyr import Eykthyr


def paga_spatial_simulation(
    eykthyr: Eykthyr,
    TFs: Sequence[str],
    cluster_name: str,
):
    """"""
    umap_spatial_simulation(
        eykthyr,
        TFs,
        [eykthyr.perturbed_X],
        cluster_name=cluster_name,
    )


def prep_paga(
    eykthyr: Eykthyr,
    groups: str,
):
    sc.pp.neighbors(eykthyr.perturbed_X, use_rep="normalized_X")
    sc.tl.umap(eykthyr.perturbed_X)
    sc.tl.paga(eykthyr.perturbed_X, groups=groups)
    sc.tl.draw_graph(eykthyr.perturbed_X, init_pos="X_umap", random_state=123)
    sc.pl.draw_graph(eykthyr.perturbed_X, color=groups, legend_loc="on data")


def umap_spatial_simulation(
    eykthyr,
    TFs,
    datasets,
    n_grid=40,
    min_masses=[0.27, 0.007],
    scales=[30, 1.2],
    embeddings=["spatial", "X_draw_graph_fr"],
    n_neighbors=[20, 25],
    cluster_name="original_leiden",
    show_plots=[True, True],
):

    eykthyr.RNA.X = eykthyr.RNA.X.astype("double")

    for TF in TFs:
        for embedding in embeddings:
            eykthyr.embeddings[embedding] = Embedding()
        eykthyr.estimate_transition_probs(
            embedding_names=embeddings,
            tf_name=TF,
            n_neighbors=n_neighbors,
            sampled_fraction=1,
        )
        eykthyr.calculate_embedding_shifts(embedding_names=embeddings, sigma_corr=0.05)
        eykthyr.calculate_p_mass(
            embeddings,
            smooth=0.8,
            n_grid=n_grid,
            n_neighbors=n_neighbors,
        )
        eykthyr.calculate_mass_filter(embeddings, min_mass=min_masses, plot=False)
        # eykthyr.suggest_mass_thresholds(n_suggestion=12)
        for embedding, show_plot, scale in zip(embeddings, show_plots, scales):
            if show_plot == True:
                fig, ax = plt.subplots(1, 2, figsize=[13, 6])

                eykthyr.plot_simulation_flow_on_grid(embedding, scale=scale, ax=ax[0])
                ax[0].set_title(f"Simulated cell identity shift vector: {TF} KO")

                # Show quiver plot that was calculated with randomized graph.
                eykthyr.plot_simulation_flow_random_on_grid(
                    embedding,
                    scale=scale,
                    ax=ax[1],
                )
                ax[1].set_title(f"Randomized simulation vector")

                plt.show()

        fig2, ax2 = plt.subplots(1, len(embeddings), figsize=[16, 6])

        for i, embedding in enumerate(embeddings):

            eykthyr.plot_cluster_whole(embedding, cluster_name, ax=ax2[i], s=5)
            eykthyr.plot_simulation_flow_on_grid(
                embedding,
                scale=scales[i],
                ax=ax2[i],
                show_background=False,
            )
            ax2[i].set_title(f"Simulated cell identity shift {embedding}: {TF} KO")

        plt.show()


"""
def old_umap_spatial_simulation(TFs,
    datasets,
    n_grid=40,
    min_masses=[.27, 0.007],
    scales=[30,1.2],
    embeddings=['spatial','X_draw_graph_fa'],
    n_neighbors=[20,25],
    cluster_name='original_leiden',
    show_plots=[True, True]
):

    for TF in TFs:
        oracless = []
        for embedding, n in zip(embeddings, n_neighbors):
            oracles = []
            for d in datasets:

                oracle = co.Oracle()
                oracle.import_anndata_as_raw_count(adata=d,
                                   cluster_column_name=cluster_name,
                                   embedding_name=embedding)
                oracles.append(oracle)
            oracless.append(oracles)


            for oracle in oracles:
                oracle.adata.X = oracle.adata.X.astype('double')
                oracle.adata.layers['imputed_count'] = oracle.adata.obsm[f'normalized_X_{TF}_dropout'].astype('double')
                oracle.adata.layers['delta_X'] =  oracle.adata.obsm[
                                        f'normalized_X_{TF}_dropout'] - oracle.adata.obsm['normalized_X']
                oracle.estimate_transition_prob(
                                n_neighbors=n,
                                sampled_fraction=1
                                )
                oracle.calculate_embedding_shift(sigma_corr=0.05)



        for i in range(1):
            for j,oracles in enumerate(oracless):


                goi = TF
                # Show quiver plot

                oracles[i].calculate_p_mass(smooth=0.8, n_grid=n_grid, n_neighbors=n_neighbors[j])
#                 oracles[i].suggest_mass_thresholds(n_suggestion=12)
                oracles[i].calculate_mass_filter(min_mass=min_masses[j], plot=False)
                if show_plots[j] == True:
                    fig, ax = plt.subplots(1, 2,  figsize=[13, 6])

                    oracles[i].plot_simulation_flow_on_grid(scale=scales[j], ax=ax[0])
                    ax[0].set_title(f"Simulated cell identity shift vector: {goi} KO")

                    # Show quiver plot that was calculated with randomized graph.
                    oracles[i].plot_simulation_flow_random_on_grid(scale=scales[j], ax=ax[1])
                    ax[1].set_title(f"Randomized simulation vector")

                    plt.show()

        # Plot vector field with cell cluster
            fig2, ax2 = plt.subplots(1,len(embeddings),figsize=[16, 6])
            for oracles, embedding, j in zip(oracless, embeddings, range(len(oracless))):

                oracles[i].plot_cluster_whole(ax=ax2[j], s=5)
                oracles[i].plot_simulation_flow_on_grid(scale=scales[j], ax=ax2[j], show_background=False)
                ax2[j].set_title(f"Simulated cell identity shift {embedding}: {goi} KO")

            plt.show()

def development_simulation():

    from celloracle.applications import Gradient_calculator
    from celloracle.applications import Oracle_development_module


    return
"""
