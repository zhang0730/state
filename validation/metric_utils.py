"""Utility functions for computing metrics."""

import os
import scipy
import torch
import anndata as ad
import numpy as np
import scanpy as sc
import pandas as pd

from typing import Optional
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

from models.base import DecoderInterface


def to_dense(X):
    if scipy.sparse.issparse(X):
        return np.asarray(X.todense())
    else:
        return X


def compute_jaccard(pred, true, ctrl, pred_ctrl):
    """Computes Jaccard score between x and y."""
    set1 = set(pred)
    set2 = set(true)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0


def compute_wasserstein(pred, true, control, pred_ctrl, epsilon=0.1):
    """Computes transport between x and y via Sinkhorn algorithm."""
    # Compute cost
    # should be preprocessed outside of this function
    # pred = pred.detach().cpu().numpy()
    # true = true.detach().cpu().numpy()

    geom_pred_true = pointcloud.PointCloud(pred, true, epsilon=epsilon)
    ot_prob = linear_problem.LinearProblem(geom_pred_true)

    # Solve ot problem
    solver = sinkhorn.Sinkhorn()
    out_pred_true = solver(ot_prob)

    # Return regularized ot cost
    # converting to float bc returning as np array
    return float(out_pred_true.reg_ot_cost)


def mmd_distance(x, y, gamma):
    xx = rbf_kernel(x, x, gamma)
    xy = rbf_kernel(x, y, gamma)
    yy = rbf_kernel(y, y, gamma)

    return xx.mean() + yy.mean() - 2 * xy.mean()


def compute_mmd(pred, true, ctrl, pred_ctrl, gammas=None):
    """Computes MMD between x and y using RBF kernel with multiple gammas."""
    if gammas is None:
        gammas = [2, 1, 0.5, 0.1, 0.01, 0.005]

    def safe_mmd(*args):
        try:
            mmd = mmd_distance(*args)
        except ValueError:
            mmd = np.nan
        return mmd

    return np.mean(list(map(lambda gamma: safe_mmd(pred, true, gamma), gammas)))


def compute_mse(pred, true, ctrl, pred_ctrl):
    """Computes mean squared error between x and y."""
    return mean_squared_error(pred, true)


def compute_pearson(pred, true, ctrl, pred_ctrl):
    """Computes average pairwise Pearson correlation between batches x and y."""
    return np.mean([np.corrcoef(pred[i], true[j])[0, 1] for i in range(len(pred)) for j in range(len(true)) if i == j])


def compute_pearson_delta(pred, true, ctrl, pred_ctrl):
    """Computes Pearson correlation between pred and true after subtracting control."""
    return pearsonr(pred.mean(0) - ctrl.mean(0), true.mean(0) - ctrl.mean(0))[0]


def compute_pearson_delta_separate_controls(pred, true, ctrl, pred_ctrl):
    """
    Computes Pearson correlation between pred and true after subtracting pred and true controls
    If we have learned a function f(control cells, pert) -> pred, then:
        pred = f(ctrl, pert)
        pred_ctrl = f(ctrl, ctrl_pert)
    true are the real perturbed cells and ctrl are the real control cells
    """
    return pearsonr(pred.mean(0) - pred_ctrl.mean(0), true.mean(0) - ctrl.mean(0))[0]


def compute_pearson_delta_batched(batched_means, weightings):
    """Computes Pearson correlation between pred and true after subtracting control."""
    ## TODO should we use the weightings here

    pred_de = pd.DataFrame(batched_means["pert_pred"] - batched_means["ctrl_pred"])
    pred_de = pred_de.dropna().mean(0)

    true_de = pd.DataFrame(batched_means["pert_true"] - batched_means["ctrl_true"])
    true_de = true_de.dropna().mean(0)

    return pearsonr(pred_de, true_de)[0]


def compute_cosine_similarity(pred, true, ctrl, pred_ctrl):
    """Computes cosine similarity between x and y."""

    centroid_true = np.mean(true, axis=0, keepdims=True)
    cos_sim_scores = []
    for i in range(pred.shape[0]):
        pred_i = pred[i]
        true = centroid_true.flatten()
        cos_sim_calc = np.dot(pred_i, true) / (np.linalg.norm(pred_i) * np.linalg.norm(true))
        cos_sim_scores.append(cos_sim_calc)

    # return cos_sim_scores, max(cos_sim_scores)
    return np.mean(cos_sim_scores)


def compute_cosine_similarity_v2(pred, true, ctrl, pred_ctrl):
    pred_mean = pred.mean(0)
    true_mean = true.mean(0)
    cosine_similarity = np.dot(pred_mean, true_mean) / (np.linalg.norm(pred_mean) * np.linalg.norm(true_mean))
    return cosine_similarity.item()


def get_top_k_de(subset_adata, k):
    return [x[0] for x in subset_adata.uns["rank_genes_groups"]["names"][:k]]


def jaccard_score(x, y):
    set1 = set(x)
    set2 = set(y)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0


def compute_gene_overlap_pert(pert_pred, pert_true, ctrl_true, ctrl_pred, gene_names=None, k=50):
    # TODO this is too slow

    pert_true = sc.AnnData(pert_true)
    pert_true.obs["condition"] = "pert"
    ctrl_true = sc.AnnData(ctrl_true)
    ctrl_true.obs["condition"] = "ctrl"
    adata_true = sc.concat([pert_true, ctrl_true])
    adata_true.obs = adata_true.obs.reset_index()
    sc.tl.rank_genes_groups(adata_true, groupby="condition")
    true_de = get_top_k_de(adata_true, k)

    pert_pred = sc.AnnData(pert_pred)
    pert_pred.obs["condition"] = "pert"
    ctrl_pred = sc.AnnData(ctrl_pred)
    ctrl_pred.obs["condition"] = "ctrl"
    adata_pred = sc.concat([pert_pred, ctrl_pred])
    adata_pred.obs = adata_pred.obs.reset_index()
    sc.tl.rank_genes_groups(adata_pred, groupby="condition")
    pred_de = get_top_k_de(adata_pred, k)

    overlap = len(set(true_de).intersection(set(pred_de))) / k

    return overlap


def compute_gene_overlap_cross_pert(DE_true, DE_pred, control_pert="non-targeting", k=50):
    all_overlaps = {}
    k = max(k, len(DE_true.columns))
    for c in DE_pred.index:
        if c == control_pert:
            continue
        all_overlaps[c] = len(set(DE_true.loc[c].values).intersection(set(DE_pred.loc[c].values))) / k

    print("Average DE: ", np.mean(list(all_overlaps.values())))
    return all_overlaps


def compute_DE_for_truth_and_pred(
    adata_real_ct,  # ground truth in gene space (or latent if output_space=latent but has real_exp?)
    adata_pred_ct,  # predicted data (latent or gene)
    control_pert: str,
    pert_col: str = "gene",
    n_top_genes: int = 2000,  # HVG cutoff
    k_de_genes: int = 50,
    output_space: str = "gene",
    model_decoder: Optional[DecoderInterface] = None,
):
    """
    Unify logic for computing DE from both the ground truth and model predictions.

    Steps:
        1) If adata_pred_ct is in latent space (output_space != 'gene'), but we do have a 'model_decoder',
           call model_decoder.decode_to_adata(adata_pred_ct). -> returns an adata_pred_gene in gene space.

           Otherwise, if output_space='gene', we already have gene space in .X.

        2) Subset ground truth and predictions to the same gene set, then do HVG filtering (n_top_genes).
           We'll unify them by intersection, or the predicted data might already have the same features.

        3) For each: run rank_genes_groups to get top k DE genes vs. control_pert. Return the final
           lists as a DataFrame: row=pert, columns=ranked DE genes in descending order.

    Returns:
        (DE_true, DE_pred) as two data frames (index=pert_name, columns=top_k DE genes).
    """

    adata_real_ct = adata_real_ct.copy()
    adata_pred_ct = adata_pred_ct.copy()

    # 1) Normalize the ground truth predictions if they are in gene space
    using_gene_space = (output_space == "gene") or (model_decoder is not None)
    if using_gene_space:
        sc.pp.normalize_total(adata_real_ct)
        sc.pp.log1p(adata_real_ct)

    # 2) HVG filtering (applied to each or to the combined data).
    # This happens to the ground truth regardless of input space.
    sc.pp.highly_variable_genes(adata_real_ct, n_top_genes=n_top_genes)
    hvg_mask = adata_real_ct.var["highly_variable"].values
    adata_real_hvg = adata_real_ct[:, hvg_mask].copy()
    DE_true = _compute_topk_DE(adata_real_hvg, control_pert, pert_col, k_de_genes)

    # 3) Possibly decode predictions if in latent space
    if output_space == "latent" and model_decoder is not None:
        # decode to gene space
        DE_pred = model_decoder.compute_de_genes(
            adata_pred_ct,
            pert_col=pert_col,
            control_pert=control_pert,
            genes=adata_real_hvg.var.index.values,
        )
    else:
        # assume adata_pred_ct is already in gene space
        adata_pred_gene = adata_pred_ct
        if output_space == "gene":
            sc.pp.normalize_total(adata_pred_gene)
            sc.pp.log1p(adata_pred_gene)
        adata_pred_gene.obs.index = adata_pred_gene.obs.index.astype(str)
        adata_pred_hvg = adata_pred_gene[:, hvg_mask].copy()
        DE_pred = _compute_topk_DE(adata_pred_hvg, control_pert, pert_col, k_de_genes)

    return DE_true, DE_pred


def _compute_topk_DE(adata_gene, control_pert, pert_col, k):
    """
    Convenience: runs rank_genes_groups (with standard log1p if needed),
    returns a DataFrame: row=pert_name, columns=top genes in descending order
    """

    # rank Genes
    sc.tl.rank_genes_groups(
        adata_gene,
        groupby=pert_col,
        reference=control_pert,
        rankby_abs=True,
        n_genes=k,
        method="wilcoxon",
    )

    # Extract results to DataFrame
    de_genes = pd.DataFrame(adata_gene.uns["rank_genes_groups"]["names"])
    # transpose so each row=pert, columns=the top K genes
    return de_genes.T


def compute_DE(adata, pert_col="gene", control_pert="non-targeting", k=50):
    """
    Compute DE in gene space.
    """

    sc.tl.rank_genes_groups(adata, groupby=pert_col, reference=control_pert, rankby_abs=True, n_genes=k)
    de_genes = pd.DataFrame(adata.uns["rank_genes_groups"]["names"])

    return de_genes.T


def compute_DE_pca(adata_pred, gene_names, pert_col, control_pert, k=50, transform=None):
    """
    Compute differential expression in gene space after decoding from PCA.
    """
    if transform is None:
        raise ValueError("PCA transform required for compute_DE_pca")

    # Decode predictions back to gene space
    decoded_pred = transform.decode(adata_pred.X)

    # Create new anndata with decoded predictions
    decoded_adata = ad.AnnData(X=decoded_pred.cpu().numpy(), obs=adata_pred.obs, var=pd.DataFrame(index=gene_names))

    # Compute DE using scanpy
    sc.tl.rank_genes_groups(decoded_adata, groupby=pert_col, reference=control_pert, rankby_abs=True, n_genes=k)

    # Extract results
    de_genes = pd.DataFrame(decoded_adata.uns["rank_genes_groups"]["names"])
    return de_genes.T


def compute_mean_perturbation_effect(adata, pert_col="gene", ctrl_pert="non-targeting"):
    adata_df = adata.to_df()
    adata_df["pert"] = adata.obs[pert_col].values
    mean_df = adata_df.groupby("pert").mean()
    mean_pert_effect = np.abs(mean_df - mean_df.loc[ctrl_pert])
    return mean_pert_effect


def compute_perturbation_id_score(adata_pred, adata_real, pert_col="gene", ctrl_pert="non-targeting"):
    ## Given a specific perturbation, identify which model-predicted
    # perturbation is most similar to it

    ## Compute true mean perturbation effect
    mean_real_effect = compute_mean_perturbation_effect(adata_real, pert_col, ctrl_pert)
    mean_pred_effect = compute_mean_perturbation_effect(adata_pred, pert_col, ctrl_pert)
    all_perts = mean_real_effect.index.values
    ## For each true perturbation effect, find the nearest neighbor predicted
    # perturbation effect

    pred_perts = []
    for pert in all_perts:
        real_effect = mean_real_effect.loc[pert].values
        pred_effects = mean_pred_effect.values
        pred_pert = all_perts[np.argmax(cosine_similarity(real_effect.reshape(1, -1), pred_effects))]
        pred_perts.append(pred_pert)

    accuracy_score = np.sum(pred_perts == all_perts) / len(all_perts)

    return accuracy_score


def compute_perturbation_ranking_score(adata_pred, adata_real, pert_col="gene", ctrl_pert="non-targeting"):
    ## Compute true mean perturbation effect
    mean_real_effect = compute_mean_perturbation_effect(adata_real, pert_col, ctrl_pert)
    mean_pred_effect = compute_mean_perturbation_effect(adata_pred, pert_col, ctrl_pert)
    all_perts = mean_real_effect.index.values

    ranks = []

    ## For each true perturbation effect, compute similarity to all predicted effects
    for pert in all_perts:
        real_effect = mean_real_effect.loc[pert].values.reshape(1, -1)
        pred_effects = mean_pred_effect.values

        # Compute cosine similarities between the real effect and all predicted effects
        similarities = cosine_similarity(real_effect, pred_effects).flatten()

        # Get the rank of the true perturbation based on similarity
        true_pert_index = np.where(all_perts == pert)[0][0]
        sorted_indices = np.argsort(similarities)[::-1]  # Sort in descending order of similarity
        rank_of_true_pert = np.where(sorted_indices == true_pert_index)[0][0]  # 1-based rank

        ranks.append(rank_of_true_pert)

    ## mean normalized rank
    mean_rank = np.mean(ranks) / len(all_perts)

    return mean_rank


def compute_GI_score(pred_adata, true_adata, pert_col="gene", celltype_col="cell_type"):
    """Compute the Spearman correlation between the
    predicted and true GI scores"""
    pass


def get_GI_params(preds, combo):
    """
    Get GI parameters

    Args:
        preds (dict): dictionary of predictions
        combo (list): list of perturbations

    """
    singles_expr = np.array([preds[combo[0]], preds[combo[1]]]).T
    first_expr = np.array(preds[combo[0]]).T
    second_expr = np.array(preds[combo[1]]).T
    double_expr = np.array(preds[combo[0] + "_" + combo[1]]).T

    return get_GI_coeffs(singles_expr, first_expr, second_expr, double_expr)


def get_GI_coeffs(singles_expr, first_expr, second_expr, double_expr):
    """
    Get coefficients for GI calculation

    Args:
        singles_expr (np.array): single perturbation expression
        first_expr (np.array): first perturbation expression
        second_expr (np.array): second perturbation expression
        double_expr (np.array): double perturbation expression

    """
    results = {}
    results["ts"] = TheilSenRegressor(fit_intercept=False, max_subpopulation=1e5, max_iter=1000, random_state=1000)
    X = singles_expr
    y = double_expr
    results["ts"].fit(X, y.ravel())
    Zts = results["ts"].predict(X)
    results["c1"] = results["ts"].coef_[0]
    results["c2"] = results["ts"].coef_[1]
    results["mag"] = np.sqrt((results["c1"] ** 2 + results["c2"] ** 2))

    results["dcor"] = distance_correlation(singles_expr, double_expr)
    results["dcor_singles"] = distance_correlation(first_expr, second_expr)
    results["dcor_first"] = distance_correlation(first_expr, double_expr)
    results["dcor_second"] = distance_correlation(second_expr, double_expr)
    results["corr_fit"] = np.corrcoef(Zts.flatten(), double_expr.flatten())[0, 1]
    results["dominance"] = np.abs(np.log10(results["c1"] / results["c2"]))
    results["eq_contr"] = np.min([results["dcor_first"], results["dcor_second"]]) / np.max(
        [results["dcor_first"], results["dcor_second"]]
    )
