import numpy as np
from sklearn.neighbors import NearestNeighbors
from .mapping_strategies import BaseMappingStrategy


class PseudoNearestMappingStrategy(BaseMappingStrategy):
    """
    A strategy that *does not* use the actual perturbed cell's embedding to
    pick the control. Instead, we:
      1) Precompute a global control mean (global_basal) across training data.
      2) Precompute pert_mean_offsets[pert_name] as the (mean(pert) - global_basal).
      3) For a given perturbed cell at inference:
         - Identify its perturbation name
         - Build a 'pseudo' embedding = global_basal + offset
         - Then find the nearest real control cell in the chosen embedding space
         - Return that control cell index.
    """

    def __init__(
        self,
        name="pseudo_nearest",
        random_state=42,
        n_basal_samples=1,
        global_basal=None,  # shape [embedding_dim]
        pert_mean_offsets=None,  # dict: {pert_name -> np.ndarray([embedding_dim])}
        embed_key="X_uce",
        use_gene_space=False,
        **kwargs,
    ):
        """
        Args:
            global_basal (np.ndarray): The global control mean in embedding dimension.
            pert_mean_offsets (dict[str->np.ndarray]): Offsets for each perturbation.
            embed_key (str): Name of the embedding key in the dataset (e.g. 'X_uce' or 'X_scGPT').
            use_gene_space (bool): If True, we'll fetch raw gene expression instead of embed_key.
            random_state, n_basal_samples: From BaseMappingStrategy.
        """
        super().__init__(name, random_state, n_basal_samples)
        self.global_basal = global_basal
        self.pert_mean_offsets = pert_mean_offsets or {}
        self.embed_key = embed_key
        self.use_gene_space = use_gene_space

        # We'll store for each split:
        #   - the control indices
        #   - a NearestNeighbors model for those control embeddings
        self.split_control_indices = {"train": None, "train_eval": None, "val": None, "test": None}
        self.split_nn_model = {"train": None, "train_eval": None, "val": None, "test": None}

    def register_split_indices(
        self,
        dataset: "PerturbationDataset",
        split: str,
        perturbed_indices: np.ndarray,
        control_indices: np.ndarray,
    ):
        """Build a nearest-neighbor model for the control cells in this split."""
        if len(control_indices) == 0:
            return

        # Cache
        self.split_control_indices[split] = control_indices

        # Gather control embeddings
        control_embeddings = []
        for c_idx in control_indices:
            if self.use_gene_space:
                arr = dataset.fetch_gene_expression(c_idx).cpu().numpy()
            else:
                arr = dataset.fetch_obsm_expression(c_idx, self.embed_key).cpu().numpy()
            control_embeddings.append(arr)
        control_embeddings = np.vstack(control_embeddings)

        # Build a NearestNeighbors model
        nn_model = NearestNeighbors(n_neighbors=self.n_basal_samples)
        nn_model.fit(control_embeddings)
        self.split_nn_model[split] = {"model": nn_model, "embeddings": control_embeddings}

    def get_control_indices(self, dataset: "PerturbationDataset", split: str, perturbed_idx: int) -> np.ndarray:
        """
        Synthesize the 'pseudo' embedding from global_basal + offset, find nearest
        neighbor in the split's control set, return that index (or multiple if n_basal_samples>1).
        """
        if self.split_nn_model[split] is None:
            raise ValueError(f"No nearest-neighbor model found for split={split}")

        # 1) Identify the perturbation name
        #    We rely on the dataset.pert_categories and dataset.pert_col
        pert_code = dataset.h5_file[f"obs/{dataset.pert_col}/codes"][perturbed_idx]
        pert_name = dataset.pert_categories[pert_code]

        # 2) Synthesize the pseudo embedding
        offset_vec = self.pert_mean_offsets.get(pert_name, np.zeros_like(self.global_basal))
        pseudo_emb = self.global_basal + offset_vec  # shape [dim]

        # 3) Nearest neighbor
        nn_model = self.split_nn_model[split]["model"]
        dist, nbrs = nn_model.kneighbors([pseudo_emb])  # shape (1,n_basal_samples)

        ctrl_array_indices = nbrs[0]
        # Map from these array indices to the actual dataset indices
        actual_indices = self.split_control_indices[split][ctrl_array_indices]
        return actual_indices
