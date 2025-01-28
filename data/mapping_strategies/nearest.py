import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from .mapping_strategies import BaseMappingStrategy
from ..transforms import PCATransform


class NearestNeighborMappingStrategy(BaseMappingStrategy):
    """
    Maps a perturbed cell to one of its k nearest neighbor controls, sampled with
    probability proportional to proximity.

    We use scikit-learn's NearestNeighbors to find k nearest controls for each
    perturbed cell, then store both indices and normalized weights for sampling.
    """

    def __init__(
        self,
        name="nearest",
        random_state=42,
        n_basal_samples=1,
        k_neighbors=10,
        pca_transform: PCATransform = None,
        **kwargs,
    ):
        super().__init__(name, random_state, n_basal_samples)

        # if k_neighbors is not an int, set it to a default value
        # this is done to help enforce train / eval map separation,
        # though we prolly can find a better solution.
        self.k_neighbors = k_neighbors if isinstance(k_neighbors, int) else 10

        # to train using a PCA embedding, need to store the PCATransform object
        self.pca_transform = pca_transform  # TODO-Abhi: huh, maybe this isn't needed.

        # For each split, store neighbor indices and weights for each cell
        # (n_cells, k_neighbors) for both indices and weights
        self.split_neighbor_indices = {"train": None, "train_eval": None, "val": None, "test": None}
        self.split_neighbor_weights = {"train": None, "train_eval": None, "val": None, "test": None}
        self.split_index_lookup = {"train": None, "train_eval": None, "val": None, "test": None}

    def register_split_indices(
        self,
        dataset: "PerturbationDataset",  # a reference to the PerturbationDataset
        split: str,
        perturbed_indices: np.ndarray,
        control_indices: np.ndarray,
    ):
        """
        For each perturbed cell in the split, find its k nearest control neighbors
        and store both the indices and normalized weights for sampling.
        """

        if len(perturbed_indices) == 0 and len(control_indices) == 0:
            return

        all_split_indices = np.sort(np.concatenate([perturbed_indices, control_indices]))
        self.split_index_lookup[split] = all_split_indices

        # Now just store a single neighbor index per cell
        n_cells = len(all_split_indices)
        self.split_neighbor_indices[split] = np.zeros((n_cells, self.k_neighbors), dtype=np.int64)
        self.split_neighbor_weights[split] = np.zeros((n_cells, self.k_neighbors), dtype=np.float32)

        # Get expression matrix
        if dataset.embed_key:
            X = dataset.h5_file[f"obsm/{dataset.embed_key}"][:]
        else:
            all_x = []
            for idx in range(len(dataset)):
                all_x.append(dataset.fetch_gene_expression(idx))
            X = torch.vstack(all_x)

        control_mask = dataset.control_mask

        # Build neighbors per cell type
        for ct in np.unique(dataset.cell_type_categories):
            ct_code = np.where(dataset.cell_type_categories == ct)[0][0]
            ct_mask = (
                dataset.h5_file[f"obs/{dataset.cell_type_key}/codes"][:] == ct_code
            )  # this will just be the whole file?

            ct_indices = all_split_indices[ct_mask[all_split_indices]]
            if len(ct_indices) == 0:
                continue

            ct_controls = ct_indices[control_mask[ct_indices]]
            ct_perturbed = ct_indices[~control_mask[ct_indices]]
            if len(ct_perturbed) == 0 or len(ct_controls) == 0:
                continue

            # Find k nearest neighbors (k=10 like in TransitionDataset)
            n_nbrs = min(self.k_neighbors, len(ct_controls))
            nn_model = NearestNeighbors(n_neighbors=n_nbrs)
            nn_model.fit(X[ct_controls])

            # Get neighbor indices for each perturbed cell
            distances, neighbors = nn_model.kneighbors(X[ct_perturbed])

            # Get neighbor weights for each perturbed cell
            # weights = np.exp(-distances)  # Convert distance to similarity
            # weights = weights / weights.sum(axis=1, keepdims=True)  # Normalize
            weights = np.ones((len(ct_perturbed), n_nbrs)) / n_nbrs

            # Store results for each perturbed cell
            for i, pert_idx in enumerate(ct_perturbed):
                # Find where this perturbed cell lives in the all_split_indices array
                row_idx = np.where(all_split_indices == pert_idx)[0][0]

                # Store neighbor indices (converting from neighbor indices to global indices)
                neighbor_indices = ct_controls[neighbors[i]]
                self.split_neighbor_indices[split][row_idx, :n_nbrs] = neighbor_indices

                # Store sampling weights
                self.split_neighbor_weights[split][row_idx, :n_nbrs] = weights[i]

        # For control cells, they map to themselves
        control_locs = np.where(control_mask[all_split_indices])[0]
        for idx in control_locs:
            self.split_neighbor_indices[split][idx, 0] = all_split_indices[idx]
            self.split_neighbor_weights[split][idx, 0] = 1.0

    def get_control_indices(self, dataset: "PerturbationDataset", split: str, perturbed_idx: int) -> np.ndarray:
        """
        Sample a control cell index based on stored neighbor weights.
        """
        # Find this cell's location in the split's index array
        all_split_indices = self.split_index_lookup[split]
        row_idx = np.where(all_split_indices == perturbed_idx)[0]
        if len(row_idx) == 0:
            raise ValueError(f"Perturbed index {perturbed_idx} not found in split {split}")

        row_idx = row_idx[0]

        # Get valid neighbors and their weights
        neighbor_indices = self.split_neighbor_indices[split][row_idx]

        # TODO-Abhi: Ablate uniform sampling here?
        weights = self.split_neighbor_weights[split][row_idx]

        # Sample based on weights
        sampled_idx = self.rng.choice(neighbor_indices, size=self.n_basal_samples, p=weights, replace=True)

        return sampled_idx
