from typing import Dict, List, Optional, Set
import numpy as np
from sklearn.cluster import KMeans
import logging
from .mapping_strategies import BaseMappingStrategy

logger = logging.getLogger(__name__)

class ClusteringMappingStrategy(BaseMappingStrategy):
    """
    Maps perturbed cells to control cells by clustering control cells and randomly sampling 
    from the largest cluster. Clustering is done per cell type and split.
    """
    def __init__(
        self, 
        name="clustering", 
        random_state=42, 
        n_basal_samples=1, 
        n_clusters=50, 
        **kwargs
    ):
        super().__init__(name, random_state, n_basal_samples)
        self.n_clusters = n_clusters
        
        # For each split, store the indices for the largest cluster
        self.split_control_indices = {
            'train': {},
            'train_eval': {},
            'val': {},
            'test': {}
        }

    def register_split_indices(
        self,
        dataset: "PerturbationDataset", 
        split: str,
        perturbed_indices: np.ndarray,
        control_indices: np.ndarray
    ):
        """
        For a given split, cluster the control cells and store indices from the largest cluster.
        Clustering is done per cell type.
        """
        if len(control_indices) == 0:
            return
        control_indices = np.sort(control_indices)

        # Get expression matrix for control cells
        if dataset.embed_key:
            X = dataset.h5_file[f'obsm/{dataset.embed_key}'][:]
        else:
            X = []
            for idx in control_indices:
                X.append(dataset.fetch_expression(idx))
            X = np.vstack(X)

        # Cluster the control cells
        kmeans = KMeans(
            n_clusters=self.n_clusters, 
            random_state=self.random_state
        )
        cluster_labels = kmeans.fit_predict(X)
        
        # Find the largest cluster
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        largest_cluster = unique_labels[np.argmax(counts)]
        largest_cluster_mask = cluster_labels == largest_cluster
        
        # Store indices from largest cluster
        self.split_control_indices[split] = control_indices[largest_cluster_mask]
        
        n_controls = len(self.split_control_indices[split])
        logger.info(
            f"[{dataset.name}] Split {split} has {n_controls} control cells in largest cluster"
        )

    def get_control_indices(
        self,
        dataset: "PerturbationDataset",
        split: str, 
        perturbed_idx: int
    ) -> np.ndarray:
        """
        Sample control cells from the largest cluster for this split.
        """
        control_pool = self.split_control_indices.get(split, [])
        if len(control_pool) == 0:
            raise ValueError(f"No control cells found for split {split}")
        
        # Randomly sample from the largest cluster
        return self.rng.choice(
            control_pool,
            size=self.n_basal_samples,
            replace=True
        )
