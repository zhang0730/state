from typing import Dict, List, Optional, Set
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import logging
from .mapping_strategies import BaseMappingStrategy

logger = logging.getLogger(__name__)

class CentroidMappingStrategy(BaseMappingStrategy):
    """
    Maps perturbed cells to control cells by clustering control cells and sampling
    from the cluster centroids. For each cluster, identifies the control cell 
    closest to its centroid.
    """
    def __init__(
        self, 
        name="centroid", 
        random_state=42, 
        n_basal_samples=1,
        n_clusters=50,
        **kwargs
    ):
        super().__init__(name, random_state, n_basal_samples)
        self.n_clusters = n_clusters
        
        # For each split, store the indices for centroid cells
        self.split_centroid_indices = {
            'train': [],
            'train_eval': [],
            'val': [],
            'test': []
        }

    def register_split_indices(
        self,
        dataset: "PerturbationDataset",
        split: str,
        perturbed_indices: np.ndarray,
        control_indices: np.ndarray
    ):
        """
        For a given split, cluster the control cells and find the cells closest 
        to each centroid. These become our control cell pool.
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
        kmeans.fit(X)
        
        # Find the control cell closest to each centroid
        centroid_indices, _ = pairwise_distances_argmin_min(
            kmeans.cluster_centers_, 
            X
        )
        
        # Map back to original indices
        self.split_centroid_indices[split] = control_indices[centroid_indices]
        
        logger.info(
            f"[{dataset.name}] Split {split} has {len(centroid_indices)} centroid cells"
        )

    def get_control_indices(
        self,
        dataset: "PerturbationDataset",
        split: str,
        perturbed_idx: int
    ) -> np.ndarray:
        """
        Sample control cells uniformly from the centroid cells for this split.
        """
        centroid_pool = self.split_centroid_indices.get(split, [])
        if len(centroid_pool) == 0:
            raise ValueError(f"No centroid cells found for split {split}")
        
        # Randomly sample from the centroid cells
        return self.rng.choice(
            centroid_pool,
            size=self.n_basal_samples,
            replace=True
        )
